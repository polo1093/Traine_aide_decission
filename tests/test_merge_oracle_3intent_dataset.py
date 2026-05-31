from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.merge_oracle_3intent_dataset import (
    FEATURE_COLUMNS,
    merge_oracle_3intent_dataset,
    stage_group_for_row,
)


def feature_row(index: int, *, street: str, board_count: float, label: str, hand_id: str | None = None) -> dict[str, object]:
    row = {
        "snapshot_id": f"fixture:{index}",
        "split": "train",
        "hand_id": hand_id,
        "bootstrap_label": label,
    }
    for feature in FEATURE_COLUMNS:
        if feature == "metadata.street":
            row[feature] = street
        elif feature == "features.hero_position":
            row[feature] = "BB"
        elif feature == "features.board_card_count":
            row[feature] = board_count
        elif feature.startswith("features.has_"):
            row[feature] = 1.0
        else:
            row[feature] = float(index % 7 + 1)
    return row


def write_fixture_pokerbench(path: Path) -> None:
    rows = []
    labels = ["NO_INVEST", "CALL", "RAISE"]
    for index in range(45):
        rows.append(feature_row(index, street="PREFLOP", board_count=0.0, label=labels[index % 3], hand_id=f"pre_{index // 2}"))
    for index in range(45, 90):
        rows.append(feature_row(index, street="FLOP", board_count=3.0, label=labels[index % 3], hand_id=f"post_{index // 2}"))
    rows.append(feature_row(91, street="UNKNOWN", board_count=float("nan"), label="CALL", hand_id="unknown_1"))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stage_group_rules() -> None:
    assert stage_group_for_row(pd.Series({"metadata.street": "RIVER", "features.board_card_count": 0})) == "POSTFLOP"
    assert stage_group_for_row(pd.Series({"metadata.street": "UNKNOWN", "features.board_card_count": 0})) == "PREFLOP"
    assert stage_group_for_row(pd.Series({"metadata.street": "FLOP", "features.board_card_count": 0})) == "POSTFLOP"
    assert stage_group_for_row(pd.Series({"metadata.street": "UNKNOWN", "features.board_card_count": 3})) == "POSTFLOP"
    assert stage_group_for_row(pd.Series({"metadata.street": "UNKNOWN", "features.board_card_count": None})) == "UNKNOWN"


def test_merge_schema_antileakage_unknown_exclusion_and_hand_split(tmp_path: Path) -> None:
    source = tmp_path / "pokerbench_model_input.csv"
    output_dir = tmp_path / "merged"
    write_fixture_pokerbench(source)

    report = merge_oracle_3intent_dataset(
        output_dir=output_dir,
        force=True,
        input_paths={"pokerbench": source},
        external_feature_paths={},
        external_audit_paths={},
    )

    assert report["unknown_stage_rows_excluded"] == 1
    assert report["stage_group_distribution"]["PREFLOP"] == 45
    assert report["stage_group_distribution"]["POSTFLOP"] == 45

    model_input = pd.read_csv(output_dir / "model_input.csv")
    for column in [*FEATURE_COLUMNS, "label_3intent", "source_dataset", "stage_group"]:
        assert column in model_input.columns

    for stage in ["preflop", "postflop"]:
        for split in ["train", "validation", "test"]:
            x = pd.read_csv(output_dir / f"X_{split}_{stage}.csv")
            assert list(x.columns) == list(FEATURE_COLUMNS)
            assert "source_dataset" not in x.columns
            assert "label_3intent" not in x.columns
            assert "raw_prompt" not in x.columns

    split_by_hand = model_input[model_input["hand_id"].notna()].groupby(["source_dataset", "hand_id"])["split"].nunique()
    assert int(split_by_hand.max()) == 1
