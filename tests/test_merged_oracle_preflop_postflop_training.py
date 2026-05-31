from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasets.merge_oracle_3intent_dataset import FEATURE_COLUMNS, merge_oracle_3intent_dataset
from experiments.merged_oracle_preflop_postflop_v1 import run_merged_oracle_preflop_postflop_v1


def feature_row(index: int, *, street: str, board_count: float, label: str) -> dict[str, object]:
    row = {
        "snapshot_id": f"fixture:{index}",
        "split": "train",
        "hand_id": f"hand_{index}",
        "bootstrap_label": label,
    }
    for feature in FEATURE_COLUMNS:
        if feature == "metadata.street":
            row[feature] = street
        elif feature == "features.hero_position":
            row[feature] = "SB" if index % 2 else "BB"
        elif feature == "features.board_card_count":
            row[feature] = board_count
        elif feature.startswith("features.has_"):
            row[feature] = 1.0 if index % 2 else 0.0
        else:
            row[feature] = float((index % 11) + 1)
    return row


def write_training_fixture(path: Path) -> None:
    rows = []
    labels = ["NO_INVEST", "CALL", "RAISE"]
    for index in range(90):
        rows.append(feature_row(index, street="PREFLOP", board_count=0.0, label=labels[index % 3]))
    for index in range(90, 180):
        rows.append(feature_row(index, street=["FLOP", "TURN", "RIVER"][index % 3], board_count=3.0 + (index % 3), label=labels[index % 3]))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_minimal_merged_preflop_postflop_training_outputs_reports(tmp_path: Path) -> None:
    source = tmp_path / "pokerbench_model_input.csv"
    data_dir = tmp_path / "merged"
    output_root = tmp_path / "outputs"
    write_training_fixture(source)

    merge_oracle_3intent_dataset(
        output_dir=data_dir,
        force=True,
        input_paths={"pokerbench": source},
        external_feature_paths={},
        external_audit_paths={},
    )
    result = run_merged_oracle_preflop_postflop_v1(data_dir=data_dir, output_root=output_root, force=True)

    assert result["status"] == "ok"
    for name in ["merged_oracle_preflop_model_v1", "merged_oracle_postflop_model_v1"]:
        model_dir = output_root / name
        report_path = model_dir / "training_report.json"
        assert report_path.exists()
        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "confusion_matrix.svg").exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_train"] > 0
        assert report["rows_validation"] > 0
        assert report["rows_test"] > 0
        assert report["leakage_columns_used_by_model"] == []
        assert report["model_path"]
        assert set(report["label_distribution"]) == {"CALL", "NO_INVEST", "RAISE"}
