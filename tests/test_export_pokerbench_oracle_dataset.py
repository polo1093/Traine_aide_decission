from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets.export_pokerbench_oracle_dataset import export_pokerbench_oracle_dataset
from experiments.pokerbench_oracle_3intent_v1 import INTENT_LABELS
from experiments.pokerbench_oracle_baseline_v1 import FEATURE_COLUMNS


def write_pokerbench_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    preflop = data_dir / "preflop_60k_train_set_game_scenario_information.csv"
    postflop = data_dir / "postflop_10k_test_set_game_scenario_information.csv"

    with preflop.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["", "prev_line", "hero_pos", "hero_holding", "correct_decision", "num_players", "num_bets", "available_moves", "pot_size"],
        )
        writer.writeheader()
        labels = ["check", "fold", "call", "10.0bb"] * 12
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "": index,
                    "prev_line": "HJ/2.0bb/CO/call/BTN/10.0bb",
                    "hero_pos": "HJ",
                    "hero_holding": "AhKc",
                    "correct_decision": label,
                    "num_players": 4,
                    "num_bets": 2,
                    "available_moves": "['call', 'fold', 'allin']",
                    "pot_size": 25.0 + index,
                }
            )

    with postflop.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "preflop_action",
                "board_flop",
                "board_turn",
                "board_river",
                "aggressor_position",
                "postflop_action",
                "evaluation_at",
                "available_moves",
                "pot_size",
                "hero_position",
                "holding",
                "correct_decision",
            ],
        )
        writer.writeheader()
        labels = ["Check", "Fold", "Call", "Raise 29"] * 12
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "": index,
                    "preflop_action": "CO/2.3bb/BTN/call",
                    "board_flop": "QcAdKh",
                    "board_turn": "Ts",
                    "board_river": "Tc",
                    "aggressor_position": "OOP",
                    "postflop_action": "OOP_CHECK/IP_BET_3/IP_RAISE_11",
                    "evaluation_at": "River",
                    "available_moves": "['Fold', 'Call', 'Raise 29']",
                    "pot_size": 20.0 + index,
                    "hero_position": "OOP",
                    "holding": "Kc5c",
                    "correct_decision": label,
                }
            )


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def test_export_pokerbench_oracle_dataset_writes_feature_aligned_3intent_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "pokerbench"
    output_dir = tmp_path / "pokerbench_oracle_dataset_v1"
    write_pokerbench_fixture(data_dir)

    report = export_pokerbench_oracle_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        label_mode="3intent",
        download=False,
        random_seed=17,
    )

    assert report["status"] == "ok"
    assert report["label_mode"] == "3intent"
    assert report["allowed_labels"] == list(INTENT_LABELS)
    assert report["label_distribution"]["NO_INVEST"] == 48
    assert report["label_distribution"]["CALL"] == 24
    assert report["label_distribution"]["RAISE"] == 24
    assert report["leakage_columns_used_by_x"] == []
    assert report["x_files_exclude_label_audit_and_raw_text"] is True
    assert set(report["split_counts"]) == {"train", "validation", "test"}

    x_header = read_header(output_dir / "X_train.csv")
    assert x_header == list(FEATURE_COLUMNS)
    assert "bootstrap_label" not in x_header
    assert "pokerbench.correct_decision_raw" not in x_header
    assert not any(column.startswith(("labels.", "debug.", "audit.")) for column in x_header)

    y_header = read_header(output_dir / "y_train.csv")
    assert y_header == ["snapshot_id", "bootstrap_label"]
    assert (output_dir / "model_input.csv").exists()
    assert (output_dir / "audit_candidates.csv").exists()
    assert (output_dir / "dataset.jsonl").exists()
    assert (output_dir / "dataset_card.md").exists()

    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["features_model_used"] == list(FEATURE_COLUMNS)
    assert contract["leakage_columns_used_by_x"] == []
