from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.train_bootstrap_v5 import (
    build_feature_contract,
    flatten_dist_snapshot,
    load_first_dist_snapshot,
    train_bootstrap_v5,
)
from models.predict_bootstrap_model import predict_bootstrap_model
from models.train_bootstrap_model import FEATURE_COLUMNS, feature_payload


V5_FIELDS = [
    "features.pot",
    "features.to_call",
    "features.to_call_pot_ratio",
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_known",
    "features.equity_required",
    "features.ev",
    "features.call_max",
    "features.player_start",
    "features.player_active",
    "features.board_card_count",
    "features.hero_stack",
    "features.effective_stack",
    "features.has_check",
    "features.has_call",
    "features.has_raise",
    "metadata.street",
    "features.hero_position",
    "features.hero_cards",
    "features.board_cards",
    "labels.final_action",
    "debug.decision_reason",
    "audit.label_source",
    "metadata.label_source",
    "bootstrap_label",
    "excluded",
]


def write_v5_dataset(path: Path) -> None:
    rows = []
    for index, label in enumerate(["CHECK"] * 12 + ["FOLD"] * 12 + ["RAISE"] * 12):
        facing_bet = label == "FOLD"
        rows.append(
            {
                "features.pot": 200 + index,
                "features.to_call": 80 if facing_bet else 0,
                "features.to_call_pot_ratio": 0.4 if facing_bet else 0,
                "features.equity_table": 0.18 if label == "FOLD" else 0.72 if label == "RAISE" else 0.42,
                "features.equity_1v1": 0.28 if label == "FOLD" else 0.84 if label == "RAISE" else 0.52,
                "features.equity_known": True,
                "features.equity_required": 0.2857 if facing_bet else 0,
                "features.ev": -10.0 if label == "FOLD" else 120.0 if label == "RAISE" else 30.0,
                "features.call_max": 44.0 if label == "FOLD" else 600.0 if label == "RAISE" else 145.0,
                "features.player_start": 2,
                "features.player_active": 2,
                "features.board_card_count": 5 if label == "RAISE" else 0,
                "features.hero_stack": 1000,
                "features.effective_stack": 1000,
                "features.has_check": not facing_bet,
                "features.has_call": facing_bet,
                "features.has_raise": True,
                "metadata.street": "RIVER",
                "features.hero_position": "IP" if facing_bet else "OOP",
                "features.hero_cards": '["Ah", "As"]' if label == "RAISE" else "[]",
                "features.board_cards": '["Ac", "Kd", "7s", "2h", "2c"]' if label == "RAISE" else "[]",
                "labels.final_action": label,
                "debug.decision_reason": "must_not_be_feature",
                "audit.label_source": "solver_candidate",
                "metadata.label_source": "solver_candidate",
                "bootstrap_label": label,
                "excluded": False,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V5_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_train_bootstrap_v5_writes_contract_and_excludes_leakage(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model_v5"
    write_v5_dataset(input_csv)

    report = train_bootstrap_v5(input_csv=input_csv, output_dir=output_dir, min_rows=10)

    assert report["status"] == "ok"
    assert set(report["model_feature_columns"]).issubset(set(FEATURE_COLUMNS))
    assert report["candidate_feature_columns"] == list(FEATURE_COLUMNS)
    assert "features.player_start" in report["constant_or_useless_features"]
    assert report["train_test_split_seed"] == 17
    assert report["dataset_size"] == {"total": 36, "usable": 36, "excluded": 0}
    assert report["leakage_columns_used_by_model"] == []
    assert "labels.final_action" in report["leakage_columns_excluded"]
    assert "debug.decision_reason" in report["leakage_columns_excluded"]
    assert "features.hero_cards" in report["specific_feature_audit"]
    assert report["heatmaps"]["v5_input_numeric_by_output_heatmap"].endswith(".svg")
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "feature_contract.json").exists()
    assert (output_dir / "preprocessing_schema.json").exists()
    assert (output_dir / "feature_bias_audit.json").exists()
    assert (output_dir / "ablation_report.json").exists()
    assert (output_dir / "training_report_v5.json").exists()
    assert (output_dir / "v5_input_numeric_by_output_heatmap.svg").exists()
    assert (output_dir / "v5_input_category_by_output_heatmap.svg").exists()

    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["features_model_used"] == report["model_feature_columns"]
    assert "features.hero_cards" in contract["features_audit_only"]
    assert contract["features_leakage_excluded"]
    assert contract["leakage_columns_used_by_model"] == []

    audit = json.loads((output_dir / "feature_bias_audit.json").read_text(encoding="utf-8"))
    assert audit["normalization"] == "none_raw_sklearn_feature_payload_values"
    assert "features.pot" in audit["feature_statistics"]

    ablation = json.loads((output_dir / "ablation_report.json").read_text(encoding="utf-8"))
    assert {variant["name"] for variant in ablation["variants"]} >= {
        "full_current",
        "without_constant_features",
        "solver_candidate_only",
        "weak_rule_bootstrap_only",
    }

    numeric = json.loads((output_dir / "v5_input_numeric_by_output_heatmap.json").read_text(encoding="utf-8"))
    assert numeric["features"] == [
        "features.pot",
        "features.to_call",
        "features.to_call_pot_ratio",
        "features.equity_table",
        "features.equity_1v1",
        "features.equity_known",
        "features.equity_required",
        "features.equity_gap",
        "features.ev",
        "features.call_max",
        "features.player_start",
        "features.player_active",
        "features.active_opponents",
        "features.board_card_count",
        "features.hero_cards_known",
        "features.opponent_looseness_avg",
        "features.opponent_aggression_avg",
        "features.opponent_confidence_avg",
        "features.hero_stack",
        "features.effective_stack",
        "features.stack_to_pot_ratio",
        "features.has_check",
        "features.has_call",
        "features.has_raise",
    ]
    values = [value for row in numeric["matrix"] for value in row["values"].values() if value is not None]
    assert values
    assert all(0 <= value <= 1 for value in values)

    categorical = json.loads((output_dir / "v5_input_category_by_output_heatmap.json").read_text(encoding="utf-8"))
    assert categorical["features"] == ["metadata.street", "features.hero_position"]
    category_values = [value for row in categorical["matrix"] for value in row["values"].values() if value is not None]
    assert category_values
    assert all(0 <= value <= 1 for value in category_values)


def test_real_dist_example_can_be_flattened_and_predicted_offline(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model_v5"
    write_v5_dataset(input_csv)
    train_bootstrap_v5(input_csv=input_csv, output_dir=output_dir, min_rows=10)

    snapshot_path = Path("dist/ml_dataset_export/example_training_dataset.jsonl")
    snapshot = load_first_dist_snapshot(snapshot_path)
    payload = flatten_dist_snapshot(snapshot)
    features = feature_payload(payload)
    contract = build_feature_contract(fieldnames=list(payload))

    assert snapshot["type"] == "ml_decision_snapshot"
    assert set(FEATURE_COLUMNS).issubset(features)
    assert list(features) == list(FEATURE_COLUMNS)
    assert contract["leakage_columns_used_by_model"] == []
    assert not any(column.startswith(("audit.", "debug.", "labels.")) for column in FEATURE_COLUMNS)

    result = predict_bootstrap_model(model_dir=output_dir, input_payload=payload)

    assert result["status"] == "ok"
    assert result["prediction"] in {"CHECK", "FOLD", "RAISE"}
