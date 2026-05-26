from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.advanced_bootstrap_v5_diagnostics import run_advanced_diagnostics
from experiments.train_bootstrap_v5 import flatten_dist_snapshot, load_first_dist_snapshot, train_bootstrap_v5
from models.predict_bootstrap_model import predict_bootstrap_model


FIELDS = [
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


def write_dataset(path: Path) -> None:
    rows = []
    labels = ["CHECK"] * 14 + ["FOLD"] * 14 + ["RAISE"] * 14
    for index, label in enumerate(labels):
        facing_bet = label == "FOLD"
        source = "solver_candidate" if index % 2 == 0 else "weak_rule_bootstrap"
        pot = 180 + index * 4
        to_call = 70 if facing_bet else 20 if label == "RAISE" and index % 3 == 0 else 0
        equity = 0.2 if label == "FOLD" else 0.9 if label == "RAISE" else 0.5
        required = to_call / (pot + to_call) if to_call else 0.0
        rows.append(
            {
                "features.pot": pot,
                "features.to_call": to_call,
                "features.to_call_pot_ratio": to_call / pot if pot else 0,
                "features.equity_table": equity,
                "features.equity_1v1": equity,
                "features.equity_known": True,
                "features.equity_required": required,
                "features.ev": equity * pot - to_call,
                "features.call_max": equity * pot,
                "features.player_start": 2,
                "features.player_active": 2,
                "features.board_card_count": 5,
                "features.hero_stack": 900 + index * 3,
                "features.effective_stack": 900 + index * 3,
                "features.has_check": not facing_bet,
                "features.has_call": facing_bet,
                "features.has_raise": True,
                "metadata.street": "RIVER",
                "features.hero_position": "IP" if facing_bet else "OOP",
                "features.hero_cards": '["Ah", "As"]',
                "features.board_cards": '["Ac", "Kd", "7s", "2h", "2c"]',
                "labels.final_action": label,
                "debug.decision_reason": "must_not_be_feature",
                "audit.label_source": source,
                "metadata.label_source": source,
                "bootstrap_label": label,
                "excluded": False,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_advanced_diagnostics_and_bb_model_do_not_overwrite_current_model(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    current_model_dir = tmp_path / "model_v5_4000"
    bb_dataset = tmp_path / "candidate_v5_4000_bb" / "candidates.csv"
    bb_model_dir = tmp_path / "model_v5_4000_bb"
    write_dataset(input_csv)
    train_bootstrap_v5(input_csv=input_csv, output_dir=current_model_dir, min_rows=10)
    before = sha256(current_model_dir / "model.joblib")

    summary = run_advanced_diagnostics(
        input_csv=input_csv,
        current_model_dir=current_model_dir,
        bb_dataset_csv=bb_dataset,
        bb_model_dir=bb_model_dir,
    )

    assert summary["status"] == "ok"
    assert summary["current_model_unchanged"] is True
    assert sha256(current_model_dir / "model.joblib") == before
    assert (current_model_dir / "input_feature_correlation_matrix.svg").exists()
    assert (current_model_dir / "high_correlation_pairs.md").exists()
    assert (current_model_dir / "learning_curve.svg").exists()
    assert (current_model_dir / "learning_curve_report.md").exists()
    assert (current_model_dir / "multicollinearity_report.md").exists()
    assert (current_model_dir / "robust_generalization_report.md").exists()
    assert bb_dataset.exists()
    assert (bb_model_dir / "model.joblib").exists()
    assert bb_model_dir != current_model_dir
    assert (bb_model_dir / "comparison_current_vs_bb.md").exists()

    bb_rows = list(csv.DictReader(bb_dataset.open("r", encoding="utf-8", newline="")))
    assert "features.pot_bb" in bb_rows[0]
    assert bb_rows[0]["big_blind_missing_or_inferred"] == "big_blind_missing_or_inferred"

    current_contract = json.loads((current_model_dir / "feature_contract.json").read_text(encoding="utf-8"))
    bb_contract = json.loads((bb_model_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert not any(feature.startswith(("audit.", "debug.", "labels.")) for feature in current_contract["features_model_used"])
    assert not any(feature.startswith(("audit.", "debug.", "labels.")) for feature in bb_contract["features_model_used"])
    assert "features.player_start" in current_contract["features_constant_excluded"]
    assert bb_contract["leakage_columns_used_by_model"] == []


def test_real_dist_line_still_predicts_after_advanced_diagnostics_artifacts() -> None:
    snapshot = load_first_dist_snapshot(Path("dist/ml_dataset_export/example_training_dataset.jsonl"))
    payload = flatten_dist_snapshot(snapshot)

    result = predict_bootstrap_model(
        model_dir=Path("outputs/readiness/bootstrap_model_v5_4000"),
        input_payload=payload,
    )

    assert result["status"] == "ok"
    assert result["prediction"] in {"CHECK", "FOLD", "RAISE"}
