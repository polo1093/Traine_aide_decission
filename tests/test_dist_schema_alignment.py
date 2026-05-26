from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets.export_candidate_dataset import dist_aligned_candidate_row, export_dist_aligned_candidate_csv
from experiments.dist_schema_alignment import run_alignment
from models.train_bootstrap_model import FEATURE_COLUMNS


def test_dist_aligned_candidate_row_uses_dist_shape_without_leakage_features() -> None:
    row = dist_aligned_candidate_row(
        {
            "source_id": "solver_001",
            "street": "RIVER",
            "hero_cards": '["Ah", "As"]',
            "villain_hand": '["8c", "3d"]',
            "board_cards": '["Ac", "Kd", "7s", "2h", "2c"]',
            "pot": "300",
            "to_call": "0",
            "position_model": "OOP",
            "bootstrap_label": "RAISE",
            "label_source": "solver_candidate",
            "label_quality": "bootstrap_solver_untrusted",
            "dominant_action_frequency": "0.9",
            "excluded": "False",
        }
    )

    assert row["type"] == "ml_decision_snapshot"
    assert row["metadata.street"] == "RIVER"
    assert row["features.pot"] == 300.0
    assert row["features.has_check"] is True
    assert row["features.equity_1v1"] is not None
    assert row["features.equity_table"] is not None
    assert row["features.equity_known"] is True
    assert row["features.board_card_count"] == 5
    assert row["features.ev"] is not None
    assert row["features.call_max"] is not None
    assert row["features.active_opponents"] == 1
    assert row["features.players"]
    assert row["features.opponent_profiles"]
    assert row["labels.final_action"] == "RAISE"
    assert row["quality_flags.usable_for_training"] is True
    assert row["audit.dominant_action_frequency"] == 0.9
    assert "audit.dominant_action_frequency" not in FEATURE_COLUMNS


def test_export_dist_aligned_candidate_csv_writes_expected_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "v5.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "street", "hero_cards", "board_cards", "pot", "to_call", "position_model", "bootstrap_label", "label_source", "excluded"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "solver_001",
                "street": "RIVER",
                "hero_cards": "null",
                "board_cards": "null",
                "pot": "100",
                "to_call": "50",
                "position_model": "IP",
                "bootstrap_label": "FOLD",
                "label_source": "solver_candidate",
                "excluded": "False",
            }
        )

    summary = export_dist_aligned_candidate_csv(input_csv, output_csv)
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8", newline="")))

    assert summary["rows_usable"] == 1
    assert rows[0]["schema_version"] == "ml_dataset_v1"
    assert rows[0]["features.to_call_pot_ratio"] == "0.5"
    assert json.loads(rows[0]["features.buttons_active"]) == ["relance", "paie", "fold"]


def test_run_alignment_reports_v5_alignment_and_model_features(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "example_training_dataset.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "ml_dataset_v1",
                "type": "ml_decision_snapshot",
                "snapshot_id": "snap_1",
                "metadata": {"street": "RIVER", "decision_engine_version": "decision_engine_v2"},
                "features": {"pot": 100.0, "to_call": 0.0, "hero_position": "BB", "buttons": [], "has_check": True},
                "labels": {"final_action": "CHECK", "label_valid": True, "known_bug_risk": False},
                "quality_flags": {"usable_for_training": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    input_csv = tmp_path / "candidates.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "street", "hero_cards", "board_cards", "pot", "to_call", "position_model", "bootstrap_label", "label_source", "excluded"],
        )
        writer.writeheader()
        writer.writerow({"source_id": "solver_001", "street": "RIVER", "pot": "100", "to_call": "0", "position_model": "BB", "bootstrap_label": "CHECK", "label_source": "solver_candidate", "excluded": "False"})
    v5_csv = tmp_path / "v5.csv"
    report_path = tmp_path / "report.json"

    report = run_alignment(dist_dir=dist_dir, current_csv=input_csv, v5_output=v5_csv, report_output=report_path)

    assert report["alignment_status"] == "aligned"
    assert report["model_feature_columns"] == list(FEATURE_COLUMNS)
    assert report["leakage_risk_columns"]["used_by_model"] == []
    assert report_path.exists()
