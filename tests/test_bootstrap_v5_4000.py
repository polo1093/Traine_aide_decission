from __future__ import annotations

import json
from pathlib import Path

from experiments import run_bootstrap_v5_4000 as runner


def fake_record(row: dict, *, timeout_s: float, backend: str) -> dict:
    actions = ("CHECK", "FOLD", "BET_33")
    action = actions[row["group_index"] % len(actions)]
    return {
        "record_type": "candidate_sensitivity_result",
        "context": row["context"],
        "scenario": row["scenario"],
        "street": "RIVER",
        "stack": row["stack"],
        "pot": row["pot"],
        "to_call": row["to_call"],
        "to_call_ratio": row["to_call_ratio"],
        "spr": 1.0,
        "bet_size_fractions": row["bet_sizes"],
        "iterations": row["iterations"],
        "backend": backend,
        "solver_job_id": f"test_{row['scenario']}_it{row['iterations']}",
        "solver_status": "ok",
        "root_matches_hero": True,
        "root_player_role": "hero",
        "root_validation_status": "ok",
        "strategy_source": "average_strategy",
        "legal_actions": ["CHECK", "FOLD", "BET_33"],
        "action_frequencies": {action: 0.9, "CHECK": 0.1} if action != "CHECK" else {"CHECK": 0.9, "BET_33": 0.1},
        "dominant_action": action,
        "dominant_frequency": 0.9,
        "candidate_status": "ok",
        "candidate_exclusion_reason": None,
        "exploitability_last": 0.1,
        "danger_flags": [],
        "quality_status": "ok",
        "is_training_label": False,
        "label_quality": "solver_candidate_untrusted",
        "recommendation": "usable_for_candidate_analysis",
        "error": None,
    }


def test_v5_4000_orchestrator_keeps_baseline_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "run_one_solver_candidate_row", fake_record)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    dist_sample = dist_dir / "example_training_dataset.jsonl"
    dist_sample.write_text(
        json.dumps(
            {
                "schema_version": "ml_dataset_v1",
                "type": "ml_decision_snapshot",
                "snapshot_id": "snap",
                "metadata": {"street": "RIVER", "label_source": "legacy"},
                "features": {
                    "pot": 100.0,
                    "to_call": 0.0,
                    "to_call_pot_ratio": 0.0,
                    "equity_required": 0.0,
                    "player_start": 2,
                    "player_active": 2,
                    "has_check": True,
                    "has_call": False,
                    "has_raise": True,
                    "hero_position": "OOP",
                    "hero_cards": [],
                    "board_cards": [],
                },
                "labels": {"final_action": "CHECK", "label_valid": True, "known_bug_risk": False},
                "quality_flags": {"usable_for_training": True},
                "debug": {"decision_reason": "example"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_model_dir = tmp_path / "baseline_model_v5"
    baseline_model_dir.mkdir()
    baseline_report = baseline_model_dir / "training_report_v5.json"
    baseline_report.write_text(
        json.dumps({"dataset_size": {"total": 1, "usable": 1, "excluded": 0}, "label_distribution": {"CHECK": 1}}),
        encoding="utf-8",
    )
    before = baseline_report.read_text(encoding="utf-8")

    result = runner.run_bootstrap_v5_4000(
        output_dir=tmp_path / "candidate_v5_4000",
        model_dir=tmp_path / "model_v5_4000",
        baseline_model_dir=baseline_model_dir,
        target_solves=6,
        min_usable_rows=12,
        class_floor=2,
        min_training_rows=6,
        dist_dir=dist_dir,
        dist_sample=dist_sample,
    )

    assert result["status"] == "ok"
    assert result["dataset_report"]["root_player_not_hero_errors"] == 0
    assert result["dataset_report"]["critical_warnings"] == []
    assert result["training"]["leakage_columns_used_by_model"] == []
    assert result["dist_sample_prediction"]["prediction_status"] == "ok"
    assert "dataset_report_solver" not in result["dataset_report"]["outputs"]
    legacy_marker = "v" + "4_"
    assert legacy_marker not in (tmp_path / "candidate_v5_4000" / "candidate_sensitivity_results.jsonl").read_text(
        encoding="utf-8"
    )
    assert legacy_marker not in (tmp_path / "candidate_v5_4000" / "candidates.csv").read_text(encoding="utf-8")
    assert Path(result["outputs"]["candidates_csv"]).exists()
    assert Path(result["outputs"]["model"]).exists()
    assert Path(result["outputs"]["v5_input_numeric_by_output_heatmap"]).exists()
    assert Path(result["outputs"]["v5_input_category_by_output_heatmap"]).exists()
    assert Path(result["outputs"]["comparison_v5_vs_v5_4000"]).exists()
    assert baseline_report.read_text(encoding="utf-8") == before
