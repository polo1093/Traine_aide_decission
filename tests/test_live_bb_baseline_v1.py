from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.live_bb_baseline_v1 import (
    ALLOWED_LABELS,
    normalize_label,
    predict_live_bb_model,
    run_live_bb_baseline_v1,
)


def row(label: str, index: int) -> dict:
    is_call = label in {"CALL", "COL"}
    is_raise = label in {"RAISE", "RES"}
    is_fold = label == "FOLD"
    is_check = label == "CHECK"
    pot = 80.0 + index * 2
    to_call = 20.0 if is_call or is_fold else 0.0
    equity = 0.28 if is_fold else 0.42 if is_call else 0.78 if is_raise else 0.5
    return {
        "type": "ml_decision_snapshot",
        "snapshot_id": f"snap_{index}",
        "features": {
            "amount_unit": "big_blind",
            "amount_unit_value": 20.0,
            "amount_unit_source": "test",
            "pot": pot,
            "pot_bb": pot / 20.0,
            "to_call": to_call,
            "to_call_bb": to_call / 20.0,
            "to_call_pot_ratio": to_call / pot if pot else 0.0,
            "equity_table": equity,
            "equity_1v1": equity,
            "equity_required": to_call / (pot + to_call) if to_call else 0.0,
            "ev": equity * (pot + to_call) - to_call,
            "ev_bb": (equity * (pot + to_call) - to_call) / 20.0,
            "call_max": equity * pot,
            "call_max_bb": equity * pot / 20.0,
            "has_check": is_check or is_raise,
            "has_call": is_call or is_fold,
            "has_raise": True,
            "hero_position": "IP" if is_call or is_fold else "OOP",
            "player_active": 3,
            "players": [
                {"name": "Flael", "active": True, "stack_bb": 40.0 + index},
                {"name": "Villain", "active": True, "stack_bb": 60.0},
            ],
            "hero_cards": ["Ah", "As"],
            "board_cards": ["2c", "3d", "4h"],
        },
        "labels": {
            "final_action": label,
            "legacy_action": label,
            "label_valid": True,
            "known_bug_risk": False,
        },
        "metadata": {
            "street": "FLOP" if index % 2 else "PREFLOP",
            "label_source": "legacy",
        },
        "quality_flags": {
            "usable_for_training": True,
            "amount_unit_missing": False,
        },
        "debug": {"decision_reason": "not_a_feature"},
    }


def write_jsonl(path: Path) -> None:
    labels = ["CHECK"] * 10 + ["FOLD"] * 10 + ["COL"] * 10 + ["RES"] * 10
    path.write_text("\n".join(json.dumps(row(label, i), ensure_ascii=False) for i, label in enumerate(labels)) + "\n", encoding="utf-8")


def write_prefop_only_jsonl(path: Path) -> None:
    labels = ["CHECK"] * 10 + ["FOLD"] * 10 + ["COL"] * 10 + ["RES"] * 10
    rows = []
    for index, label in enumerate(labels):
        payload = row(label, index)
        payload["metadata"]["street"] = "PREFLOP"
        rows.append(payload)
    path.write_text("\n".join(json.dumps(payload, ensure_ascii=False) for payload in rows) + "\n", encoding="utf-8")


def write_river_jsonl(path: Path) -> None:
    labels = ["CHECK"] * 10 + ["FOLD"] * 10 + ["COL"] * 10 + ["RES"] * 10
    rows = []
    for index, label in enumerate(labels):
        payload = row(label, index)
        payload["metadata"]["street"] = "RIVER"
        payload["features"]["board_cards"] = ["2c", "3d", "4h", "5s", "6c"]
        rows.append(payload)
    path.write_text("\n".join(json.dumps(payload, ensure_ascii=False) for payload in rows) + "\n", encoding="utf-8")


def test_label_normalization_supports_call_and_raise_aliases() -> None:
    assert normalize_label("COL") == "CALL"
    assert normalize_label("RES") == "RAISE"
    assert normalize_label("CALL") == "CALL"
    assert normalize_label("RAISE") == "RAISE"


def test_live_bb_baseline_trains_four_classes_without_leakage_or_overwrite(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "training_dataset.jsonl"
    output_dir = tmp_path / "live_bb_baseline_v1"
    write_jsonl(input_jsonl)

    report = run_live_bb_baseline_v1(input_jsonl=input_jsonl, output_dir=output_dir)

    assert report["status"] == "ok"
    assert report["supported_classes"] == list(ALLOWED_LABELS)
    assert report["generation_mode"] == "imported_only"
    assert report["source_snapshot_id_unique_count"] == 40
    assert report["inflation_ratio"] == 1.0
    assert set(report["label_distribution"]) == set(ALLOWED_LABELS)
    assert report["label_distribution"]["CALL"] == 10
    assert report["label_distribution"]["RAISE"] == 10
    assert report["split_strategy"] == "grouped_by_source_snapshot_id_stratified_by_label"
    assert report["bot_live_connection"] == "not_modified"
    assert (output_dir / "candidates.csv").exists()
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "training_report.json").exists()
    assert (output_dir / "feature_contract.json").exists()
    assert (output_dir / "input_feature_correlation_matrix.svg").exists()
    assert (output_dir / "input_feature_correlation_matrix.json").exists()
    assert (output_dir / "high_correlation_pairs.md").exists()
    assert (output_dir / "learning_curve.svg").exists()
    assert (output_dir / "learning_curve_report.md").exists()
    assert (output_dir / "feature_importance.json").exists()
    assert (output_dir / "feature_importance.md").exists()

    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["call_is_native_class"] is True
    assert contract["leakage_columns_used_by_model"] == []
    assert not any(feature.startswith(("labels.", "debug.", "audit.", "quality_flags.")) for feature in contract["features_model_used"])

    result = predict_live_bb_model(model_dir=output_dir, input_payload=row("COL", 99))

    assert result["status"] == "ok"
    assert set(result["labels"]) == set(ALLOWED_LABELS)
    assert result["prediction"] in set(ALLOWED_LABELS)


def test_live_bb_baseline_can_generate_target_rows_without_touching_historical_model(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "training_dataset.jsonl"
    output_dir = tmp_path / "live_bb_baseline_v1_80"
    write_jsonl(input_jsonl)

    report = run_live_bb_baseline_v1(input_jsonl=input_jsonl, output_dir=output_dir, target_rows=80)

    assert report["status"] == "ok"
    assert report["rows_used"] == 80
    assert report["train_size"] + report["test_size"] == 80
    assert report["generation_mode"] == "imported_plus_resampling"
    assert report["augmentation"]["applied"] is True
    assert report["augmentation"]["requested_target_rows"] == 80

    lines = (output_dir / "candidates.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 81


def test_solver_requested_without_eligible_rows_is_reported_and_can_be_strict(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "training_dataset.jsonl"
    output_dir = tmp_path / "live_bb_no_solver_eligible"
    write_prefop_only_jsonl(input_jsonl)

    report = run_live_bb_baseline_v1(input_jsonl=input_jsonl, output_dir=output_dir, use_solver=True, solver_rows=2)

    assert report["solver_attempted"] == 0
    assert report["solver_generated_rows"] == 0
    assert report["solver_generation"]["warning"] == "solver_requested_but_no_postflop_rows_with_amounts"
    assert report["solver_connected_real_generation"] is False
    assert "SOLVER_REQUESTED_BUT_ZERO_SOLVER_GENERATED_ROWS" not in report.get("warnings", [])
    assert any(warning.startswith("solver_generated_rows_below_min_solver_rows") for warning in report.get("warnings", []))

    with pytest.raises(RuntimeError):
        run_live_bb_baseline_v1(input_jsonl=input_jsonl, output_dir=tmp_path / "strict", use_solver=True, solver_rows=2, strict_solver=True)
    assert (tmp_path / "strict" / "solver_generation_failure_report.json").exists()


def test_solver_generated_rows_are_distinct_from_resampled_jitter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_jsonl = tmp_path / "training_dataset.jsonl"
    output_dir = tmp_path / "live_bb_solver_generated"
    write_river_jsonl(input_jsonl)

    def fake_build_hero_oriented_solver_job(**kwargs):
        return {
            "status": "ok",
            "job": {
                "solver_job_id": kwargs["solver_job_id"],
                "hero_solver_player": 0 if kwargs["hero_position_model"] == "IP" else 1,
                "iterations": kwargs["iterations"],
            },
        }

    def fake_run_solver_job_subprocess(job, *, timeout_s):
        return {
            "solver_status": "ok",
            "solver_result": {
                "status": "ok",
                "solver_job_id": job["solver_job_id"],
                "output": {
                    "root_strategy_raw": {
                        "hero_solver_player": job["hero_solver_player"],
                        "root_matches_hero": True,
                        "root_player_role": "hero",
                        "action_labels": ["CHECK", "BET_33", "ALL_IN"],
                        "frequencies": [0.8, 0.2, 0.0],
                    }
                },
            },
            "quality": {"iterations": job["iterations"], "is_label_candidate": False},
            "error": None,
        }

    monkeypatch.setattr("experiments.live_bb_baseline_v1.build_hero_oriented_solver_job", fake_build_hero_oriented_solver_job)
    monkeypatch.setattr("experiments.live_bb_baseline_v1.run_solver_job_subprocess", fake_run_solver_job_subprocess)
    monkeypatch.setattr(
        "experiments.live_bb_baseline_v1.compute_equity_hand_vs_hand",
        lambda *args, **kwargs: {"status": "ok", "output": {"hero_equity": 0.6}},
    )

    report = run_live_bb_baseline_v1(input_jsonl=input_jsonl, output_dir=output_dir, use_solver=True, solver_rows=2, target_rows=80)

    assert report["solver_attempted"] == 2
    assert report["solver_success"] == 2
    assert report["solver_generated_rows"] == 2
    assert report["solver_connected_real_generation"] is True
    assert report["generation_mode"] == "imported_plus_solver_plus_resampling"
    assert report["generation_method_distribution"]["solver_generated"] == 2
    assert report["generation_method_distribution"]["resampled_jitter"] > 0

    with pytest.raises(RuntimeError):
        run_live_bb_baseline_v1(
            input_jsonl=input_jsonl,
            output_dir=tmp_path / "strict_solver_too_few",
            use_solver=True,
            solver_rows=2,
            min_solver_rows=3,
            target_rows=80,
            strict_solver=True,
        )
    assert (tmp_path / "strict_solver_too_few" / "solver_generation_failure_report.json").exists()
