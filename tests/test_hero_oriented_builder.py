from __future__ import annotations

from typing import Any

from solver_jobs.hero_oriented_builder import (
    build_hero_oriented_solver_job,
    validate_hero_root_alignment,
)
from solver_jobs.job_runner import run_solver_job
from solver_jobs.strategy_extractor import extract_root_strategy


def build_case(**overrides: Any) -> dict[str, Any]:
    params = {
        "source_snapshot_id": "snapshot_hero_case",
        "street": "FLOP",
        "hero_hand": ["Ah", "Kc"],
        "villain_hand": ["Qd", "Qh"],
        "board": ["As", "7c", "2d"],
        "pot": 10.0,
        "stack": 100.0,
        "hero_position_model": "OOP",
        "decision_context_type": "hero_check_or_bet",
        "to_call": 0.0,
        "bet_sizes": [0.33, 0.75],
        "iterations": 1,
        "timeout_s": 5.0,
        "backend": "python",
        "solver_job_id": "solver_job_hero_case",
        "created_at": "2026-05-25T00:00:00+00:00",
    }
    params.update(overrides)
    result = build_hero_oriented_solver_job(**params)
    assert result["status"] == "ok", result
    return result["job"]


def test_hero_oop_check_or_bet_root_matches_hero() -> None:
    job = build_case(
        solver_job_id="solver_job_hero_oop_check_bet",
        hero_position_model="OOP",
        decision_context_type="hero_check_or_bet",
        to_call=0.0,
    )
    validation = validate_hero_root_alignment(job)

    assert job["hero_solver_player"] == 1
    assert job["villain_solver_player"] == 0
    assert job["initial_hole_cards"] == [["Qd", "Qh"], ["Ah", "Kc"]]
    assert job["initial_contributions"] == [5.0, 5.0]
    assert validation["status"] == "ok"
    assert validation["root_player"] == 1
    assert validation["root_matches_hero"] is True
    assert {"CHECK", "ALL_IN"}.issubset(set(validation["legal_action_labels"]))
    assert any(label.startswith("BET_") for label in validation["legal_action_labels"])


def test_hero_ip_facing_bet_root_matches_hero() -> None:
    job = build_case(
        solver_job_id="solver_job_hero_ip_facing_bet",
        hero_position_model="IP",
        decision_context_type="hero_facing_bet",
        to_call=2.5,
        pot=12.5,
    )
    validation = validate_hero_root_alignment(job)

    assert job["hero_solver_player"] == 0
    assert job["villain_solver_player"] == 1
    assert job["initial_hole_cards"] == [["Ah", "Kc"], ["Qd", "Qh"]]
    assert job["initial_contributions"] == [5.0, 7.5]
    assert validation["root_player"] == 0
    assert validation["root_matches_hero"] is True
    assert {"FOLD", "CALL", "ALL_IN"}.issubset(set(validation["legal_action_labels"]))
    assert any(label.startswith("RAISE_") for label in validation["legal_action_labels"])


def test_hero_oop_facing_bet_root_matches_hero() -> None:
    job = build_case(
        solver_job_id="solver_job_hero_oop_facing_bet",
        hero_position_model="OOP",
        decision_context_type="hero_facing_bet",
        to_call=2.5,
        pot=12.5,
    )
    validation = validate_hero_root_alignment(job)

    assert job["hero_solver_player"] == 1
    assert job["villain_solver_player"] == 0
    assert job["initial_hole_cards"] == [["Qd", "Qh"], ["Ah", "Kc"]]
    assert job["initial_contributions"] == [7.5, 5.0]
    assert validation["root_player"] == 1
    assert validation["root_matches_hero"] is True
    assert {"FOLD", "CALL", "ALL_IN"}.issubset(set(validation["legal_action_labels"]))
    assert any(label.startswith("RAISE_") for label in validation["legal_action_labels"])


def test_bad_mapping_is_refused_before_solver_call(monkeypatch) -> None:
    job = build_case(
        solver_job_id="solver_job_bad_mapping",
        hero_position_model="OOP",
        decision_context_type="hero_check_or_bet",
        to_call=0.0,
    )
    job["hero_solver_player"] = 0
    job["villain_solver_player"] = 1
    job["initial_hole_cards"] = [["Ah", "Kc"], ["Qd", "Qh"]]

    validation = validate_hero_root_alignment(job)
    assert validation["status"] == "failed"
    assert validation["error"] == "root_player_not_hero"

    def fail_if_solver_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("solver must not be called when root is not hero")

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fail_if_solver_called)
    result = run_solver_job(job)

    assert result["status"] == "failed"
    assert result["error"] == "root_player_not_hero"
    assert result["quality"]["is_label_candidate"] is False
    assert result["quality"]["exclusion_reason"] == "root_validation_failed"


def test_runner_passes_oriented_fields_to_adapter(monkeypatch) -> None:
    job = build_case(
        solver_job_id="solver_job_passes_oriented_fields",
        hero_position_model="IP",
        decision_context_type="hero_facing_bet",
        to_call=2.5,
        pot=12.5,
    )
    captured: dict[str, Any] = {}

    def fake_solve_tiny_postflop_spot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "status": "ok",
            "solver_name": "PokerSolver",
            "input": {},
            "output": {
                "backend": "python",
                "iterations": 1,
                "game_value": 0.0,
                "exploitability_history": [],
                "strategy_entry_count": 1,
                "root_strategy_raw": {
                    "root_player": 0,
                    "hero_solver_player": 0,
                    "root_matches_hero": True,
                    "root_player_role": "hero",
                    "action_labels": ["FOLD", "CALL", "ALL_IN"],
                    "frequencies": [0.2, 0.7, 0.1],
                },
            },
            "error": None,
            "duration_ms": 1.0,
        }

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fake_solve_tiny_postflop_spot)
    result = run_solver_job(job)

    assert result["status"] == "ok"
    assert captured["kwargs"]["hero_solver_player"] == 0
    assert captured["kwargs"]["initial_hole_cards"] == [["Ah", "Kc"], ["Qd", "Qh"]]
    assert captured["kwargs"]["initial_contributions"] == [5.0, 7.5]
    assert result["quality"]["is_label_candidate"] is False


def test_strategy_extractor_allows_only_root_hero() -> None:
    ok = {
        "solver_job_id": "job-ok",
        "solver_result": {
            "output": {
                "root_strategy_raw": {
                    "root_player": 0,
                    "hero_solver_player": 0,
                    "root_matches_hero": True,
                    "root_player_role": "hero",
                    "action_labels": ["FOLD", "CALL", "ALL_IN"],
                    "frequencies": [0.2, 0.7, 0.1],
                }
            }
        },
    }
    bad = {
        "solver_job_id": "job-bad",
        "solver_result": {
            "output": {
                "root_strategy_raw": {
                    "root_player": 1,
                    "hero_solver_player": 0,
                    "root_matches_hero": False,
                    "root_player_role": "villain",
                    "action_labels": ["CHECK", "BET_33"],
                    "frequencies": [0.5, 0.5],
                }
            }
        },
    }

    assert extract_root_strategy(ok)["status"] == "ok"
    rejected = extract_root_strategy(bad)
    assert rejected["status"] == "failed"
    assert rejected["error"] == "root_player_not_hero"


def test_no_training_label_or_label_candidate_created() -> None:
    job = build_case()

    assert "training_label" not in job
    assert "is_label_candidate" not in job
