from __future__ import annotations

from typing import Any

from solver_jobs.eligibility import evaluate_solver_eligibility
from solver_jobs.hero_oriented_builder import validate_hero_root_alignment
from solver_jobs.job_runner import run_solver_job
from solver_jobs.job_schema import validate_solver_job
from synthetic.hero_oriented_spot_generator import generate_hero_oriented_solver_jobs


def generate_one(context: str) -> dict[str, Any]:
    return generate_hero_oriented_solver_jobs(
        count=1,
        seed=42,
        context=context,
        street="RIVER",
        iterations=25,
        timeout_s=5,
        backend="python",
    )[0]


def assert_valid_hero_job(job: dict[str, Any], *, expected_player: int, expected_context: str) -> None:
    validation = validate_solver_job(job)
    root_validation = validate_hero_root_alignment(job)

    assert validation["status"] == "ok", validation
    assert root_validation["status"] == "ok", root_validation
    assert job["source_type"] == "synthetic"
    assert job["generation_profile"] == expected_context
    assert job["hero_solver_player"] == expected_player
    assert job["villain_solver_player"] == 1 - expected_player
    assert job["decision_actor"] == "hero"
    assert job["root_must_be_hero"] is True
    assert root_validation["root_matches_hero"] is True
    assert root_validation["root_player_role"] == "hero"


def test_generate_hero_oop_check_or_bet_root_matches_hero() -> None:
    job = generate_one("hero_oop_check_or_bet")

    assert_valid_hero_job(job, expected_player=1, expected_context="hero_oop_check_or_bet")
    assert job["hero_position_model"] == "OOP"
    assert job["decision_context_type"] == "hero_check_or_bet"
    assert job["initial_hole_cards"][1] == job["hero_hand"]
    assert job["initial_contributions"][0] == job["initial_contributions"][1]


def test_generate_hero_ip_facing_bet_root_matches_hero() -> None:
    job = generate_one("hero_ip_facing_bet")

    assert_valid_hero_job(job, expected_player=0, expected_context="hero_ip_facing_bet")
    assert job["hero_position_model"] == "IP"
    assert job["decision_context_type"] == "hero_facing_bet"
    assert job["initial_hole_cards"][0] == job["hero_hand"]
    assert job["initial_contributions"][0] < job["initial_contributions"][1]


def test_generate_hero_oop_facing_bet_root_matches_hero() -> None:
    job = generate_one("hero_oop_facing_bet")

    assert_valid_hero_job(job, expected_player=1, expected_context="hero_oop_facing_bet")
    assert job["hero_position_model"] == "OOP"
    assert job["decision_context_type"] == "hero_facing_bet"
    assert job["initial_hole_cards"][1] == job["hero_hand"]
    assert job["initial_contributions"][1] < job["initial_contributions"][0]


def test_generated_job_has_no_duplicate_cards() -> None:
    job = generate_one("hero_ip_facing_bet")
    cards = job["hero_hand"] + job["villain_hand"] + job["board"]

    assert len(cards) == len(set(cards))


def test_bad_mapping_is_refused_by_alignment() -> None:
    job = generate_one("hero_oop_check_or_bet")
    job["hero_solver_player"] = 0
    job["villain_solver_player"] = 1
    job["initial_hole_cards"] = [job["hero_hand"], job["villain_hand"]]

    result = validate_hero_root_alignment(job)

    assert result["status"] == "failed"
    assert result["error"] == "root_player_not_hero"


def test_eligibility_accepts_valid_hero_oriented_job() -> None:
    job = generate_one("hero_oop_facing_bet")

    result = evaluate_solver_eligibility(job)

    assert result == {"eligible": True, "reason": None, "warnings": []}


def test_eligibility_refuses_when_root_must_be_hero_absent() -> None:
    job = generate_one("hero_oop_facing_bet")
    del job["root_must_be_hero"]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "root_must_be_hero_required"


def test_eligibility_refuses_when_decision_actor_is_not_hero() -> None:
    job = generate_one("hero_oop_facing_bet")
    job["decision_actor"] = "villain"

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "decision_actor_must_be_hero"


def test_no_training_label_or_label_candidate_created() -> None:
    job = generate_one("hero_ip_facing_bet")

    assert "training_label" not in job
    assert "is_label_candidate" not in job


def test_runner_keeps_is_label_candidate_false(monkeypatch) -> None:
    job = generate_one("hero_ip_facing_bet")

    def fake_solve_tiny_postflop_spot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "solver_name": "PokerSolver",
            "input": {},
            "output": {
                "backend": "python",
                "iterations": 25,
                "game_value": 0.0,
                "exploitability_history": [1.0],
                "strategy_entry_count": 1,
            },
            "error": None,
            "duration_ms": 1.0,
        }

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fake_solve_tiny_postflop_spot)
    result = run_solver_job(job)

    assert result["quality"]["is_label_candidate"] is False
