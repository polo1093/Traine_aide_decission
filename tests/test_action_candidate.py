from __future__ import annotations

from typing import Any

from solver_jobs.action_candidate import build_solver_action_candidate


def strategy_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "solver_job_id": "job-candidate",
        "solver_status": "ok",
        "root_player_role": "hero",
        "action_frequencies": {"CHECK": 0.7, "BET_33": 0.2, "ALL_IN": 0.1},
        "dominant_action": "CHECK",
        "dominant_action_frequency": 0.7,
        "confidence": "medium",
        "iterations": 25,
        "exploitability_last": 0.1,
    }
    payload.update(overrides)
    return payload


def assert_not_label(result: dict[str, Any]) -> None:
    assert "training_label" not in result
    assert "gto_label" not in result
    assert result["is_training_label"] is False
    assert result["label_quality"] == "solver_candidate_untrusted"


def test_root_hero_strong_dominant_frequency_returns_candidate() -> None:
    result = build_solver_action_candidate(strategy_payload())

    assert result["status"] == "ok"
    assert result["solver_job_id"] == "job-candidate"
    assert result["candidate_action"] == "CHECK"
    assert result["candidate_frequency"] == 0.7
    assert result["candidate_confidence"] == "medium"
    assert result["exclusion_reason"] is None
    assert_not_label(result)


def test_root_non_hero_is_refused() -> None:
    result = build_solver_action_candidate(strategy_payload(root_player_role="villain"))

    assert result["status"] == "failed"
    assert result["candidate_action"] is None
    assert result["exclusion_reason"] == "root_not_hero"
    assert_not_label(result)


def test_strategy_absent_is_refused() -> None:
    result = build_solver_action_candidate({"solver_job_id": "job-missing", "solver_status": "ok", "iterations": 25})

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "strategy_not_available"
    assert_not_label(result)


def test_weak_dominant_frequency_is_refused() -> None:
    result = build_solver_action_candidate(
        strategy_payload(
            action_frequencies={"CHECK": 0.55, "BET_33": 0.45},
            dominant_action_frequency=0.55,
            confidence="medium",
        )
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "dominant_action_too_weak"
    assert_not_label(result)


def test_iterations_too_low_is_refused() -> None:
    result = build_solver_action_candidate(strategy_payload(iterations=24))

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "iterations_too_low"
    assert_not_label(result)


def test_solver_status_failed_is_refused() -> None:
    result = build_solver_action_candidate(strategy_payload(solver_status="failed"))

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "solver_not_ok"
    assert_not_label(result)


def test_invalid_frequencies_are_refused() -> None:
    result = build_solver_action_candidate(
        strategy_payload(action_frequencies={"CHECK": 1.2}, dominant_action_frequency=1.2)
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "invalid_frequencies"
    assert_not_label(result)


def test_extracted_root_not_hero_is_mapped_to_root_not_hero() -> None:
    run_result = {
        "solver_job_id": "job-run",
        "solver_status": "ok",
        "solver_result": {
            "status": "ok",
            "output": {
                "root_strategy_raw": {
                    "root_player": 1,
                    "hero_solver_player": 0,
                    "root_matches_hero": False,
                    "root_player_role": "villain",
                    "action_labels": ["CHECK", "BET_33"],
                    "frequencies": [0.7, 0.3],
                },
                "iterations": 25,
            },
        },
        "quality": {"iterations": 25},
    }

    result = build_solver_action_candidate(run_result)

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "root_not_hero"
    assert_not_label(result)


def test_candidate_from_full_run_result() -> None:
    run_result = {
        "solver_job_id": "job-run",
        "solver_status": "ok",
        "solver_result": {
            "status": "ok",
            "output": {
                "iterations": 25,
                "root_strategy_raw": {
                    "root_player": 0,
                    "hero_solver_player": 0,
                    "root_matches_hero": True,
                    "root_player_role": "hero",
                    "action_labels": ["FOLD", "CALL", "ALL_IN"],
                    "frequencies": [0.8, 0.1, 0.1],
                },
            },
        },
        "quality": {"iterations": 25, "exploitability_last": 0.1},
    }

    result = build_solver_action_candidate(run_result)

    assert result["status"] == "ok"
    assert result["candidate_action"] == "FOLD"
    assert result["candidate_frequency"] == 0.8
    assert result["candidate_confidence"] == "high"
    assert_not_label(result)


def test_is_training_label_is_always_false_even_when_input_claims_true() -> None:
    result = build_solver_action_candidate(strategy_payload(is_training_label=True, training_label="CHECK", gto_label="CHECK"))

    assert result["is_training_label"] is False
    assert "training_label" not in result
    assert "gto_label" not in result
