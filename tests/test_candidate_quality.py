from __future__ import annotations

from typing import Any

from solver_jobs.candidate_quality import evaluate_candidate_quality


def run_payload(
    *,
    action: str = "CHECK",
    frequency: float = 0.7,
    solver_status: str = "ok",
    root_role: str = "hero",
    iterations: int = 25,
    exploitability_last: float | None = 0.1,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {
        "solver_job_id": "job-quality",
        "solver_status": solver_status,
        "root_player_role": root_role,
        "root_matches_hero": root_role == "hero",
        "action_candidate": {
            "status": "ok",
            "solver_job_id": "job-quality",
            "candidate_action": action,
            "candidate_frequency": frequency,
            "candidate_confidence": "high" if frequency > 0.75 else "medium",
            "is_training_label": False,
            "label_quality": "solver_candidate_untrusted",
            "exclusion_reason": None,
            "warnings": [],
        },
        "quality": {
            "iterations": iterations,
            "exploitability_last": exploitability_last,
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low",
        },
    }
    payload.update(overrides)
    return payload


def assert_not_label(result: dict[str, Any]) -> None:
    assert "training_label" not in result
    assert "gto_label" not in result
    assert result["is_training_label"] is False
    assert result["label_quality"] == "solver_candidate_untrusted"


def test_stable_strong_action_is_quality_ok_but_not_label() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(frequency=0.72),
            run_payload(frequency=0.81),
            run_payload(frequency=0.77),
        ]
    )

    assert result["status"] == "ok"
    assert result["candidate_action"] == "CHECK"
    assert result["stable_action"] is True
    assert result["dominant_frequency_min"] == 0.72
    assert result["dominant_action_consistency"] == 1.0
    assert result["danger_flags"] == []
    assert result["recommendation"] == "usable_for_candidate_analysis"
    assert_not_label(result)


def test_unstable_action_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(action="CHECK", frequency=0.8),
            run_payload(action="BET_33", frequency=0.82),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "dominant_action_unstable"
    assert "dominant_action_unstable" in result["danger_flags"]
    assert_not_label(result)


def test_low_frequency_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(frequency=0.59),
            run_payload(frequency=0.7),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "frequency_too_low"
    assert "frequency_too_low" in result["danger_flags"]
    assert_not_label(result)


def test_all_in_dominant_adds_danger_flag() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(action="ALL_IN", frequency=0.8),
            run_payload(action="ALL_IN", frequency=0.78),
        ]
    )

    assert result["status"] == "ok"
    assert result["candidate_action"] == "ALL_IN"
    assert "extreme_action_all_in" in result["danger_flags"]
    assert_not_label(result)


def test_timeout_run_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(),
            run_payload(solver_status="timeout"),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "timeout"
    assert "timeout" in result["danger_flags"]
    assert_not_label(result)


def test_root_non_hero_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(),
            run_payload(root_role="villain"),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "root_not_hero"
    assert_not_label(result)


def test_single_run_is_refused() -> None:
    result = evaluate_candidate_quality([run_payload()])

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "single_run_only"
    assert "single_run_only" in result["danger_flags"]
    assert_not_label(result)


def test_iterations_too_low_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(iterations=24),
            run_payload(iterations=25),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "iterations_too_low"
    assert "iterations_too_low" in result["danger_flags"]
    assert_not_label(result)


def test_exploitability_missing_is_refused() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(exploitability_last=None),
            run_payload(),
        ]
    )

    assert result["status"] == "failed"
    assert result["exclusion_reason"] == "exploitability_missing"
    assert "exploitability_missing" in result["danger_flags"]
    assert_not_label(result)


def test_no_training_or_gto_label_output() -> None:
    result = evaluate_candidate_quality(
        [
            run_payload(training_label="CHECK", gto_label="CHECK"),
            run_payload(training_label="CHECK", gto_label="CHECK"),
        ]
    )

    assert "training_label" not in result
    assert "gto_label" not in result
    assert result["is_training_label"] is False
