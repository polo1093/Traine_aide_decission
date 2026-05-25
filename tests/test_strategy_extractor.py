from __future__ import annotations

from typing import Any

from solver_jobs.strategy_extractor import extract_root_strategy


def run_result(root_strategy: Any = None, *, output: dict[str, Any] | None = None) -> dict[str, Any]:
    solver_output = {"game_value": 1.0}
    if root_strategy is not None:
        solver_output["root_strategy"] = root_strategy
    if output:
        solver_output.update(output)
    return {
        "record_type": "solver_run_result",
        "solver_job_id": "job-1",
        "solver_status": "ok",
        "solver_result": {
            "status": "ok",
            "solver_job_id": "job-1",
            "output": solver_output,
        },
        "quality": {
            "iterations": 25,
            "exploitability_last": 0.1,
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low",
        },
    }


def test_extracts_mocked_root_strategy() -> None:
    result = extract_root_strategy(
        run_result(
            {
                "action_frequencies": {"CHECK": 0.4, "BET_33": 0.6},
                "action_evs": {"CHECK": 1.0, "BET_33": 1.2},
            }
        )
    )

    assert result["status"] == "ok"
    assert result["available"] is True
    assert result["action_frequencies"] == {"BET_33": 0.6, "CHECK": 0.4}
    assert result["action_evs"] == {"CHECK": 1.0, "BET_33": 1.2}
    assert result["dominant_action"] == "BET_33"
    assert result["dominant_action_frequency"] == 0.6
    assert result["confidence"] == "medium"
    assert result["error"] is None


def test_extracts_root_strategy_raw_from_adapter_shape() -> None:
    result = extract_root_strategy(
        run_result(
            output={
                "root_strategy_raw": {
                    "infoset_key": "root",
                    "player": 1,
                    "root_player": 1,
                    "root_player_role": "unknown",
                    "action_ids": [1, 3, 13],
                    "action_labels": ["CHECK", "BET_66", "ALL_IN"],
                    "frequencies": [0.2, 0.7, 0.1],
                    "source": "average_strategy",
                    "bet_size_fractions": [0.66],
                }
            }
        )
    )

    assert result["status"] == "ok"
    assert result["available"] is True
    assert result["root_player"] == 1
    assert result["root_player_role"] == "unknown"
    assert result["action_frequencies"] == {"ALL_IN": 0.1, "BET_66": 0.7, "CHECK": 0.2}
    assert result["dominant_action"] == "BET_66"
    assert result["dominant_action_frequency"] == 0.7
    assert result["confidence"] == "medium"


def test_solver_result_without_strategy_fails_cleanly() -> None:
    result = extract_root_strategy(run_result(root_strategy=None))

    assert result["status"] == "failed"
    assert result["available"] is False
    assert result["error"] == "strategy_not_available"


def test_game_value_only_does_not_create_action() -> None:
    result = extract_root_strategy(run_result(root_strategy=None, output={"strategy_entry_count": 12}))

    assert result["status"] == "failed"
    assert result["action_frequencies"] == {}
    assert result["dominant_action"] is None
    assert result["error"] == "strategy_not_available"


def test_invalid_frequencies_fail_cleanly() -> None:
    result = extract_root_strategy(run_result({"CHECK": 0.2, "BET_33": 0.2}))

    assert result["status"] == "failed"
    assert result["error"] == "invalid_frequency_sum"


def test_dominant_action_is_calculated() -> None:
    result = extract_root_strategy(run_result({"CHECK": 0.1, "BET_33": 0.2, "BET_75": 0.7}))

    assert result["status"] == "ok"
    assert result["dominant_action"] == "BET_75"
    assert result["dominant_action_frequency"] == 0.7


def test_confidence_low_medium_high() -> None:
    low = extract_root_strategy(run_result({"CHECK": 0.46, "BET_33": 0.54}))
    medium = extract_root_strategy(run_result({"CHECK": 0.4, "BET_33": 0.6}))
    medium_boundary = extract_root_strategy(run_result({"CHECK": 0.25, "BET_33": 0.75}))
    high = extract_root_strategy(run_result({"CHECK": 0.2, "BET_33": 0.8}))

    assert low["confidence"] == "low"
    assert medium["confidence"] == "medium"
    assert medium_boundary["confidence"] == "medium"
    assert high["confidence"] == "high"


def test_action_rows_are_supported() -> None:
    result = extract_root_strategy(
        run_result(
            [
                {"action": "check", "frequency": 0.25, "ev": 0.1},
                {"action": "bet-33", "frequency": 0.75, "ev": 0.3},
            ]
        )
    )

    assert result["status"] == "ok"
    assert result["action_frequencies"] == {"BET_33": 0.75, "CHECK": 0.25}
    assert result["action_evs"] == {"CHECK": 0.1, "BET_33": 0.3}
    assert result["confidence"] == "high"


def test_no_raw_exception_for_bad_payload() -> None:
    result = extract_root_strategy({"solver_result": {"output": {"root_strategy": [{"action": "CHECK"}]}}})

    assert result["status"] == "failed"
    assert result["error"] == "invalid_frequency_value"


def test_root_strategy_raw_rejects_invalid_frequencies() -> None:
    result = extract_root_strategy(
        run_result(
            output={
                "root_strategy_raw": {
                    "action_labels": ["CHECK", "BET_66"],
                    "frequencies": [0.7, 0.7],
                }
            }
        )
    )

    assert result["status"] == "failed"
    assert result["error"] == "invalid_frequency_sum"
