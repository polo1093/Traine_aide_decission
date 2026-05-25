from __future__ import annotations

from typing import Any

from solver_jobs.strategy_extractor import extract_root_strategy


def root_strategy_raw(**overrides: Any) -> dict[str, Any]:
    raw = {
        "infoset_key": "root",
        "player": 0,
        "root_player": 0,
        "hero_solver_player": 0,
        "root_matches_hero": True,
        "root_player_role": "hero",
        "decision_actor": "hero",
        "root_must_be_hero": True,
        "action_ids": [1, 3, 13],
        "action_labels": ["CHECK", "BET_66", "ALL_IN"],
        "frequencies": [0.2, 0.7, 0.1],
        "source": "average_strategy",
        "bet_size_fractions": [0.66],
    }
    raw.update(overrides)
    return raw


def run_result(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {"game_value": 1.0}
    if raw is not None:
        output["root_strategy_raw"] = raw
    return {
        "record_type": "solver_run_result",
        "solver_job_id": "job-1",
        "solver_status": "ok",
        "solver_result": {
            "status": "ok",
            "solver_job_id": "job-1",
            "output": output,
        },
        "quality": {
            "iterations": 25,
            "exploitability_last": 0.1,
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low",
        },
    }


def test_root_player_matching_hero_allows_extraction() -> None:
    result = extract_root_strategy(run_result(root_strategy_raw()))

    assert result["status"] == "ok"
    assert result["available"] is True
    assert result["root_player"] == 0
    assert result["root_player_role"] == "hero"
    assert result["action_frequencies"] == {"ALL_IN": 0.1, "BET_66": 0.7, "CHECK": 0.2}
    assert result["dominant_action"] == "BET_66"
    assert result["dominant_action_frequency"] == 0.7
    assert result["confidence"] == "medium"
    assert result["error"] is None


def test_root_player_not_matching_hero_is_refused() -> None:
    result = extract_root_strategy(
        run_result(
            root_strategy_raw(
                player=1,
                root_player=1,
                hero_solver_player=0,
                root_matches_hero=False,
                root_player_role="villain",
            )
        )
    )

    assert result["status"] == "failed"
    assert result["available"] is False
    assert result["action_frequencies"] == {}
    assert result["dominant_action"] is None
    assert result["error"] == "root_player_not_hero"


def test_missing_hero_solver_player_is_refused() -> None:
    result = extract_root_strategy(
        run_result(root_strategy_raw(hero_solver_player=None, root_matches_hero=None, root_player_role="unknown"))
    )

    assert result["status"] == "failed"
    assert result["error"] == "hero_solver_player_unknown"


def test_unknown_root_player_role_is_refused() -> None:
    result = extract_root_strategy(run_result(root_strategy_raw(root_player_role="unknown")))

    assert result["status"] == "failed"
    assert result["error"] == "hero_solver_player_unknown"


def test_absent_root_strategy_raw_is_refused() -> None:
    result = extract_root_strategy(run_result(None))

    assert result["status"] == "failed"
    assert result["available"] is False
    assert result["error"] == "strategy_not_available"


def test_game_value_only_does_not_create_action() -> None:
    result = extract_root_strategy(
        {
            "solver_job_id": "job-1",
            "solver_result": {"output": {"game_value": 1.0, "strategy_entry_count": 12}},
        }
    )

    assert result["status"] == "failed"
    assert result["action_frequencies"] == {}
    assert result["dominant_action"] is None
    assert result["error"] == "strategy_not_available"


def test_invalid_frequencies_fail_cleanly() -> None:
    result = extract_root_strategy(run_result(root_strategy_raw(frequencies=[0.7, 0.7, 0.1])))

    assert result["status"] == "failed"
    assert result["error"] == "invalid_frequency_sum"


def test_frequency_shape_mismatch_fails_cleanly() -> None:
    result = extract_root_strategy(run_result(root_strategy_raw(action_labels=["CHECK"], frequencies=[1.0, 0.0])))

    assert result["status"] == "failed"
    assert result["error"] == "invalid_frequency_shape"


def test_confidence_low_medium_high() -> None:
    low = extract_root_strategy(run_result(root_strategy_raw(frequencies=[0.46, 0.54, 0.0])))
    medium = extract_root_strategy(run_result(root_strategy_raw(frequencies=[0.4, 0.6, 0.0])))
    medium_boundary = extract_root_strategy(run_result(root_strategy_raw(frequencies=[0.25, 0.75, 0.0])))
    high = extract_root_strategy(run_result(root_strategy_raw(frequencies=[0.2, 0.8, 0.0])))

    assert low["confidence"] == "low"
    assert medium["confidence"] == "medium"
    assert medium_boundary["confidence"] == "medium"
    assert high["confidence"] == "high"


def test_no_raw_exception_for_bad_payload() -> None:
    result = extract_root_strategy({"solver_result": {"output": {"root_strategy_raw": {"frequencies": "bad"}}}})

    assert result["status"] == "failed"
    assert result["error"] in {"hero_solver_player_unknown", "strategy_not_available"}


def test_no_label_fields_are_created() -> None:
    result = extract_root_strategy(run_result(root_strategy_raw()))

    assert "training_label" not in result
    assert result.get("is_label_candidate") is not True
