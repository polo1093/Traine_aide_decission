from __future__ import annotations

from copy import deepcopy
from typing import Any

from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job


EXPECTED_KEYS = {"status", "source_snapshot_id", "solver_job", "error", "warnings"}


def valid_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "ml_dataset_v1",
        "snapshot_id": "snapshot_valid_flop",
        "metadata": {
            "street": "FLOP",
        },
        "features": {
            "hero_cards": ["Ah", "Kh"],
            "villain_hand": ["Qd", "Qc"],
            "board_cards": ["2h", "7h", "9d"],
            "pot": 100.0,
            "to_call": 20.0,
            "to_call_is_estimated": False,
            "decision_context_known": True,
            "stack": 1000.0,
            "active_opponents": 1,
            "hero_position": "BTN",
            "units": "chips",
        },
        "labels": {},
        "confidence": {
            "overall": 0.95,
        },
        "quality_flags": {
            "usable_for_training": True,
        },
    }


def assert_stable_mapping(result: dict[str, Any]) -> None:
    assert set(result) == EXPECTED_KEYS
    assert result["status"] in {"ok", "failed"}
    assert result["source_snapshot_id"] is None or isinstance(result["source_snapshot_id"], str)
    assert result["solver_job"] is None or isinstance(result["solver_job"], dict)
    assert result["error"] is None or isinstance(result["error"], str)
    assert isinstance(result["warnings"], list)


def test_valid_flop_heads_up_snapshot_maps_to_solver_job() -> None:
    result = map_snapshot_to_solver_job(valid_snapshot())

    assert_stable_mapping(result)
    assert result["status"] == "ok", result["error"]
    job = result["solver_job"]
    assert job["schema_version"] == "solver_job_v1"
    assert job["source_snapshot_id"] == "snapshot_valid_flop"
    assert job["source_type"] == "ml_snapshot"
    assert job["units"] == "chips"
    assert job["street"] == "FLOP"
    assert job["hero_hand"] == ["Ah", "Kh"]
    assert job["villain_hand"] == ["Qd", "Qc"]
    assert job["villain_range"] is None
    assert job["board"] == ["2h", "7h", "9d"]
    assert job["pot"] == 100.0
    assert job["to_call"] == 20.0
    assert job["stack"] == 1000.0


def test_snapshot_without_villain_hand_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    del snapshot["features"]["villain_hand"]

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "villain_hand_missing"


def test_multiway_snapshot_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["features"]["active_opponents"] = 2

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "non_heads_up_snapshot"


def test_board_street_incoherent_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["metadata"]["street"] = "TURN"

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert "board_card_count" in result["error"]


def test_unusable_snapshot_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["quality_flags"]["usable_for_training"] = False

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "snapshot_not_usable_for_training"


def test_invalid_cards_fail_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["features"]["hero_cards"] = ["Ah", "Kx"]

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert "invalid_card" in result["error"]


def test_invalid_pot_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["features"]["pot"] = 0

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert "pot_must_be_positive" in result["error"]


def test_low_confidence_fails_cleanly() -> None:
    snapshot = valid_snapshot()
    snapshot["confidence"]["overall"] = 0.2

    result = map_snapshot_to_solver_job(snapshot, min_confidence=0.5)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "confidence_too_low:overall"


def test_villain_range_is_rejected() -> None:
    snapshot = valid_snapshot()
    snapshot["features"]["villain_range"] = "QQ+,AKs"

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "villain_range_not_supported"


def test_output_is_stable_with_warnings() -> None:
    snapshot = valid_snapshot()
    del snapshot["features"]["stack"]
    del snapshot["features"]["units"]

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "ok", result["error"]
    assert "stack_missing_defaulted_to_1000" in result["warnings"]
    assert "units_missing_defaulted_to_chips" in result["warnings"]
    assert result["solver_job"]["stack"] == 1000.0
    assert result["solver_job"]["units"] == "chips"


def test_missing_to_call_fails_without_defaulting_to_zero() -> None:
    snapshot = valid_snapshot()
    del snapshot["features"]["to_call"]

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "to_call_unknown"


def test_unknown_decision_context_fails_even_with_numeric_to_call() -> None:
    snapshot = valid_snapshot()
    snapshot["features"]["decision_context_known"] = False

    result = map_snapshot_to_solver_job(snapshot)

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "decision_context_unknown"


def test_no_raw_exception_for_non_mapping_snapshot() -> None:
    result = map_snapshot_to_solver_job(["not", "a", "mapping"])  # type: ignore[arg-type]

    assert_stable_mapping(result)
    assert result["status"] == "failed"
    assert result["error"] == "snapshot_must_be_mapping"


def test_mapper_does_not_mutate_input_snapshot() -> None:
    snapshot = valid_snapshot()
    before = deepcopy(snapshot)

    result = map_snapshot_to_solver_job(snapshot)

    assert result["status"] == "ok"
    assert snapshot == before
