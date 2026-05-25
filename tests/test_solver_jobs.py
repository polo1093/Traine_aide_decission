from __future__ import annotations

from typing import Any

from solver_jobs.job_builder import build_solver_job, manual_fixture_spots
from solver_jobs.job_runner import RUNNER_KEYS, run_solver_job


def valid_job() -> dict[str, Any]:
    result = build_solver_job(
        solver_job_id="solver_job_test_valid",
        source_snapshot_id="snapshot_test_valid",
        created_at="2026-05-25T00:00:00+00:00",
        source_type="manual_fixture",
        units="chips",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=20,
        stack=1000,
        bet_sizes=[0.33],
        iterations=25,
        timeout_s=5,
        backend="rust",
        label_intent="solver_smoke",
    )
    assert result["status"] == "ok", result["error"]
    return result["job"]


def assert_builder_failed(result: dict[str, Any], expected: str) -> None:
    assert result["status"] == "failed"
    assert result["job"] is None
    assert expected in result["error"]


def assert_runner_stable(result: dict[str, Any]) -> None:
    assert tuple(result) == RUNNER_KEYS
    assert result["status"] in {"ok", "failed"}
    assert result["solver_job_id"] is None or isinstance(result["solver_job_id"], str)
    assert isinstance(result["input"], dict)
    assert result["output"] is None or isinstance(result["output"], dict)
    assert result["error"] is None or isinstance(result["error"], str)
    assert isinstance(result["duration_ms"], float)
    assert isinstance(result["quality"], dict)
    assert result["quality"]["is_label_candidate"] is False


def test_valid_job_created_cleanly() -> None:
    job = valid_job()

    assert job["schema_version"] == "solver_job_v1"
    assert job["solver_job_id"] == "solver_job_test_valid"
    assert job["source_snapshot_id"] == "snapshot_test_valid"
    assert job["source_type"] == "manual_fixture"
    assert job["units"] == "chips"
    assert job["street"] == "FLOP"
    assert job["hero_hand"] == ["Ah", "Kh"]
    assert job["villain_hand"] == ["Qd", "Qc"]
    assert job["villain_range"] is None
    assert job["board"] == ["2h", "7h", "9d"]


def test_manual_fixtures_include_valid_and_rejected_examples() -> None:
    fixtures = manual_fixture_spots()

    assert len(fixtures) == 5
    assert fixtures[0]["status"] == "ok"
    assert fixtures[1]["status"] == "ok"
    assert fixtures[2]["status"] == "ok"
    assert fixtures[3]["status"] == "failed"
    assert fixtures[4]["status"] == "failed"


def test_invalid_card_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_bad_card",
        street="FLOP",
        hero_hand=["Ah", "Kx"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
    )

    assert_builder_failed(result, "invalid_card")


def test_duplicate_cards_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_duplicate",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["Ah", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
    )

    assert_builder_failed(result, "duplicate_cards")


def test_street_board_incoherent_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_bad_street",
        street="TURN",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
    )

    assert_builder_failed(result, "board_card_count")


def test_invalid_pot_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_bad_pot",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=0,
        to_call=0,
        stack=1000,
    )

    assert_builder_failed(result, "pot_must_be_positive")


def test_invalid_stack_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_bad_stack",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=0,
    )

    assert_builder_failed(result, "stack_must_be_positive")


def test_iterations_too_high_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_high_iterations",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
        iterations=101,
    )

    assert_builder_failed(result, "iterations_exceeds_limit:100")


def test_timeout_too_high_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_high_timeout",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
        timeout_s=10.1,
    )

    assert_builder_failed(result, "timeout_s_exceeds_limit:10")


def test_timeout_absent_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_missing_timeout",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=["Qd", "Qc"],
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
        timeout_s=None,
    )

    assert_builder_failed(result, "timeout_s_required")


def test_villain_range_rejected() -> None:
    result = build_solver_job(
        source_snapshot_id="snapshot_range",
        street="FLOP",
        hero_hand=["Ah", "Kh"],
        villain_hand=None,
        villain_range="QQ+,AKs",
        board=["2h", "7h", "9d"],
        pot=100,
        to_call=0,
        stack=1000,
    )

    assert_builder_failed(result, "villain_range_not_supported")


def test_runner_returns_stable_structure_for_invalid_job() -> None:
    job = valid_job()
    job["board"] = ["2h"]

    result = run_solver_job(job)

    assert_runner_stable(result)
    assert result["status"] == "failed"
    assert "board_card_count" in result["error"]
    assert result["quality"]["exclusion_reason"] == "job_validation_failed"


def test_runner_ok_with_mock_adapter(monkeypatch) -> None:
    job = valid_job()

    def fake_solve_tiny_postflop_spot(*args, **kwargs):
        return {
            "status": "ok",
            "solver_name": "PokerSolver",
            "input": {"args": args, "kwargs": kwargs},
            "output": {
                "backend": "rust",
                "iterations": 25,
                "game_value": 1.25,
                "exploitability_history": [0.5],
                "strategy_entry_count": 12,
            },
            "error": None,
            "duration_ms": 1.0,
        }

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fake_solve_tiny_postflop_spot)
    result = run_solver_job(job)

    assert_runner_stable(result)
    assert result["status"] == "ok"
    assert result["output"]["backend"] == "rust"
    assert result["quality"]["iterations"] == 25
    assert result["quality"]["exploitability_last"] == 0.5
    assert result["quality"]["is_label_candidate"] is False
    assert result["quality"]["exclusion_reason"] == "iterations_too_low"


def test_runner_failed_with_mock_error(monkeypatch) -> None:
    job = valid_job()

    def fake_solve_tiny_postflop_spot(*args, **kwargs):
        return {
            "status": "failed",
            "solver_name": "PokerSolver",
            "input": {},
            "output": None,
            "error": "solver_timeout:5.0s",
            "duration_ms": 5000.0,
        }

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fake_solve_tiny_postflop_spot)
    result = run_solver_job(job)

    assert_runner_stable(result)
    assert result["status"] == "failed"
    assert result["error"] == "solver_timeout:5.0s"
    assert result["quality"]["is_label_candidate"] is False
    assert result["quality"]["exclusion_reason"] == "solver_failed"


def test_runner_catches_adapter_exception(monkeypatch) -> None:
    job = valid_job()

    def fake_solve_tiny_postflop_spot(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("solver_jobs.job_runner.solve_tiny_postflop_spot", fake_solve_tiny_postflop_spot)
    result = run_solver_job(job)

    assert_runner_stable(result)
    assert result["status"] == "failed"
    assert "RuntimeError:boom" in result["error"]
    assert result["quality"]["exclusion_reason"] == "runner_exception"
