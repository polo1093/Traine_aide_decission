from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solver_jobs.batch_runner import BATCH_RESULT_KEYS, run_solver_batch, write_solver_batch_jsonl


def snapshot(
    snapshot_id: str,
    *,
    street: str = "FLOP",
    board_cards: list[str] | None = None,
    villain_hand: list[str] | None = None,
    active_opponents: int = 1,
    pot: float = 100.0,
) -> dict[str, Any]:
    if board_cards is None:
        board_cards = ["2h", "7h", "9d"]
    features: dict[str, Any] = {
        "hero_cards": ["Ah", "Kh"],
        "board_cards": board_cards,
        "pot": pot,
        "to_call": 20.0,
        "to_call_is_estimated": False,
        "decision_context_known": True,
        "stack": 1000.0,
        "active_opponents": active_opponents,
        "hero_position": "BTN",
        "units": "chips",
    }
    if villain_hand is not None:
        features["villain_hand"] = villain_hand
    return {
        "schema_version": "ml_dataset_v1",
        "snapshot_id": snapshot_id,
        "metadata": {"street": street},
        "features": features,
        "labels": {},
        "confidence": {"overall": 0.95},
        "quality_flags": {"usable_for_training": True},
    }


def solver_ok(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "solver_job_id": job["solver_job_id"],
        "input": job,
        "output": {
            "backend": "rust",
            "iterations": job["iterations"],
            "game_value": 1.0,
            "exploitability_history": [0.5],
            "strategy_entry_count": 16,
        },
        "error": None,
        "duration_ms": 1.0,
        "quality": {
            "iterations": job["iterations"],
            "exploitability_last": 0.5,
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low",
        },
    }


def assert_batch_stable(result: dict[str, Any]) -> None:
    assert tuple(result) == BATCH_RESULT_KEYS
    assert result["status"] in {"ok", "partial", "failed"}
    assert isinstance(result["total"], int)
    assert isinstance(result["mapped"], int)
    assert isinstance(result["solved"], int)
    assert isinstance(result["mapping_failed"], int)
    assert isinstance(result["solver_failed"], int)
    assert isinstance(result["failed_total"], int)
    assert isinstance(result["results"], list)


def assert_row_stable(row: dict[str, Any]) -> None:
    assert set(row) == {
        "source_snapshot_id",
        "mapping_status",
        "solver_status",
        "solver_job",
        "solver_result",
        "error",
        "warnings",
        "quality",
    }
    assert row["mapping_status"] in {"ok", "failed"}
    assert row["solver_status"] in {"ok", "failed", "skipped"}
    assert row["solver_job"] is None or isinstance(row["solver_job"], dict)
    assert row["solver_result"] is None or isinstance(row["solver_result"], dict)
    assert row["error"] is None or isinstance(row["error"], str)
    assert isinstance(row["warnings"], list)
    assert row["quality"]["is_label_candidate"] is False


def test_batch_all_valid_snapshots(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_solver_job(job):
        calls.append(job["source_snapshot_id"])
        return solver_ok(job)

    monkeypatch.setattr("solver_jobs.batch_runner.run_solver_job", fake_run_solver_job)
    snapshots = [
        snapshot("snapshot_flop", villain_hand=["Qd", "Qc"]),
        snapshot("snapshot_turn", street="TURN", board_cards=["2h", "7h", "9d", "4c"], villain_hand=["Qd", "Qc"]),
    ]

    result = run_solver_batch(snapshots)

    assert_batch_stable(result)
    assert result["status"] == "ok"
    assert result["total"] == 2
    assert result["mapped"] == 2
    assert result["solved"] == 2
    assert result["mapping_failed"] == 0
    assert result["solver_failed"] == 0
    assert result["failed_total"] == 0
    assert calls == ["snapshot_flop", "snapshot_turn"]
    for row in result["results"]:
        assert_row_stable(row)
        assert row["quality"]["is_label_candidate"] is False


def test_batch_mixed_valid_invalid(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_solver_job(job):
        calls.append(job["source_snapshot_id"])
        return solver_ok(job)

    monkeypatch.setattr("solver_jobs.batch_runner.run_solver_job", fake_run_solver_job)
    snapshots = [
        snapshot("snapshot_flop", villain_hand=["Qd", "Qc"]),
        snapshot("snapshot_missing_villain"),
        snapshot("snapshot_multiway", villain_hand=["Qd", "Qc"], active_opponents=2),
        snapshot("snapshot_bad_board", street="TURN", board_cards=["2h", "7h", "9d"], villain_hand=["Qd", "Qc"]),
    ]

    result = run_solver_batch(snapshots)

    assert result["status"] == "partial"
    assert result["total"] == 4
    assert result["mapped"] == 1
    assert result["solved"] == 1
    assert result["mapping_failed"] == 3
    assert result["solver_failed"] == 0
    assert result["failed_total"] == 3
    assert calls == ["snapshot_flop"]
    assert [row["solver_status"] for row in result["results"]] == ["ok", "skipped", "skipped", "skipped"]


def test_mapping_failed_does_not_call_solver(monkeypatch) -> None:
    def fake_run_solver_job(job):
        raise AssertionError("solver should not be called")

    monkeypatch.setattr("solver_jobs.batch_runner.run_solver_job", fake_run_solver_job)

    result = run_solver_batch([snapshot("snapshot_missing_villain")])

    assert result["status"] == "failed"
    assert result["total"] == 1
    assert result["mapped"] == 0
    assert result["solved"] == 0
    assert result["mapping_failed"] == 1
    assert result["solver_failed"] == 0
    assert result["failed_total"] == 1
    assert result["results"][0]["error"] == "villain_hand_missing"


def test_solver_failed_result_is_stable(monkeypatch) -> None:
    def fake_run_solver_job(job):
        return {
            "status": "failed",
            "solver_job_id": job["solver_job_id"],
            "input": job,
            "output": None,
            "error": "solver_timeout:5.0s",
            "duration_ms": 5000.0,
            "quality": {
                "iterations": job["iterations"],
                "exploitability_last": None,
                "is_label_candidate": True,
                "exclusion_reason": "solver_failed",
            },
        }

    monkeypatch.setattr("solver_jobs.batch_runner.run_solver_job", fake_run_solver_job)

    result = run_solver_batch([snapshot("snapshot_flop", villain_hand=["Qd", "Qc"])])

    assert result["status"] == "failed"
    assert result["mapped"] == 1
    assert result["solved"] == 0
    assert result["solver_failed"] == 1
    row = result["results"][0]
    assert_row_stable(row)
    assert row["solver_status"] == "failed"
    assert row["error"] == "solver_timeout:5.0s"
    assert row["quality"]["is_label_candidate"] is False


def test_jsonl_write_and_readback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("solver_jobs.batch_runner.run_solver_job", solver_ok)
    batch = run_solver_batch([snapshot("snapshot_flop", villain_hand=["Qd", "Qc"])])
    output_path = tmp_path / "solver_run.jsonl"

    write_result = write_solver_batch_jsonl(
        batch,
        output_path,
        solver_info={
            "solver_name": "PokerSolver",
            "version": "1.7.0",
            "rust_backend_available": True,
            "status": "ok",
            "error": None,
        },
    )

    assert write_result["status"] == "ok", write_result["error"]
    assert write_result["records_written"] == 1
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["record_type"] == "solver_run_result"
    assert record["source_snapshot_id"] == "snapshot_flop"
    assert record["mapping_status"] == "ok"
    assert record["solver_status"] == "ok"
    assert record["solver_job"]["source_snapshot_id"] == "snapshot_flop"
    assert record["solver_result"]["status"] == "ok"
    assert record["quality"]["is_label_candidate"] is False
    assert record["solver"]["version"] == "1.7.0"
    assert "recorded_at" in record
    assert "label_action" not in record
    assert "gto_label" not in record
    assert "training_label" not in record


def test_no_raw_exception_for_invalid_batch_input() -> None:
    result = run_solver_batch({"not": "a list"})  # type: ignore[arg-type]

    assert_batch_stable(result)
    assert result["status"] == "failed"
    assert result["failed_total"] == 1
    assert result["results"][0]["error"] == "snapshots_must_be_list"
