from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pokerth.pipeline import run_pokerth_solver_pipeline


CLEAN_HAND = """
## Game: 4 | Hand: 38 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
polo bets $480.
Player 3 calls $480.
*** TURN *** [Kd]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
polo wins $1920.
"""


NO_SHOWDOWN_HAND = """
## Game: 4 | Hand: 39 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo folds.
"""


MULTIWAY_HAND = """
## Game: 4 | Hand: 40 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
Player 4 calls $320.
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 4 folds.
polo checks.
Player 3 checks.
*** TURN *** [Kd]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
polo wins $1440.
"""


def fake_batch(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for snapshot in snapshots:
        job = {
            "source_snapshot_id": snapshot["snapshot_id"],
            "schema_version": "solver_job_v1",
            "iterations": snapshot["features"].get("solver_iterations", 25),
        }
        rows.append(
            {
                "source_snapshot_id": snapshot["snapshot_id"],
                "mapping_status": "ok",
                "solver_status": "ok",
                "solver_job": job,
                "solver_result": {
                    "status": "ok",
                    "solver_job_id": f"solver_job_from_{snapshot['snapshot_id']}",
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
                },
                "error": None,
                "warnings": [],
                "quality": {
                    "iterations": job["iterations"],
                    "exploitability_last": 0.5,
                    "is_label_candidate": False,
                    "exclusion_reason": "iterations_too_low",
                },
            }
        )
    return {
        "status": "ok" if rows else "failed",
        "total": len(snapshots),
        "mapped": len(snapshots),
        "solved": len(rows),
        "mapping_failed": 0,
        "solver_failed": 0,
        "failed_total": 0,
        "results": rows,
    }


def assert_pipeline_stable(result: dict[str, Any]) -> None:
    assert set(result) == {
        "status",
        "hands_total",
        "hands_parsed",
        "hands_rejected",
        "snapshots_built",
        "snapshots_rejected",
        "jobs_mapped",
        "jobs_solved",
        "solver_failed",
        "output_path",
        "results",
    }
    assert result["status"] in {"ok", "partial", "failed"}
    assert isinstance(result["results"], list)


def test_clean_heads_up_hand_runs_with_mocked_solver(monkeypatch) -> None:
    calls: list[int] = []

    def tracking_batch(snapshots):
        calls.append(len(snapshots))
        return fake_batch(snapshots)

    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", tracking_batch)

    result = run_pokerth_solver_pipeline(text=CLEAN_HAND, street="FLOP", to_call_by_street={"FLOP": 0.0})

    assert_pipeline_stable(result)
    assert result["status"] == "ok"
    assert result["hands_total"] == 1
    assert result["hands_parsed"] == 1
    assert result["hands_rejected"] == 0
    assert result["snapshots_built"] == 1
    assert result["snapshots_rejected"] == 0
    assert result["jobs_mapped"] == 1
    assert result["jobs_solved"] == 1
    assert calls == [1]


def test_hand_without_showdown_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", fake_batch)

    result = run_pokerth_solver_pipeline(text=NO_SHOWDOWN_HAND, street="FLOP", to_call_by_street={"FLOP": 0.0})

    assert result["status"] == "failed"
    assert result["hands_rejected"] == 1
    assert result["snapshots_built"] == 0
    assert result["jobs_solved"] == 0
    assert result["results"][0]["rejection_reason"] == "showdown_missing"


def test_multiway_flop_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", fake_batch)

    result = run_pokerth_solver_pipeline(text=MULTIWAY_HAND, street="FLOP", to_call_by_street={"FLOP": 0.0})

    assert result["status"] == "failed"
    assert result["hands_parsed"] == 1
    assert result["snapshots_rejected"] == 1
    assert result["results"][-1]["rejection_reason"] == "multiway_context_not_supported"


def test_to_call_unknown_is_rejected_without_solver_call(monkeypatch) -> None:
    def should_not_call_batch(snapshots):
        raise AssertionError("solver batch should not run")

    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", should_not_call_batch)

    result = run_pokerth_solver_pipeline(text=CLEAN_HAND, street="FLOP")

    assert result["status"] == "failed"
    assert result["snapshots_rejected"] == 1
    assert result["jobs_solved"] == 0
    assert result["results"][-1]["rejection_reason"] == "to_call_unknown"


def test_mixed_batch_is_partial(monkeypatch) -> None:
    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", fake_batch)

    result = run_pokerth_solver_pipeline(
        text=CLEAN_HAND + "\n" + NO_SHOWDOWN_HAND,
        street="FLOP",
        to_call_by_street={"FLOP": 0.0},
    )

    assert result["status"] == "partial"
    assert result["hands_total"] == 2
    assert result["hands_parsed"] == 1
    assert result["hands_rejected"] == 1
    assert result["snapshots_built"] == 1
    assert result["jobs_solved"] == 1


def test_jsonl_output_is_readable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pokerth.pipeline.run_solver_batch", fake_batch)
    output_path = tmp_path / "solver_run.jsonl"

    result = run_pokerth_solver_pipeline(
        text=CLEAN_HAND,
        street="FLOP",
        to_call_by_street={"FLOP": 0.0},
        output_path=output_path,
    )

    assert result["status"] == "ok"
    assert result["output_path"] == str(output_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["record_type"] == "solver_run_result"
    assert records[0]["quality"]["is_label_candidate"] is False
    assert "training_label" not in records[0]
    assert "gto_label" not in records[0]
    assert "label_action" not in records[0]


def test_no_raw_exception_for_missing_input() -> None:
    result = run_pokerth_solver_pipeline()

    assert_pipeline_stable(result)
    assert result["status"] == "failed"
    assert result["results"][0]["rejection_reason"] == "input_missing"
