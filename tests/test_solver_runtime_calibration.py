from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.calibrate_solver_runtime import run_calibration, write_calibration_jsonl


def fake_solver(job: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    profile = job["generation_profile"]
    if profile == "drawy_board_spot":
        return {
            "status": "failed",
            "solver_job_id": job["solver_job_id"],
            "solver_status": "timeout",
            "solver_result": None,
            "error": f"solver_subprocess_timeout:{timeout_s:g}s",
            "duration_ms": timeout_s * 1000,
            "quality": {
                "iterations": None,
                "exploitability_last": None,
                "is_label_candidate": False,
                "exclusion_reason": "timeout",
            },
        }
    return {
        "status": "ok",
        "solver_job_id": job["solver_job_id"],
        "solver_status": "ok",
        "solver_result": {
            "status": "ok",
            "solver_job_id": job["solver_job_id"],
            "quality": {
                "iterations": job["iterations"],
                "exploitability_last": 0.5,
                "is_label_candidate": True,
                "exclusion_reason": "iterations_too_low",
            },
        },
        "error": None,
        "duration_ms": 12.5,
        "quality": {
            "iterations": job["iterations"],
            "exploitability_last": 0.5,
            "is_label_candidate": True,
            "exclusion_reason": "iterations_too_low",
        },
    }


def test_calibration_summary_counts_successes_and_timeouts() -> None:
    summary = run_calibration(
        profiles=["random_flop_spot", "drawy_board_spot"],
        jobs_per_profile=1,
        iterations=[1, 5],
        timeout_s=5,
        solver_func=fake_solver,
    )

    assert summary["status"] == "ok"
    assert summary["total_planned"] == 4
    assert summary["total_run"] == 4
    assert summary["successes"] == 2
    assert summary["timeouts"] == 2
    assert summary["errors"] == 0
    assert summary["profiles"]["random_flop_spot"]["successes"] == 2
    assert summary["profiles"]["drawy_board_spot"]["timeouts"] == 2
    assert summary["profiles_too_heavy"] == ["drawy_board_spot"]
    assert summary["profiles_exploitable_for_smoke"] == ["random_flop_spot"]
    assert summary["recommended_parameters"]["iterations"] == 1
    assert all(row["is_label_candidate"] is False for row in summary["results"])
    assert all(row["quality"]["is_label_candidate"] is False for row in summary["results"])


def test_calibration_refuses_plan_above_max_total_jobs() -> None:
    summary = run_calibration(
        profiles=["random_flop_spot", "random_turn_spot"],
        jobs_per_profile=2,
        iterations=[1, 5],
        max_total_jobs=3,
        solver_func=fake_solver,
    )

    assert summary["status"] == "failed"
    assert "calibration_plan_exceeds_max_total_jobs:3" in summary["error"]
    assert summary["total_run"] == 0


def test_calibration_refuses_iterations_above_step_limit() -> None:
    summary = run_calibration(
        profiles=["random_flop_spot"],
        jobs_per_profile=1,
        iterations=[26],
        solver_func=fake_solver,
    )

    assert summary["status"] == "failed"
    assert "iterations_exceeds_calibration_limit:25" in summary["error"]


def test_write_calibration_jsonl_roundtrip(tmp_path: Path) -> None:
    summary = run_calibration(
        profiles=["random_flop_spot"],
        jobs_per_profile=1,
        iterations=[1],
        timeout_s=5,
        solver_func=fake_solver,
    )
    output_path = tmp_path / "calibration.jsonl"

    write_result = write_calibration_jsonl(summary, output_path)

    assert write_result["status"] == "ok", write_result["error"]
    assert write_result["records_written"] == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["record_type"] == "solver_runtime_calibration"
    assert rows[0]["profile"] == "random_flop_spot"
    assert rows[0]["status"] == "ok"
    assert rows[0]["is_label_candidate"] is False
    assert "training_label" not in rows[0]
    assert "gto_label" not in rows[0]
    assert "label_action" not in rows[0]
