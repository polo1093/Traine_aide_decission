from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from solver_jobs.job_file_runner import run_solver_job_file
from solver_jobs.solver_worker import run_worker_payload
from solver_jobs.subprocess_runner import SUBPROCESS_RESULT_KEYS, run_solver_job_subprocess
from synthetic.spot_generator import generate_solver_jobs


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
            "strategy_entry_count": 12,
        },
        "error": None,
        "duration_ms": 1.0,
        "quality": {
            "iterations": job["iterations"],
            "exploitability_last": 0.5,
            "is_label_candidate": True,
            "exclusion_reason": "iterations_too_low",
        },
    }


def write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True))
            handle.write("\n")


def assert_subprocess_result_stable(result: dict[str, Any]) -> None:
    assert tuple(result) == SUBPROCESS_RESULT_KEYS
    assert result["status"] in {"ok", "failed"}
    assert result["solver_job_id"] is None or isinstance(result["solver_job_id"], str)
    assert result["solver_status"] in {"ok", "failed", "timeout"}
    assert result["solver_result"] is None or isinstance(result["solver_result"], dict)
    assert result["error"] is None or isinstance(result["error"], str)
    assert isinstance(result["duration_ms"], float)
    assert result["quality"]["is_label_candidate"] is False


def make_worker_module(tmp_path: Path, name: str, body: str) -> tuple[str, dict[str, str]]:
    package = tmp_path / "mock_workers"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{name}.py").write_text(body, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    return f"mock_workers.{name}", env


def test_worker_returns_stable_json_with_mock_solver() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_flop_spot", iterations=25, timeout_s=5)[0]

    result = run_worker_payload(job, solver_func=solver_ok)

    assert set(result) == {"status", "solver_job_id", "input", "output", "error", "duration_ms", "quality"}
    assert result["status"] == "ok"
    assert result["solver_job_id"] == job["solver_job_id"]
    assert result["quality"]["is_label_candidate"] is False


def test_subprocess_ok_with_fast_mock_worker(tmp_path: Path) -> None:
    job = generate_solver_jobs(count=1, seed=1, profile="paired_board_spot", iterations=25, timeout_s=5)[0]
    module, env = make_worker_module(
        tmp_path,
        "fast_worker",
        """
import json
import sys

job = json.loads(sys.stdin.read())
print(json.dumps({
    "status": "ok",
    "solver_job_id": job["solver_job_id"],
    "input": job,
    "output": {"backend": "mock", "iterations": job["iterations"], "exploitability_history": [0.5], "strategy_entry_count": 1},
    "error": None,
    "duration_ms": 1.0,
    "quality": {"iterations": job["iterations"], "exploitability_last": 0.5, "is_label_candidate": True, "exclusion_reason": "iterations_too_low"},
}))
""",
    )

    result = run_solver_job_subprocess(job, timeout_s=1, worker_module=module, env=env)

    assert_subprocess_result_stable(result)
    assert result["status"] == "ok"
    assert result["solver_status"] == "ok"
    assert result["solver_result"]["output"]["backend"] == "mock"
    assert result["quality"]["is_label_candidate"] is False


def test_subprocess_timeout_kills_process(tmp_path: Path) -> None:
    job = generate_solver_jobs(count=1, seed=2, profile="drawy_board_spot", iterations=25, timeout_s=5)[0]
    module, env = make_worker_module(
        tmp_path,
        "slow_worker",
        """
import time

time.sleep(5)
""",
    )

    result = run_solver_job_subprocess(job, timeout_s=0.1, worker_module=module, env=env)

    assert_subprocess_result_stable(result)
    assert result["status"] == "failed"
    assert result["solver_status"] == "timeout"
    assert result["solver_result"] is None
    assert "solver_subprocess_timeout:0.1s" in result["error"]
    assert result["quality"]["exclusion_reason"] == "timeout"


def test_invalid_job_fails_before_subprocess() -> None:
    job = generate_solver_jobs(count=1, seed=3, profile="top_pair_spot", iterations=25, timeout_s=5)[0]
    job["board"] = ["2h"]

    result = run_solver_job_subprocess(job, timeout_s=1, worker_module="module_that_should_not_run")

    assert_subprocess_result_stable(result)
    assert result["status"] == "failed"
    assert result["solver_status"] == "failed"
    assert "board_card_count" in result["error"]
    assert result["quality"]["exclusion_reason"] == "job_validation_failed"


def test_worker_error_returns_failed_stable_result(tmp_path: Path) -> None:
    job = generate_solver_jobs(count=1, seed=4, profile="two_pair_plus_spot", iterations=25, timeout_s=5)[0]
    module, env = make_worker_module(
        tmp_path,
        "error_worker",
        """
import sys

sys.stderr.write("boom from worker")
raise SystemExit(2)
""",
    )

    result = run_solver_job_subprocess(job, timeout_s=1, worker_module=module, env=env)

    assert_subprocess_result_stable(result)
    assert result["status"] == "failed"
    assert result["solver_status"] == "failed"
    assert "worker_failed:2" in result["error"]
    assert result["quality"]["exclusion_reason"] == "worker_failed"


def test_job_file_runner_can_use_subprocess_runner(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=1, seed=5, profile="random_turn_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)

    def fake_subprocess(job: dict[str, Any], *, timeout_s: float):
        return {
            "status": "ok",
            "solver_job_id": job["solver_job_id"],
            "solver_status": "ok",
            "solver_result": solver_ok(job),
            "error": None,
            "duration_ms": 1.0,
            "quality": {
                "iterations": job["iterations"],
                "exploitability_last": 0.5,
                "is_label_candidate": False,
                "exclusion_reason": "iterations_too_low",
            },
        }

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job_subprocess", fake_subprocess)

    summary = run_solver_job_file(input_path, output_path, max_jobs=1)

    assert summary["status"] == "ok"
    assert summary["jobs_solved"] == 1
    assert summary["results"][0]["warnings"] == ["line_index:0", "subprocess"]


def test_job_file_runner_jsonl_contains_timeout(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=1, seed=6, profile="random_river_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)

    def fake_timeout(job: dict[str, Any], *, timeout_s: float):
        return {
            "status": "failed",
            "solver_job_id": job["solver_job_id"],
            "solver_status": "timeout",
            "solver_result": None,
            "error": "solver_subprocess_timeout:5s",
            "duration_ms": 5000.0,
            "quality": {
                "iterations": None,
                "exploitability_last": None,
                "is_label_candidate": False,
                "exclusion_reason": "timeout",
            },
        }

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job_subprocess", fake_timeout)

    summary = run_solver_job_file(input_path, output_path, max_jobs=1)
    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert summary["status"] == "failed"
    assert summary["solver_failed"] == 1
    assert record["error"] == "solver_subprocess_timeout:5s"
    assert record["solver_result"] is None
    assert record["quality"]["exclusion_reason"] == "timeout"
    assert "timeout" in record["warnings"]


def test_subprocess_result_uses_python_executable_path_as_arg(tmp_path: Path) -> None:
    job = generate_solver_jobs(count=1, seed=7, profile="random_flop_spot", iterations=25, timeout_s=5)[0]
    module, env = make_worker_module(
        tmp_path,
        "fast_worker_again",
        """
import json
import sys

job = json.loads(sys.stdin.read())
print(json.dumps({"status": "failed", "solver_job_id": job["solver_job_id"], "input": job, "output": None, "error": "mock_error", "duration_ms": 1.0, "quality": {"is_label_candidate": False, "exclusion_reason": "solver_failed"}}))
""",
    )

    result = run_solver_job_subprocess(job, timeout_s=1, python_executable=sys.executable, worker_module=module, env=env)

    assert_subprocess_result_stable(result)
    assert result["status"] == "failed"
    assert result["error"] == "mock_error"
