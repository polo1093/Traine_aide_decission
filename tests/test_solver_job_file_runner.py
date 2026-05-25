from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solver_jobs.job_file_runner import FILE_RUNNER_KEYS, run_solver_job_file
from synthetic.spot_generator import generate_solver_jobs


def write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True))
            handle.write("\n")


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
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low",
        },
    }


def assert_summary_stable(summary: dict[str, Any]) -> None:
    assert tuple(summary) == FILE_RUNNER_KEYS
    assert summary["status"] in {"ok", "partial", "failed"}
    assert isinstance(summary["input_path"], str)
    assert isinstance(summary["output_path"], str)
    assert isinstance(summary["total_read"], int)
    assert isinstance(summary["jobs_valid"], int)
    assert isinstance(summary["jobs_invalid"], int)
    assert isinstance(summary["jobs_solved"], int)
    assert isinstance(summary["solver_failed"], int)
    assert isinstance(summary["dry_run"], bool)
    assert isinstance(summary["results"], list)


def assert_record_stable(record: dict[str, Any]) -> None:
    assert set(record) == {
        "record_type",
        "solver_job_id",
        "source_snapshot_id",
        "source_type",
        "solver_job",
        "solver_result",
        "solver_status",
        "quality",
        "error",
        "warnings",
        "recorded_at",
    }
    assert record["record_type"] == "solver_run_result"
    assert record["source_type"] is None or record["source_type"] == "synthetic"
    assert record["quality"]["is_label_candidate"] is False
    assert isinstance(record["warnings"], list)
    assert "training_label" not in record
    assert "gto_label" not in record
    assert "label_action" not in record


def test_load_valid_jsonl_and_solve_with_mock(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=2, seed=42, profile="random_turn_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)
    calls: list[str] = []

    def fake_run_solver_job(job: dict[str, Any]) -> dict[str, Any]:
        calls.append(job["solver_job_id"])
        return solver_ok(job)

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", fake_run_solver_job)

    summary = run_solver_job_file(input_path, output_path, max_jobs=2, use_subprocess=False)

    assert_summary_stable(summary)
    assert summary["status"] == "ok"
    assert summary["total_read"] == 2
    assert summary["jobs_valid"] == 2
    assert summary["jobs_invalid"] == 0
    assert summary["jobs_solved"] == 2
    assert summary["solver_failed"] == 0
    assert calls == [job["solver_job_id"] for job in jobs]
    assert all(record["solver_result"]["status"] == "ok" for record in summary["results"])


def test_invalid_job_is_rejected_without_solver(tmp_path: Path, monkeypatch) -> None:
    job = generate_solver_jobs(count=1, seed=1, profile="random_flop_spot", iterations=25, timeout_s=5)[0]
    job["board"] = ["2h"]
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, [job])

    def fail_if_called(job: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - only runs on regression
        raise AssertionError("solver should not be called for invalid jobs")

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", fail_if_called)

    summary = run_solver_job_file(input_path, output_path, max_jobs=1, use_subprocess=False)

    assert summary["status"] == "failed"
    assert summary["jobs_valid"] == 0
    assert summary["jobs_invalid"] == 1
    assert summary["jobs_solved"] == 0
    assert "board_card_count" in summary["results"][0]["error"]


def test_dry_run_validates_without_solver(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=3, seed=2, profile="paired_board_spot", iterations=25, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)

    def fail_if_called(job: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - only runs on regression
        raise AssertionError("dry_run must not call solver")

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", fail_if_called)

    summary = run_solver_job_file(input_path, output_path, max_jobs=3, dry_run=True, use_subprocess=False)

    assert summary["status"] == "ok"
    assert summary["dry_run"] is True
    assert summary["jobs_valid"] == 3
    assert summary["jobs_solved"] == 0
    assert all(record["solver_result"] is None for record in summary["results"])
    assert all(record["quality"]["exclusion_reason"] == "dry_run" for record in summary["results"])


def test_max_jobs_limits_processed_jobs(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=6, seed=3, profile="random_river_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)
    calls: list[str] = []

    def fake_run_solver_job(job: dict[str, Any]) -> dict[str, Any]:
        calls.append(job["solver_job_id"])
        return solver_ok(job)

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", fake_run_solver_job)

    summary = run_solver_job_file(input_path, output_path, max_jobs=2, start_index=1, use_subprocess=False)

    assert summary["status"] == "ok"
    assert summary["total_read"] == 2
    assert calls == [jobs[1]["solver_job_id"], jobs[2]["solver_job_id"]]


def test_output_jsonl_is_readable_and_not_a_training_dataset(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=1, seed=4, profile="random_turn_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)
    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", solver_ok)

    summary = run_solver_job_file(input_path, output_path, max_jobs=1, use_subprocess=False)

    assert summary["status"] == "ok"
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert_record_stable(record)
    assert record["solver_job_id"] == jobs[0]["solver_job_id"]
    assert record["source_snapshot_id"] == jobs[0]["source_snapshot_id"]
    assert record["solver_job"]["source_type"] == "synthetic"
    assert record["solver_result"]["status"] == "ok"


def test_solver_failed_writes_stable_result_line(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=1, seed=5, profile="random_river_spot", iterations=5, timeout_s=5)
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)

    def solver_failed(job: dict[str, Any]) -> dict[str, Any]:
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

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", solver_failed)

    summary = run_solver_job_file(input_path, output_path, max_jobs=1, use_subprocess=False)

    assert summary["status"] == "failed"
    assert summary["jobs_valid"] == 1
    assert summary["jobs_solved"] == 0
    assert summary["solver_failed"] == 1
    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert_record_stable(record)
    assert record["error"] == "solver_timeout:5.0s"
    assert record["quality"]["is_label_candidate"] is False


def test_summary_counts_for_mixed_file(tmp_path: Path, monkeypatch) -> None:
    jobs = generate_solver_jobs(count=3, seed=6, profile="random_turn_spot", iterations=5, timeout_s=5)
    jobs[1]["board"] = ["2h"]
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)
    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job", solver_ok)

    summary = run_solver_job_file(input_path, output_path, max_jobs=3, use_subprocess=False)

    assert summary["status"] == "partial"
    assert summary["total_read"] == 3
    assert summary["jobs_valid"] == 2
    assert summary["jobs_invalid"] == 1
    assert summary["jobs_solved"] == 2
    assert summary["solver_failed"] == 0


def test_no_raw_exception_for_bad_json_and_invalid_parameters(tmp_path: Path) -> None:
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("{not json}\n", encoding="utf-8")

    bad_json = run_solver_job_file(input_path, output_path, max_jobs=1)
    too_large = run_solver_job_file(input_path, output_path, max_jobs=51)

    assert_summary_stable(bad_json)
    assert bad_json["status"] == "failed"
    assert bad_json["jobs_invalid"] == 1
    assert "JSONDecodeError" in bad_json["results"][0]["error"]
    assert_summary_stable(too_large)
    assert too_large["status"] == "failed"
    assert "max_jobs_exceeds_limit:50" in too_large["results"][0]["error"]
