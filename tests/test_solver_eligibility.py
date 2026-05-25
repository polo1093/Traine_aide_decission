from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solver_jobs.eligibility import evaluate_solver_eligibility
from solver_jobs.job_file_runner import run_solver_job_file
from synthetic.spot_generator import generate_solver_jobs


def write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True))
            handle.write("\n")


def test_random_turn_iterations_1_is_eligible() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_turn_spot", iterations=1, timeout_s=5)[0]

    result = evaluate_solver_eligibility(job)

    assert result == {"eligible": True, "reason": None, "warnings": []}


def test_random_river_iterations_5_is_eligible() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_river_spot", iterations=5, timeout_s=5)[0]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is True
    assert result["reason"] is None


def test_random_flop_is_refused_as_timeout_risk() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_flop_spot", iterations=1, timeout_s=5)[0]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "flop_solver_timeout_risk"


def test_drawy_profile_is_refused() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="drawy_board_spot", iterations=1, timeout_s=5)[0]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "flop_solver_timeout_risk"


def test_iterations_25_is_refused() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_turn_spot", iterations=25, timeout_s=5)[0]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "iterations_too_high_for_calibration"


def test_timeout_above_5_is_refused() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_river_spot", iterations=5, timeout_s=6)[0]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "timeout_too_high_for_calibration"


def test_invalid_job_is_refused() -> None:
    job = generate_solver_jobs(count=1, seed=42, profile="random_turn_spot", iterations=1, timeout_s=5)[0]
    job["board"] = ["2h"]

    result = evaluate_solver_eligibility(job)

    assert result["eligible"] is False
    assert result["reason"] == "invalid_solver_job"
    assert "board_card_count" in result["warnings"][0]


def test_job_file_runner_skips_non_eligible_without_solver(tmp_path: Path, monkeypatch) -> None:
    jobs = [
        generate_solver_jobs(count=1, seed=1, profile="random_flop_spot", iterations=1, timeout_s=5)[0],
        generate_solver_jobs(count=1, seed=2, profile="random_turn_spot", iterations=25, timeout_s=5)[0],
    ]
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, jobs)

    def fail_if_called(job: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("non-eligible jobs must not call solver")

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job_subprocess", fail_if_called)

    summary = run_solver_job_file(input_path, output_path, max_jobs=2)

    assert summary["status"] == "failed"
    assert summary["jobs_valid"] == 2
    assert summary["jobs_solved"] == 0
    assert summary["solver_failed"] == 0
    assert [row["solver_status"] for row in summary["results"]] == ["skipped", "skipped"]
    assert [row["quality"]["exclusion_reason"] for row in summary["results"]] == [
        "flop_solver_timeout_risk",
        "iterations_too_high_for_calibration",
    ]


def test_job_file_runner_jsonl_contains_skip_reason(tmp_path: Path, monkeypatch) -> None:
    job = generate_solver_jobs(count=1, seed=3, profile="paired_board_spot", iterations=1, timeout_s=5)[0]
    input_path = tmp_path / "jobs.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jobs(input_path, [job])

    def fail_if_called(job: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("non-eligible jobs must not call solver")

    monkeypatch.setattr("solver_jobs.job_file_runner.run_solver_job_subprocess", fail_if_called)

    run_solver_job_file(input_path, output_path, max_jobs=1)
    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert record["record_type"] == "solver_run_result"
    assert record["solver_status"] == "skipped"
    assert record["solver_result"] is None
    assert record["error"] == "flop_solver_timeout_risk"
    assert record["quality"]["is_label_candidate"] is False
    assert record["quality"]["exclusion_reason"] == "flop_solver_timeout_risk"
    assert "solver_skipped" in record["warnings"]
