from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.analyze_solver_run_results import analyze_solver_run_results


def write_jsonl(path: Path, rows: list[dict[str, Any] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def result(
    *,
    solver_status: str,
    profile: str = "random_turn_spot",
    iterations: int | None = 1,
    duration_ms: float | None = 10.0,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "solver_run_result",
        "solver_job_id": "job-1",
        "source_snapshot_id": "snapshot-1",
        "source_type": "synthetic",
        "solver_job": {
            "generation_profile": profile,
            "iterations": iterations,
            "source_type": "synthetic",
        },
        "solver_result": None,
        "solver_status": solver_status,
        "duration_ms": duration_ms,
        "quality": {
            "iterations": iterations,
            "exploitability_last": None,
            "is_label_candidate": False,
            "exclusion_reason": "iterations_too_low" if solver_status == "ok" else solver_status,
        },
        "error": error,
        "warnings": [],
        "recorded_at": "2026-05-25T00:00:00+00:00",
    }


def test_analyzes_jsonl_success(tmp_path: Path) -> None:
    path = tmp_path / "success.jsonl"
    write_jsonl(path, [result(solver_status="ok", duration_ms=12.5)])

    summary = analyze_solver_run_results([path])

    assert summary["total"] == 1
    assert summary["solved"] == 1
    assert summary["avg_duration_ms"] == 12.5
    assert summary["profiles"]["random_turn_spot"]["solved"] == 1
    assert summary["iterations"]["1"]["solved"] == 1


def test_analyzes_jsonl_with_skipped(tmp_path: Path) -> None:
    path = tmp_path / "skipped.jsonl"
    write_jsonl(path, [result(solver_status="skipped", error="profile_not_solver_safe")])

    summary = analyze_solver_run_results([path])

    assert summary["solved"] == 0
    assert summary["skipped"] == 1
    assert summary["errors"] == 0


def test_analyzes_jsonl_with_timeout(tmp_path: Path) -> None:
    path = tmp_path / "timeout.jsonl"
    write_jsonl(
        path,
        [
            result(
                solver_status="timeout",
                profile="random_river_spot",
                duration_ms=5000.0,
                error="solver_subprocess_timeout:5s",
            )
        ],
    )

    summary = analyze_solver_run_results([path])

    assert summary["timeouts"] == 1
    assert summary["profiles"]["random_river_spot"]["timeouts"] == 1
    assert summary["recommendation"].startswith("reduce_profiles_or_iterations")


def test_counts_mixed_records_correctly(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    write_jsonl(
        path,
        [
            result(solver_status="ok", duration_ms=10.0),
            result(solver_status="ok", duration_ms=20.0),
            result(solver_status="skipped", duration_ms=0.0, error="profile_not_solver_safe"),
            result(solver_status="failed", duration_ms=5.0, error="solver_failed"),
        ],
    )

    summary = analyze_solver_run_results([path])

    assert summary["total"] == 4
    assert summary["solved"] == 2
    assert summary["skipped"] == 1
    assert summary["timeouts"] == 0
    assert summary["errors"] == 1
    assert summary["avg_duration_ms"] == 8.75


def test_invalid_jsonl_line_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [result(solver_status="ok"), "{not json}"])

    summary = analyze_solver_run_results([path])

    assert summary["total"] == 2
    assert summary["solved"] == 1
    assert summary["errors"] == 1
    assert summary["profiles"]["unknown"]["errors"] == 1


def test_recommendation_is_stable_for_high_success_rate(tmp_path: Path) -> None:
    path = tmp_path / "stable.jsonl"
    write_jsonl(path, [result(solver_status="ok", duration_ms=1.0) for _ in range(5)])

    summary = analyze_solver_run_results([path])

    assert summary["recommendation"].startswith("stable_for_smoke_runs")
    assert "training_label" in summary["recommendation"]
