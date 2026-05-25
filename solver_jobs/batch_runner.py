"""Batch execution and JSONL persistence for bounded solver jobs."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solvers.poker_solver_adapter import check_solver_available
from solver_jobs.job_runner import run_solver_job
from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job


BATCH_RESULT_KEYS = (
    "status",
    "total",
    "mapped",
    "solved",
    "mapping_failed",
    "solver_failed",
    "failed_total",
    "results",
)


def run_solver_batch(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Map and solve a small list of snapshots with stable summary counts."""

    try:
        if not isinstance(snapshots, list):
            return _empty_batch_result("failed", [{"error": "snapshots_must_be_list"}])

        results: list[dict[str, Any]] = []
        for snapshot in snapshots:
            mapping = map_snapshot_to_solver_job(snapshot)
            source_snapshot_id = mapping.get("source_snapshot_id")
            if mapping.get("status") != "ok":
                results.append(
                    _row(
                        source_snapshot_id=source_snapshot_id,
                        mapping_status="failed",
                        solver_status="skipped",
                        solver_job=None,
                        solver_result=None,
                        error=mapping.get("error") or "mapping_failed",
                        warnings=mapping.get("warnings", []),
                        quality=_quality(None, None, "mapping_failed"),
                    )
                )
                continue

            solver_job = mapping["solver_job"]
            solver_result = run_solver_job(solver_job)
            solver_status = solver_result.get("status", "failed")
            error = None if solver_status == "ok" else solver_result.get("error") or "solver_failed"
            results.append(
                _row(
                    source_snapshot_id=source_snapshot_id,
                    mapping_status="ok",
                    solver_status=solver_status,
                    solver_job=solver_job,
                    solver_result=solver_result,
                    error=error,
                    warnings=mapping.get("warnings", []),
                    quality=_force_not_label_candidate(solver_result.get("quality")),
                )
            )
        return _batch_result(results)
    except Exception as exc:  # noqa: BLE001
        return _empty_batch_result("failed", [{"error": _format_error(exc)}])


def write_solver_batch_jsonl(
    batch_result: dict[str, Any],
    output_path: str | Path | None = None,
    *,
    solver_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write batch results as autonomous JSONL records."""

    started = time.perf_counter()
    try:
        if output_path is None:
            output_path = _default_output_path()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        solver = solver_info if solver_info is not None else _solver_info()
        recorded_at = _utc_now()
        records = [
            _jsonl_record(row, recorded_at=recorded_at, solver_info=solver)
            for row in batch_result.get("results", [])
        ]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

        return {
            "status": "ok",
            "output_path": str(path),
            "records_written": len(records),
            "error": None,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "output_path": str(output_path) if output_path is not None else None,
            "records_written": 0,
            "error": _format_error(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def _batch_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    mapped = sum(1 for row in results if row["mapping_status"] == "ok")
    solved = sum(1 for row in results if row["solver_status"] == "ok")
    mapping_failed = sum(1 for row in results if row["mapping_status"] == "failed")
    solver_failed = sum(1 for row in results if row["mapping_status"] == "ok" and row["solver_status"] != "ok")
    failed_total = mapping_failed + solver_failed
    if solved == 0:
        status = "failed"
    elif failed_total == 0:
        status = "ok"
    else:
        status = "partial"
    return {
        "status": status,
        "total": total,
        "mapped": mapped,
        "solved": solved,
        "mapping_failed": mapping_failed,
        "solver_failed": solver_failed,
        "failed_total": failed_total,
        "results": results,
    }


def _empty_batch_result(status: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        _row(
            source_snapshot_id=None,
            mapping_status="failed",
            solver_status="skipped",
            solver_job=None,
            solver_result=None,
            error=item.get("error") or "batch_failed",
            warnings=[],
            quality=_quality(None, None, "batch_failed"),
        )
        for item in errors
    ]
    result = _batch_result(rows)
    result["status"] = status
    return result


def _row(
    *,
    source_snapshot_id: Any,
    mapping_status: str,
    solver_status: str,
    solver_job: dict[str, Any] | None,
    solver_result: dict[str, Any] | None,
    error: str | None,
    warnings: list[str],
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_snapshot_id": None if source_snapshot_id is None else str(source_snapshot_id),
        "mapping_status": mapping_status,
        "solver_status": solver_status,
        "solver_job": solver_job,
        "solver_result": solver_result,
        "error": error,
        "warnings": list(warnings),
        "quality": _force_not_label_candidate(quality),
    }


def _jsonl_record(
    row: dict[str, Any],
    *,
    recorded_at: str,
    solver_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "solver_run_result",
        "recorded_at": recorded_at,
        "source_snapshot_id": row.get("source_snapshot_id"),
        "mapping_status": row.get("mapping_status"),
        "solver_status": row.get("solver_status"),
        "solver_job": row.get("solver_job"),
        "solver_result": row.get("solver_result"),
        "quality": _force_not_label_candidate(row.get("quality")),
        "error": row.get("error"),
        "warnings": list(row.get("warnings") or []),
        "solver": solver_info,
    }


def _quality(iterations: int | None, exploitability_last: Any, reason: str) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "exploitability_last": exploitability_last,
        "is_label_candidate": False,
        "exclusion_reason": reason,
    }


def _force_not_label_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _quality(None, None, "quality_missing")
    quality = dict(value)
    quality["is_label_candidate"] = False
    quality.setdefault("exclusion_reason", "labeling_disabled")
    quality.setdefault("iterations", None)
    quality.setdefault("exploitability_last", None)
    return quality


def _solver_info() -> dict[str, Any]:
    availability = check_solver_available()
    output = availability.get("output") or {}
    return {
        "solver_name": availability.get("solver_name", "PokerSolver"),
        "version": output.get("version"),
        "rust_backend_available": output.get("rust_backend_available"),
        "status": availability.get("status"),
        "error": availability.get("error") or output.get("rust_backend_error"),
    }


def _default_output_path() -> Path:
    return Path("outputs") / "solver_runs" / f"solver_run_{_timestamp_for_filename()}.jsonl"


def _timestamp_for_filename() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
