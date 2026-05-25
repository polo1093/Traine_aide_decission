"""Run bounded solver jobs loaded from JSONL files."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solver_jobs.eligibility import evaluate_solver_eligibility
from solver_jobs.job_schema import validate_solver_job
from solver_jobs.job_runner import run_solver_job
from solver_jobs.subprocess_runner import run_solver_job_subprocess


DEFAULT_MAX_JOBS = 5
LARGE_RUN_LIMIT = 50
FILE_RUNNER_KEYS = (
    "status",
    "input_path",
    "output_path",
    "total_read",
    "jobs_valid",
    "jobs_invalid",
    "jobs_solved",
    "solver_failed",
    "dry_run",
    "results",
)
FORBIDDEN_LABEL_FIELDS = {"training_label", "gto_label", "label_action"}


def run_solver_job_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_jobs: int = DEFAULT_MAX_JOBS,
    start_index: int = 0,
    dry_run: bool = False,
    stop_on_error: bool = False,
    allow_large_run: bool = False,
    use_subprocess: bool = True,
) -> dict[str, Any]:
    """Validate and optionally solve a bounded JSONL file of solver jobs."""

    input_path_text = str(input_path)
    output_path_text = str(output_path)
    started = time.perf_counter()
    try:
        safe_max_jobs = _validate_max_jobs(max_jobs, allow_large_run=allow_large_run)
        safe_start_index = _validate_start_index(start_index)
        rows = _load_selected_jsonl(Path(input_path), safe_start_index, safe_max_jobs)

        records: list[dict[str, Any]] = []
        for line_index, raw_job, load_error in rows:
            if load_error is not None:
                records.append(
                    _record(
                        raw_job=None,
                        solver_job=None,
                        solver_result=None,
                        error=load_error,
                        warnings=[f"line_index:{line_index}"],
                        quality=_quality(None, None, "job_load_failed"),
                    )
                )
                if stop_on_error:
                    break
                continue

            record = _run_one_job(raw_job, dry_run=dry_run, line_index=line_index, use_subprocess=use_subprocess)
            records.append(record)
            if stop_on_error and record["error"] is not None:
                break

        write_error = _write_records_jsonl(records, Path(output_path))
        result = _summary(
            input_path=input_path_text,
            output_path=output_path_text,
            dry_run=dry_run,
            records=records,
        )
        if write_error is not None:
            result["status"] = "failed"
            result["results"].append(
                _record(
                    raw_job=None,
                    solver_job=None,
                    solver_result=None,
                    error=write_error,
                    warnings=[],
                    quality=_quality(None, None, "jsonl_write_failed"),
                )
            )
        result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return _without_extra_keys(result)
    except Exception as exc:  # noqa: BLE001 - public boundary returns stable data
        return _without_extra_keys(
            {
                "status": "failed",
                "input_path": input_path_text,
                "output_path": output_path_text,
                "total_read": 0,
                "jobs_valid": 0,
                "jobs_invalid": 0,
                "jobs_solved": 0,
                "solver_failed": 0,
                "dry_run": bool(dry_run),
                "results": [
                    _record(
                        raw_job=None,
                        solver_job=None,
                        solver_result=None,
                        error=_format_error(exc),
                        warnings=[],
                        quality=_quality(None, None, "runner_failed"),
                    )
                ],
            }
        )


def _run_one_job(raw_job: Any, *, dry_run: bool, line_index: int, use_subprocess: bool) -> dict[str, Any]:
    validation = validate_solver_job(raw_job) if isinstance(raw_job, dict) else {"status": "failed", "job": None, "error": "TypeError:job_must_be_object"}
    if validation["status"] != "ok":
        return _record(
            raw_job=raw_job if isinstance(raw_job, dict) else None,
            solver_job=None,
            solver_result=None,
            error=validation["error"],
            warnings=[f"line_index:{line_index}"],
            quality=_quality(None, None, "job_validation_failed"),
        )

    solver_job = validation["job"]
    if solver_job["source_type"] != "synthetic":
        return _record(
            raw_job=solver_job,
            solver_job=None,
            solver_result=None,
            error=f"unsupported_source_type_for_file_runner:{solver_job['source_type']}",
            warnings=[f"line_index:{line_index}"],
            quality=_quality(solver_job.get("iterations"), None, "unsupported_source_type"),
        )

    if dry_run:
        return _record(
            raw_job=solver_job,
            solver_job=solver_job,
            solver_result=None,
            error=None,
            warnings=[f"line_index:{line_index}", "dry_run"],
            quality=_quality(solver_job["iterations"], None, "dry_run"),
        )

    eligibility = evaluate_solver_eligibility(solver_job)
    if not eligibility["eligible"]:
        reason = eligibility["reason"] or "not_solver_eligible"
        return _record(
            raw_job=solver_job,
            solver_job=solver_job,
            solver_result=None,
            error=reason,
            warnings=[f"line_index:{line_index}", "solver_skipped", *eligibility.get("warnings", [])],
            quality=_quality(solver_job["iterations"], None, reason),
            solver_status="skipped",
        )

    if use_subprocess:
        subprocess_result = run_solver_job_subprocess(solver_job, timeout_s=solver_job["timeout_s"])
        solver_result = subprocess_result.get("solver_result")
        solver_status = subprocess_result.get("solver_status")
        error = subprocess_result.get("error")
        quality = subprocess_result.get("quality")
        warnings = [f"line_index:{line_index}", "subprocess"]
        if subprocess_result.get("solver_status") == "timeout":
            warnings.append("timeout")
    else:
        solver_result = run_solver_job(solver_job)
        solver_status = solver_result.get("status")
        error = None if solver_status == "ok" else solver_result.get("error") or "solver_failed"
        quality = solver_result.get("quality")
        warnings = [f"line_index:{line_index}", "direct_solver"]
    return _record(
        raw_job=solver_job,
        solver_job=solver_job,
        solver_result=solver_result,
        error=error,
        warnings=warnings,
        quality=_force_not_label_candidate(quality),
        solver_status=solver_status,
    )


def _load_selected_jsonl(path: Path, start_index: int, max_jobs: int) -> list[tuple[int, Any, str | None]]:
    rows: list[tuple[int, Any, str | None]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < start_index:
                continue
            if len(rows) >= max_jobs:
                break
            text = line.strip()
            if not text:
                rows.append((line_index, None, "jsonl_empty_line"))
                continue
            try:
                rows.append((line_index, json.loads(text), None))
            except json.JSONDecodeError as exc:
                rows.append((line_index, None, f"JSONDecodeError:{exc.msg}"))
    return rows


def _write_records_jsonl(records: list[dict[str, Any]], output_path: Path) -> str | None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return None
    except Exception as exc:  # noqa: BLE001
        return _format_error(exc)


def _summary(
    *,
    input_path: str,
    output_path: str,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    total_read = len(records)
    jobs_valid = sum(1 for record in records if record["solver_job"] is not None)
    jobs_invalid = sum(1 for record in records if record["solver_job"] is None)
    jobs_solved = sum(1 for record in records if _solver_status(record) == "ok")
    solver_failed = sum(
        1
        for record in records
        if record["solver_job"] is not None
        and record["error"] is not None
        and record["quality"].get("exclusion_reason") != "dry_run"
        and record.get("solver_status") != "skipped"
    )

    if total_read == 0:
        status = "failed"
    elif dry_run:
        status = "ok" if jobs_invalid == 0 else ("partial" if jobs_valid > 0 else "failed")
    elif jobs_invalid == 0 and solver_failed == 0 and jobs_solved == jobs_valid:
        status = "ok"
    elif jobs_solved > 0:
        status = "partial"
    else:
        status = "failed"

    return {
        "status": status,
        "input_path": input_path,
        "output_path": output_path,
        "total_read": total_read,
        "jobs_valid": jobs_valid,
        "jobs_invalid": jobs_invalid,
        "jobs_solved": jobs_solved,
        "solver_failed": solver_failed,
        "dry_run": bool(dry_run),
        "results": records,
    }


def _record(
    *,
    raw_job: dict[str, Any] | None,
    solver_job: dict[str, Any] | None,
    solver_result: dict[str, Any] | None,
    error: str | None,
    warnings: list[str],
    quality: dict[str, Any],
    solver_status: str | None = None,
) -> dict[str, Any]:
    job_for_ids = solver_job or raw_job or {}
    record = {
        "record_type": "solver_run_result",
        "solver_job_id": _optional_text(job_for_ids.get("solver_job_id")),
        "source_snapshot_id": _optional_text(job_for_ids.get("source_snapshot_id")),
        "source_type": _optional_text(job_for_ids.get("source_type")),
        "solver_job": solver_job,
        "solver_result": solver_result,
        "solver_status": solver_status or _status_from_solver_result(solver_result, error),
        "quality": _force_not_label_candidate(quality),
        "error": error,
        "warnings": list(warnings),
        "recorded_at": _utc_now(),
    }
    for field_name in FORBIDDEN_LABEL_FIELDS:
        record.pop(field_name, None)
    return record


def _status_from_solver_result(solver_result: dict[str, Any] | None, error: str | None) -> str:
    if solver_result is None:
        return "failed" if error else "skipped"
    status = solver_result.get("status")
    if status == "ok":
        return "ok"
    return "failed"


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
    quality.setdefault("iterations", None)
    quality.setdefault("exploitability_last", None)
    quality.setdefault("exclusion_reason", "labeling_disabled")
    return quality


def _solver_status(record: dict[str, Any]) -> str | None:
    if record.get("solver_status") is not None:
        return str(record["solver_status"])
    solver_result = record.get("solver_result")
    if not isinstance(solver_result, dict):
        return None
    status = solver_result.get("status")
    return str(status) if status is not None else None


def _validate_max_jobs(max_jobs: int, *, allow_large_run: bool) -> int:
    try:
        safe_max_jobs = int(max_jobs)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_jobs_must_be_integer") from exc
    if safe_max_jobs <= 0:
        raise ValueError("max_jobs_must_be_positive")
    if safe_max_jobs > LARGE_RUN_LIMIT and not allow_large_run:
        raise ValueError(f"max_jobs_exceeds_limit:{LARGE_RUN_LIMIT}")
    return safe_max_jobs


def _validate_start_index(start_index: int) -> int:
    try:
        safe_start_index = int(start_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("start_index_must_be_integer") from exc
    if safe_start_index < 0:
        raise ValueError("start_index_must_be_nonnegative")
    return safe_start_index


def _without_extra_keys(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in FILE_RUNNER_KEYS}


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
