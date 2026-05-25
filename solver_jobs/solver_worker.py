"""Single-job solver worker for subprocess isolation."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from solver_jobs.job_runner import run_solver_job


WORKER_RESULT_KEYS = ("status", "solver_job_id", "input", "output", "error", "duration_ms", "quality")


def run_worker_payload(
    job: dict[str, Any],
    *,
    solver_func: Callable[[dict[str, Any]], dict[str, Any]] = run_solver_job,
) -> dict[str, Any]:
    """Run one solver job and return a stable JSON-serializable result."""

    try:
        result = solver_func(job)
        if not isinstance(result, dict):
            return _failed_result(job, "worker_solver_returned_non_mapping")
        return _normalize_solver_result(job, result)
    except Exception as exc:  # noqa: BLE001 - worker boundary must be stable
        return _failed_result(job, _format_error(exc))


def main() -> int:
    try:
        payload = sys.stdin.read()
        job = json.loads(payload)
        if not isinstance(job, dict):
            result = _failed_result({}, "job_must_be_json_object")
        else:
            result = run_worker_payload(job)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        result = _failed_result({}, _format_error(exc))
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
        sys.stdout.write("\n")
        return 1


def _normalize_solver_result(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: result.get(key) for key in WORKER_RESULT_KEYS}
    normalized["status"] = "ok" if result.get("status") == "ok" else "failed"
    normalized["solver_job_id"] = result.get("solver_job_id") or job.get("solver_job_id")
    normalized["input"] = result.get("input") if isinstance(result.get("input"), dict) else job
    normalized["output"] = result.get("output") if isinstance(result.get("output"), dict) else None
    normalized["error"] = None if normalized["status"] == "ok" else result.get("error") or "solver_failed"
    normalized["duration_ms"] = _float_or_zero(result.get("duration_ms"))
    normalized["quality"] = _force_not_label_candidate(result.get("quality"))
    return normalized


def _failed_result(job: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "solver_job_id": job.get("solver_job_id"),
        "input": job,
        "output": None,
        "error": error,
        "duration_ms": 0.0,
        "quality": _quality(None, None, "worker_failed"),
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
    quality.setdefault("iterations", None)
    quality.setdefault("exploitability_last", None)
    quality.setdefault("exclusion_reason", "labeling_disabled")
    return quality


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
