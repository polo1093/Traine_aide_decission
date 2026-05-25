"""Hard-timeout subprocess runner for one solver job."""

from __future__ import annotations

import json
from collections.abc import Mapping
import subprocess
import sys
import time
from typing import Any

from solver_jobs.job_schema import MAX_TIMEOUT_S, validate_solver_job


SUBPROCESS_RESULT_KEYS = (
    "status",
    "solver_job_id",
    "solver_status",
    "solver_result",
    "error",
    "duration_ms",
    "quality",
)


def run_solver_job_subprocess(
    job: dict[str, Any],
    *,
    timeout_s: float | None = None,
    python_executable: str | None = None,
    worker_module: str = "solver_jobs.solver_worker",
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run one solver job in a killable subprocess."""

    started = time.perf_counter()
    solver_job_id = _job_id(job)
    try:
        validation = validate_solver_job(job)
        if validation["status"] != "ok":
            return _result(
                started,
                solver_job_id,
                "failed",
                None,
                validation["error"],
                _quality(None, None, "job_validation_failed"),
            )

        normalized_job = validation["job"]
        timeout_value = _validate_timeout(timeout_s if timeout_s is not None else normalized_job.get("timeout_s"))
        timeout_label = f"{timeout_value:g}"
        completed = subprocess.run(
            [(python_executable or sys.executable), "-m", worker_module],
            input=json.dumps(normalized_job, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_value,
            check=False,
            env=None if env is None else dict(env),
            cwd=cwd,
        )
        if completed.returncode != 0:
            error = _worker_error(completed.stdout, completed.stderr, completed.returncode)
            return _result(
                started,
                normalized_job["solver_job_id"],
                "failed",
                None,
                error,
                _quality(normalized_job["iterations"], None, "worker_failed"),
            )

        worker_result = _parse_worker_stdout(completed.stdout)
        if not isinstance(worker_result, dict):
            return _result(
                started,
                normalized_job["solver_job_id"],
                "failed",
                None,
                "worker_output_not_json_object",
                _quality(normalized_job["iterations"], None, "worker_failed"),
            )

        solver_status = "ok" if worker_result.get("status") == "ok" else "failed"
        error = None if solver_status == "ok" else worker_result.get("error") or "solver_failed"
        return _result(
            started,
            normalized_job["solver_job_id"],
            solver_status,
            worker_result,
            error,
            _force_not_label_candidate(worker_result.get("quality")),
        )
    except subprocess.TimeoutExpired:
        return _result(
            started,
            solver_job_id,
            "timeout",
            None,
            f"solver_subprocess_timeout:{locals().get('timeout_label', _format_timeout(timeout_s))}s",
            _quality(None, None, "timeout"),
        )
    except Exception as exc:  # noqa: BLE001 - stable public boundary
        return _result(
            started,
            solver_job_id,
            "failed",
            None,
            _format_error(exc),
            _quality(None, None, "subprocess_runner_failed"),
        )


def _result(
    started: float,
    solver_job_id: str | None,
    solver_status: str,
    solver_result: dict[str, Any] | None,
    error: str | None,
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ok" if solver_status == "ok" and error is None else "failed",
        "solver_job_id": solver_job_id,
        "solver_status": solver_status,
        "solver_result": solver_result,
        "error": error,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "quality": _force_not_label_candidate(quality),
    }


def _validate_timeout(value: Any) -> float:
    if value is None:
        raise ValueError("timeout_s_required")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_s_must_be_number") from exc
    if timeout <= 0:
        raise ValueError("timeout_s_must_be_positive")
    if timeout > MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s_exceeds_limit:{MAX_TIMEOUT_S:g}")
    return timeout


def _parse_worker_stdout(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    return json.loads(text.splitlines()[-1])


def _worker_error(stdout: str, stderr: str, returncode: int) -> str:
    parsed = None
    with _suppress_json_error():
        parsed = _parse_worker_stdout(stdout)
    if isinstance(parsed, dict) and parsed.get("error"):
        return str(parsed["error"])
    stderr_text = (stderr or "").strip()
    if stderr_text:
        return f"worker_failed:{returncode}:{stderr_text[-500:]}"
    return f"worker_failed:{returncode}"


class _suppress_json_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return exc_type in {json.JSONDecodeError, TypeError}


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


def _job_id(job: Any) -> str | None:
    if isinstance(job, dict) and job.get("solver_job_id") is not None:
        return str(job["solver_job_id"])
    return None


def _format_timeout(value: Any) -> str:
    if value is None:
        return "job"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
