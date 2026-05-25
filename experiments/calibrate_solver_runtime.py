"""Calibrate small synthetic solver jobs through the subprocess runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver_jobs.subprocess_runner import run_solver_job_subprocess
from synthetic.spot_generator import SUPPORTED_PROFILES, generate_solver_jobs


CALIBRATION_PROFILES = (
    "random_flop_spot",
    "random_turn_spot",
    "random_river_spot",
    "drawy_board_spot",
    "paired_board_spot",
    "top_pair_spot",
    "two_pair_plus_spot",
    "made_hand_vs_draw_spot",
)
DEFAULT_MAX_TOTAL_JOBS = 20
LARGE_CALIBRATION_LIMIT = 50
MAX_CALIBRATION_ITERATIONS = 25
MAX_CALIBRATION_TIMEOUT_S = 10.0


def run_calibration(
    *,
    profiles: list[str] | tuple[str, ...] | None = None,
    jobs_per_profile: int = 1,
    iterations: list[int] | tuple[int, ...] = (1,),
    timeout_s: float = 5.0,
    seed: int = 42,
    max_total_jobs: int = DEFAULT_MAX_TOTAL_JOBS,
    allow_large_run: bool = False,
    solver_func: Callable[..., dict[str, Any]] = run_solver_job_subprocess,
) -> dict[str, Any]:
    """Run a tiny bounded runtime calibration and return a JSON-safe summary."""

    started = time.perf_counter()
    try:
        safe_profiles = _validate_profiles(profiles)
        safe_jobs_per_profile = _validate_positive_int(jobs_per_profile, "jobs_per_profile")
        safe_iterations = _validate_iterations(iterations)
        safe_timeout_s = _validate_timeout(timeout_s)
        safe_seed = int(seed)
        safe_max_total_jobs = _validate_positive_int(max_total_jobs, "max_total_jobs")
        planned = len(safe_profiles) * safe_jobs_per_profile * len(safe_iterations)
        _validate_planned_count(planned, safe_max_total_jobs, allow_large_run=allow_large_run)

        results: list[dict[str, Any]] = []
        for profile in safe_profiles:
            for iteration_count in safe_iterations:
                jobs = generate_solver_jobs(
                    count=safe_jobs_per_profile,
                    seed=safe_seed,
                    profile=profile,
                    iterations=iteration_count,
                    timeout_s=safe_timeout_s,
                )
                for job_index, job in enumerate(jobs):
                    job = _make_iteration_specific_job(job, iteration_count)
                    results.append(_run_one_calibration_job(job, profile, iteration_count, job_index, solver_func))

        summary = _summary(results, safe_profiles)
        summary.update(
            {
                "status": "ok",
                "seed": safe_seed,
                "jobs_per_profile": safe_jobs_per_profile,
                "iterations": list(safe_iterations),
                "timeout_s": safe_timeout_s,
                "max_total_jobs": safe_max_total_jobs,
                "total_planned": planned,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "results": results,
            }
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - CLI boundary should stay stable
        return {
            "status": "failed",
            "seed": seed,
            "jobs_per_profile": jobs_per_profile,
            "iterations": list(iterations) if isinstance(iterations, (list, tuple)) else iterations,
            "timeout_s": timeout_s,
            "max_total_jobs": max_total_jobs,
            "total_planned": 0,
            "total_run": 0,
            "solved": 0,
            "successes": 0,
            "timeouts": 0,
            "errors": 0,
            "avg_success_duration_ms": None,
            "profiles": {},
            "iterations_summary": {},
            "profile_iterations": {},
            "profiles_too_heavy": [],
            "profiles_exploitable_for_smoke": [],
            "recommended_parameters": _recommended_parameters([]),
            "recommendations": _recommendations({}, {}, {}),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": _format_error(exc),
            "results": [],
        }


def write_calibration_jsonl(summary: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Write calibration result rows to JSONL."""

    started = time.perf_counter()
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = summary.get("results", [])
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return {
            "status": "ok",
            "output_path": str(path),
            "records_written": len(rows),
            "error": None,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "output_path": str(path),
            "records_written": 0,
            "error": _format_error(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate tiny synthetic solver runtime with hard timeouts.")
    parser.add_argument("--profiles", nargs="*", default=list(CALIBRATION_PROFILES), choices=sorted(CALIBRATION_PROFILES))
    parser.add_argument("--jobs-per-profile", type=int, default=1)
    parser.add_argument("--iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total-jobs", type=int, default=DEFAULT_MAX_TOTAL_JOBS)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL path for per-job calibration records.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON path for the full calibration summary.")
    args = parser.parse_args()

    summary = run_calibration(
        profiles=args.profiles,
        jobs_per_profile=args.jobs_per_profile,
        iterations=args.iterations,
        timeout_s=args.timeout_s,
        seed=args.seed,
        max_total_jobs=args.max_total_jobs,
        allow_large_run=args.allow_large_run,
    )
    write_result = None
    if args.output is not None:
        write_result = write_calibration_jsonl(summary, args.output)
    summary_write_result = None
    if args.summary_output is not None:
        summary_write_result = write_calibration_summary_json(summary, args.summary_output)

    printable = dict(summary)
    printable["results"] = printable["results"][:5]
    printable["results_truncated"] = len(summary["results"]) > 5
    if write_result is not None:
        printable["write"] = write_result
    if summary_write_result is not None:
        printable["summary_write"] = summary_write_result
    print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))
    writes_ok = (write_result is None or write_result["status"] == "ok") and (
        summary_write_result is None or summary_write_result["status"] == "ok"
    )
    return 0 if summary["status"] == "ok" and writes_ok else 1


def write_calibration_summary_json(summary: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Write the full calibration summary to JSON."""

    started = time.perf_counter()
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return {
            "status": "ok",
            "output_path": str(path),
            "error": None,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "output_path": str(path),
            "error": _format_error(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def _run_one_calibration_job(
    job: dict[str, Any],
    profile: str,
    iterations: int,
    job_index: int,
    solver_func: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    result = solver_func(job, timeout_s=job["timeout_s"])
    solver_status = result.get("solver_status")
    status = "ok" if solver_status == "ok" else ("timeout" if solver_status == "timeout" else "failed")
    quality = _force_not_label_candidate(result.get("quality"))
    return {
        "record_type": "solver_runtime_calibration",
        "profile": profile,
        "street": job["street"],
        "iterations": iterations,
        "job_index": job_index,
        "solver_job_id": job["solver_job_id"],
        "source_snapshot_id": job["source_snapshot_id"],
        "status": status,
        "solver_status": solver_status,
        "duration_ms": result.get("duration_ms"),
        "error": result.get("error"),
        "quality": quality,
        "is_label_candidate": False,
    }


def _summary(results: list[dict[str, Any]], profiles: tuple[str, ...]) -> dict[str, Any]:
    successes = [row for row in results if row["status"] == "ok"]
    timeouts = [row for row in results if row["status"] == "timeout"]
    errors = [row for row in results if row["status"] == "failed"]
    profile_summary = {profile: _profile_summary(profile, results) for profile in profiles}
    iteration_summary = _iteration_summary(results)
    profile_iteration_summary = _profile_iteration_summary(results)
    too_heavy = [
        profile
        for profile, item in profile_summary.items()
        if item["timeouts"] > 0 or item["solved"] == 0
    ]
    smoke_ready = [
        profile
        for profile, item in profile_summary.items()
        if item["total"] > 0 and item["solved"] == item["total"]
    ]
    return {
        "total_run": len(results),
        "solved": len(successes),
        "successes": len(successes),
        "timeouts": len(timeouts),
        "errors": len(errors),
        "avg_success_duration_ms": _average([row["duration_ms"] for row in successes]),
        "profiles": profile_summary,
        "iterations_summary": iteration_summary,
        "profile_iterations": profile_iteration_summary,
        "profiles_too_heavy": too_heavy,
        "profiles_exploitable_for_smoke": smoke_ready,
        "recommended_parameters": _recommended_parameters(results),
        "recommendations": _recommendations(profile_summary, iteration_summary, profile_iteration_summary),
    }


def _profile_summary(profile: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row["profile"] == profile]
    return _run_summary(rows)


def _iteration_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    iterations = sorted({row["iterations"] for row in results})
    return {str(iterations_value): _run_summary([row for row in results if row["iterations"] == iterations_value]) for iterations_value in iterations}


def _profile_iteration_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    profiles = sorted({row["profile"] for row in results})
    iterations = sorted({row["iterations"] for row in results})
    return {
        profile: {
            str(iterations_value): _run_summary(
                [row for row in results if row["profile"] == profile and row["iterations"] == iterations_value]
            )
            for iterations_value in iterations
        }
        for profile in profiles
    }


def _run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["status"] == "ok"]
    timeouts = [row for row in rows if row["status"] == "timeout"]
    errors = [row for row in rows if row["status"] == "failed"]
    total = len(rows)
    return {
        "total": total,
        "solved": len(successes),
        "successes": len(successes),
        "timeouts": len(timeouts),
        "errors": len(errors),
        "success_rate": _rate(len(successes), total),
        "timeout_rate": _rate(len(timeouts), total),
        "avg_success_duration_ms": _average([row["duration_ms"] for row in successes]),
        "avg_duration_ms": _average([row["duration_ms"] for row in rows]),
        "recommendation": _recommend_group(rows),
    }


def _recommended_parameters(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in results if row["status"] == "ok"]
    if not successes:
        return {
            "max_jobs": 1,
            "iterations": 1,
            "timeout_s": 5.0,
            "note": "No successful calibration rows yet; keep runs at one job until a profile succeeds.",
        }
    safest_iterations = min(row["iterations"] for row in successes)
    return {
        "max_jobs": min(5, len(successes)),
        "iterations": safest_iterations,
        "timeout_s": 5.0,
        "note": "Use only profiles listed in profiles_exploitable_for_smoke for smoke solves.",
    }


def _recommendations(
    profile_summary: dict[str, dict[str, Any]],
    iteration_summary: dict[str, dict[str, Any]],
    profile_iteration_summary: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    stable_profile_iterations: dict[str, list[int]] = {}
    avoid_profile_iterations: dict[str, list[int]] = {}
    for profile, per_iteration in profile_iteration_summary.items():
        stable_profile_iterations[profile] = []
        avoid_profile_iterations[profile] = []
        for iteration_text, item in per_iteration.items():
            if item["recommendation"] == "stable_for_solver_batch":
                stable_profile_iterations[profile].append(int(iteration_text))
            elif item["recommendation"] != "no_rows":
                avoid_profile_iterations[profile].append(int(iteration_text))

    return {
        "profiles": {profile: item["recommendation"] for profile, item in profile_summary.items()},
        "iterations": {iteration: item["recommendation"] for iteration, item in iteration_summary.items()},
        "stable_profile_iterations": stable_profile_iterations,
        "avoid_profile_iterations": avoid_profile_iterations,
        "policy": {
            "not_recommended_success_rate_below": 0.80,
            "not_recommended_timeout_rate_above": 0.10,
            "avoid_large_batch_avg_duration_ms_above": 3000.0,
            "stable_success_rate_above": 0.90,
            "stable_avg_duration_ms_at_or_below": 3000.0,
        },
        "labeling": "is_label_candidate remains false; no training_label is created.",
    }


def _recommend_group(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    if total == 0:
        return "no_rows"
    solved = sum(1 for row in rows if row["status"] == "ok")
    timeouts = sum(1 for row in rows if row["status"] == "timeout")
    success_rate = solved / total
    timeout_rate = timeouts / total
    avg_duration = _average([row["duration_ms"] for row in rows])
    if success_rate < 0.80:
        return "not_recommended_success_rate_below_80_percent"
    if timeout_rate > 0.10:
        return "not_recommended_timeout_rate_above_10_percent"
    if avg_duration is not None and avg_duration > 3000.0:
        return "avoid_large_batches_avg_duration_above_3000_ms"
    if success_rate > 0.90 and (avg_duration is None or avg_duration <= 3000.0):
        return "stable_for_solver_batch"
    return "usable_with_caution"


def _make_iteration_specific_job(job: dict[str, Any], iterations: int) -> dict[str, Any]:
    unique_job = dict(job)
    suffix = f"_iter_{iterations}"
    unique_job["solver_job_id"] = f"{job['solver_job_id']}{suffix}"
    unique_job["source_snapshot_id"] = f"{job['source_snapshot_id']}{suffix}"
    unique_job["iterations"] = iterations
    unique_job["label_intent"] = "solver_smoke"
    return unique_job


def _validate_profiles(profiles: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = tuple(profiles or CALIBRATION_PROFILES)
    if not selected:
        raise ValueError("profiles_required")
    unsupported = [profile for profile in selected if profile not in SUPPORTED_PROFILES]
    if unsupported:
        raise ValueError(f"unsupported_profiles:{','.join(unsupported)}")
    return selected


def _validate_iterations(iterations: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(iterations, (list, tuple)) or not iterations:
        raise ValueError("iterations_required")
    normalized = tuple(_validate_positive_int(value, "iterations") for value in iterations)
    too_high = [value for value in normalized if value > MAX_CALIBRATION_ITERATIONS]
    if too_high:
        raise ValueError(f"iterations_exceeds_calibration_limit:{MAX_CALIBRATION_ITERATIONS}")
    return normalized


def _validate_timeout(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if timeout <= 0:
        raise ValueError("timeout_s_must_be_positive")
    if timeout > MAX_CALIBRATION_TIMEOUT_S:
        raise ValueError(f"timeout_s_exceeds_limit:{MAX_CALIBRATION_TIMEOUT_S:g}")
    return timeout


def _validate_planned_count(planned: int, max_total_jobs: int, *, allow_large_run: bool) -> None:
    if planned > max_total_jobs:
        raise ValueError(f"calibration_plan_exceeds_max_total_jobs:{max_total_jobs}")
    if planned > LARGE_CALIBRATION_LIMIT and not allow_large_run:
        raise ValueError(f"calibration_plan_exceeds_large_run_limit:{LARGE_CALIBRATION_LIMIT}")


def _validate_positive_int(value: Any, field_name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name}_must_be_positive")
    return number


def _average(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 3)


def _rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(count / total, 4)


def _force_not_label_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "iterations": None,
            "exploitability_last": None,
            "is_label_candidate": False,
            "exclusion_reason": "quality_missing",
        }
    quality = dict(value)
    quality["is_label_candidate"] = False
    quality.setdefault("iterations", None)
    quality.setdefault("exploitability_last", None)
    quality.setdefault("exclusion_reason", "labeling_disabled")
    return quality


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
