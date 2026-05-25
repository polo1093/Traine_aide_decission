"""Run one stable synthetic solve and inspect whether root strategy is exposed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver_jobs.strategy_extractor import extract_root_strategy
from solver_jobs.subprocess_runner import run_solver_job_subprocess
from synthetic.spot_generator import generate_solver_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect solver_result strategy shape for one synthetic smoke job."
    )
    parser.add_argument("--profile", required=True, choices=["random_turn_spot", "random_river_spot"])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job-index", type=int, default=0)
    args = parser.parse_args()

    job = generate_solver_jobs(
        count=args.job_index + 1,
        seed=args.seed,
        profile=args.profile,
        iterations=args.iterations,
        timeout_s=args.timeout_s,
    )[args.job_index]
    solver_result = run_solver_job_subprocess(job, timeout_s=job["timeout_s"])
    run_result = _solver_run_result(job, solver_result)
    extraction = extract_root_strategy(run_result)

    output = {
        "status": "ok" if solver_result.get("solver_status") == "ok" else "failed",
        "profile": args.profile,
        "iterations": args.iterations,
        "timeout_s": args.timeout_s,
        "solver_job_id": job["solver_job_id"],
        "solver_result_abridged": _abridge_solver_result(solver_result),
        "strategy_extraction": extraction,
        "labeling": {
            "is_label_candidate": False,
            "training_label_created": False,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if solver_result.get("solver_status") == "ok" else 1


def _solver_run_result(job: dict[str, Any], solver_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "solver_run_result",
        "solver_job_id": job.get("solver_job_id"),
        "source_snapshot_id": job.get("source_snapshot_id"),
        "source_type": job.get("source_type"),
        "solver_job": job,
        "solver_result": solver_result.get("solver_result"),
        "solver_status": solver_result.get("solver_status"),
        "duration_ms": solver_result.get("duration_ms"),
        "quality": _force_not_label_candidate(solver_result.get("quality")),
        "error": solver_result.get("error"),
        "warnings": ["strategy_inspection", "subprocess"],
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _abridge_solver_result(result: dict[str, Any]) -> dict[str, Any]:
    nested = result.get("solver_result")
    output = nested.get("output") if isinstance(nested, dict) else None
    return {
        "status": result.get("status"),
        "solver_status": result.get("solver_status"),
        "duration_ms": result.get("duration_ms"),
        "error": result.get("error"),
        "solver_result_keys": sorted(nested.keys()) if isinstance(nested, dict) else None,
        "output_keys": sorted(output.keys()) if isinstance(output, dict) else None,
        "output": _abridge_output(output),
        "quality": _force_not_label_candidate(result.get("quality")),
    }


def _abridge_output(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None
    return {
        key: output.get(key)
        for key in (
            "backend",
            "iterations",
            "game_value",
            "exploitability_history",
            "strategy_entry_count",
            "root_strategy",
            "action_frequencies",
            "average_strategy",
        )
        if key in output
    }


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
    quality.setdefault("exclusion_reason", "strategy_inspection")
    return quality


if __name__ == "__main__":
    raise SystemExit(main())
