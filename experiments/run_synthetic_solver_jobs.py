"""Run a small bounded solver batch from synthetic solver_job_v1 JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver_jobs.job_file_runner import DEFAULT_MAX_JOBS, LARGE_RUN_LIMIT, run_solver_job_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and solve a small sample of synthetic solver jobs."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path containing solver_job_v1 records.")
    parser.add_argument("--output", required=True, help="Output JSONL path for solver_run_result records.")
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS, help="Maximum jobs to process, default 5.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based JSONL line index to start from.")
    parser.add_argument("--dry-run", action="store_true", help="Validate jobs and write records without calling solver.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first invalid job or solver failure.")
    parser.add_argument(
        "--direct-solver",
        action="store_true",
        help="Debug only: call the solver in-process instead of through the hard-timeout subprocess.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=f"Permit --max-jobs above {LARGE_RUN_LIMIT}. Use deliberately.",
    )
    args = parser.parse_args()

    summary = run_solver_job_file(
        args.input,
        args.output,
        max_jobs=args.max_jobs,
        start_index=args.start_index,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        allow_large_run=args.allow_large_run,
        use_subprocess=not args.direct_solver,
    )
    printable = dict(summary)
    printable["results"] = printable["results"][:3]
    printable["results_truncated"] = len(summary["results"]) > 3
    print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))
    return 0 if summary["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
