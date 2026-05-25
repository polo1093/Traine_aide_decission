"""Generate deterministic synthetic solver_job_v1 JSONL without solving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.spot_generator import DEFAULT_MAX_COUNT, SUPPORTED_PROFILES, generate_solver_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic solver jobs only; this script never calls run_solver_job."
    )
    parser.add_argument("--count", type=int, required=True, help=f"Number of jobs to generate, max {DEFAULT_MAX_COUNT}.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic generation seed.")
    parser.add_argument("--profile", required=True, choices=sorted(SUPPORTED_PROFILES), help="Generation profile.")
    parser.add_argument("--iterations", type=int, default=25, help="Solver iterations stored in each job, max 100.")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Solver timeout stored in each job, max 10.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    try:
        jobs = generate_solver_jobs(
            count=args.count,
            seed=args.seed,
            profile=args.profile,
            iterations=args.iterations,
            timeout_s=args.timeout_s,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for job in jobs:
                handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    except ValueError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    print(json.dumps({"status": "ok", "jobs_written": len(jobs), "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
