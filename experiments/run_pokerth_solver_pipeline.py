"""Run PokerTH text through the strict solver pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokerth.pipeline import run_pokerth_solver_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PokerTH history through the solver pipeline")
    parser.add_argument("history_file")
    parser.add_argument("--hero-name", default="polo")
    parser.add_argument("--max-hands", type=int, default=10)
    parser.add_argument("--street", default="RIVER", choices=("FLOP", "TURN", "RIVER"))
    parser.add_argument("--to-call", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    to_call_by_street = None
    if args.to_call is not None:
        to_call_by_street = {args.street: args.to_call}

    result = run_pokerth_solver_pipeline(
        path=args.history_file,
        hero_name=args.hero_name,
        max_hands=args.max_hands,
        street=args.street,
        to_call_by_street=to_call_by_street,
        iterations=args.iterations,
        timeout_s=args.timeout_s,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
