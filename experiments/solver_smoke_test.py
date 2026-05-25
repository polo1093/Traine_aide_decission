"""Run a bounded PokerSolver Rust smoke solve and print a stable JSON result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.poker_solver_adapter import solve_tiny_postflop_spot


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded PokerSolver smoke solve")
    parser.add_argument("--hero-hand", default="AhKh")
    parser.add_argument("--villain-hand", default="QdQc")
    parser.add_argument("--board", default="As 7c 2d Kh 5s")
    parser.add_argument("--pot", type=float, default=10.0)
    parser.add_argument("--stack", type=float, default=100.0)
    parser.add_argument("--bet-sizes", default="0.33")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backend", default="rust", choices=("rust", "python"))
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    result = solve_tiny_postflop_spot(
        args.hero_hand,
        args.villain_hand,
        board=args.board,
        pot=args.pot,
        stack=args.stack,
        bet_sizes=args.bet_sizes,
        iterations=args.iterations,
        backend=args.backend,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
