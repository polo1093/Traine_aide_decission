"""CLI for generating and validating solver-labeled poker ML datasets."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

try:
    from .solver_adapter import ImportSolverAdapter, MockSolverAdapter
    from .solver_dataset_validator import validate_jsonl
    from .solver_dataset_writer import write_solver_dataset
    from .solver_spot_generator import generate_synthetic_spots, iter_existing_spots
except ImportError:  # pragma: no cover
    from solver_adapter import ImportSolverAdapter, MockSolverAdapter
    from solver_dataset_validator import validate_jsonl
    from solver_dataset_writer import write_solver_dataset
    from solver_spot_generator import generate_synthetic_spots, iter_existing_spots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="solver-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-solver-dataset")
    generate.add_argument("--output", default="solver_training_dataset.jsonl")
    generate.add_argument("--n-spots", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--mode", choices=["synthetic", "relabel-existing"], default="synthetic")
    generate.add_argument("--solver", default="mock")
    generate.add_argument("--external-solver-module", default=None)
    generate.add_argument("--input-existing", default=None)

    validate = subparsers.add_parser("validate-solver-dataset")
    validate.add_argument("--input", required=True)
    validate.add_argument("--example", default="example_training_dataset.jsonl")

    args = parser.parse_args(argv)
    if args.command == "generate-solver-dataset":
        return _generate(args)
    if args.command == "validate-solver-dataset":
        return _validate(args)
    parser.error("unknown command")
    return 2


def _generate(args: argparse.Namespace) -> int:
    solver = _solver_from_args(args)
    if args.mode == "synthetic":
        spots = generate_synthetic_spots(args.n_spots, seed=args.seed)
    else:
        if not args.input_existing:
            raise SystemExit("--input-existing is required with --mode relabel-existing")
        spots = itertools.islice(iter_existing_spots(args.input_existing), args.n_spots)

    result = write_solver_dataset(spots, solver, args.output, n_rows=args.n_spots)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if int(result["written"]) == args.n_spots else 1


def _validate(args: argparse.Namespace) -> int:
    result = validate_jsonl(args.input, example_path=args.example)
    print(json.dumps({"ok": result.ok, "lines_checked": result.lines_checked, "errors": result.errors}, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def _solver_from_args(args: argparse.Namespace) -> Any:
    if args.solver == "mock":
        return MockSolverAdapter()
    if args.solver == "external":
        if not args.external_solver_module:
            raise SystemExit("--external-solver-module is required with --solver external")
        return ImportSolverAdapter(args.external_solver_module)
    raise SystemExit(f"unsupported solver: {args.solver}")


if __name__ == "__main__":
    raise SystemExit(main())

