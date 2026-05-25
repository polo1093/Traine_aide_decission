from __future__ import annotations

try:
    from .solver_dataset_cli import main
except ImportError:  # pragma: no cover
    from solver_dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main())

