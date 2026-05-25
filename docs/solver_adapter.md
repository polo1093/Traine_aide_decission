# PokerSolver Adapter

This project uses PokerSolver through one local adapter:

```text
solvers/poker_solver_adapter.py
```

The adapter exists so offline ML code does not import PokerSolver directly in
many places. All public functions return the same shape:

```python
{
    "status": "ok" or "failed",
    "solver_name": "PokerSolver",
    "input": {...},
    "output": {...} or None,
    "error": None or "...",
    "duration_ms": 0.0,
}
```

## Local PokerSolver Path

The default local checkout is:

```text
projet importer/poker_solver-main
```

Because the path contains a space, the adapter resolves it with
`pathlib.Path`:

```python
Path(__file__).resolve().parents[1] / "projet importer" / "poker_solver-main"
```

No shell command or string-concatenated path is required.

You can override the path with:

```powershell
$env:POKER_SOLVER_PATH="C:\Users\polo\Pictures\Cours Info\Programme\Traine_aide_decission\projet importer\poker_solver-main"
```

or by passing `solver_path=...` to adapter functions.

## sys.path Behavior

PokerSolver is a copied source checkout, not a normal dependency of this ML
project yet. The adapter adds the resolved PokerSolver root to `sys.path` only
while importing `poker_solver`, and removes it immediately after import when it
was inserted by the adapter.

The imported Python module remains cached in `sys.modules`, which is normal
Python import behavior.

## Supported Functions

- `check_solver_available()`
- `compute_equity_hand_vs_hand(...)`
- `compute_equity_hand_vs_range(...)`
- `solve_simple_postflop_spot(...)`

The first target is stable Python import and simple equity. The postflop solve
function is best effort and should be treated as a smoke test, not a production
decision engine.

## Known Limits

- `psutil` is required by PokerSolver and may be missing.
- Rust, Cargo and maturin are not installed or forced by this step.
- The Rust backend can be unavailable even when Python equity works.
- Full postflop solves can be slow or memory-heavy.
- Returned solver output is for offline experiments only.
- This adapter does not train a model and is not connected to live decisions.

## Test Command

From the project root:

```powershell
python -m pytest tests/test_poker_solver_adapter.py
```

