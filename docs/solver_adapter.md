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
- `solve_tiny_postflop_spot(...)`

The first target is stable Python import and simple equity. The postflop solve
functions are best effort and should be treated as smoke tests, not production
decision engines.

## Known Limits

- `psutil` is required by PokerSolver and may be missing.
- Rust, Cargo, maturin, and the MSVC C++ linker are required for the heavy
  Rust backend on Windows.
- The Rust backend can be unavailable even when Python equity works if
  `poker_solver._rust` has not been built.
- Full postflop solves can be slow or memory-heavy.
- Returned solver output is for offline experiments only.
- This adapter does not train a model and is not connected to live decisions.

## Rust Backend Build

PokerSolver exposes its heavy backend as:

```text
poker_solver._rust
```

On Windows, install Rust stable and the Visual Studio C++ build tools first:

```powershell
winget install --id Rustlang.Rustup -e
rustup default stable

winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

In the Visual Studio installer, include `Desktop development with C++` with a
recent MSVC toolset, Windows 10/11 SDK, and C++ build tools.

Install only the Python build frontend:

```powershell
python -m pip install "maturin>=1.7,<2.0"
```

Build PokerSolver from the local checkout:

```powershell
cd "projet importer\poker_solver-main"
python -m pip install -e .
```

If `link.exe` is not visible in a normal PowerShell after installing Build
Tools, use `x64 Native Tools Command Prompt for VS 2022` or
`Developer PowerShell for VS 2022`. From plain PowerShell, the equivalent is:

```powershell
cmd.exe /v:on /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set "PATH=!USERPROFILE!\.cargo\bin;!PATH!" && python -m pip install -e .'
```

Verify the backend:

```powershell
python -c "import poker_solver._rust; print('ok')"
python -c "from solvers import poker_solver_adapter as a; print(a.check_solver_available())"
```

Normal PowerShell often cannot see `link.exe` because Visual Studio's compiler,
linker, SDK, and library paths are injected by the Visual Studio developer
environment scripts. Installing Build Tools puts the files on disk; opening a
developer terminal or running `vcvars64.bat` wires those paths into the current
shell.

## Tiny Solver Smoke Test

Use `solve_tiny_postflop_spot(...)` to verify that the heavy backend can be
called on a concrete postflop spot with bounded parameters:

```python
from solvers.poker_solver_adapter import solve_tiny_postflop_spot

result = solve_tiny_postflop_spot(
    "AhKh",
    "QdQc",
    board="As 7c 2d Kh 5s",
    pot=10,
    stack=100,
    bet_sizes=(0.33,),
    iterations=10,
    backend="rust",
    timeout_s=5,
)
```

The function returns the stable adapter shape. On success, `output` includes:

```python
{
    "backend": "rust",
    "iterations": 10,
    "game_value": 0.0,
    "exploitability_history": [...],
    "strategy_entry_count": 0,
}
```

Guard rails:

- default iterations are low (`10`) and capped at `100`;
- default timeout is `5` seconds;
- default bet menu is tiny (`0.33` pot);
- villain ranges are intentionally not supported by this smoke helper yet;
- Rust backend failures and invalid inputs return `status: "failed"`.

You can also run the command-line smoke script:

```powershell
python experiments/solver_smoke_test.py --iterations 10 --timeout-s 5
```

This is only a technical validation that the solver can be called. It is not a
strategy-quality solve, and `iterations=10` must not be interpreted as a useful
poker strategy or ML label.

## Test Command

From the project root:

```powershell
python -m pytest tests/test_poker_solver_adapter.py
```

The adapter tests include a tiny Rust postflop smoke solve. They do not train a
model, generate datasets, or run production-scale postflop solves.
