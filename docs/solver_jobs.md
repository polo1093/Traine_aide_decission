# Solver Jobs

Solver jobs are the small, explicit bridge between a poker snapshot and the
PokerSolver adapter. They make every solver call traceable before this project
starts generating offline labels.

No model is trained here. No dataset is generated here. This layer only checks
that a bounded, validated spot can be transformed into a safe solver call.

## Job Schema

Current schema version:

```text
solver_job_v1
```

Example:

```python
{
    "solver_job_id": "solver_job_fixture_flop_simple",
    "source_snapshot_id": "snapshot_fixture_flop_simple",
    "created_at": "2026-05-25T00:00:00+00:00",
    "schema_version": "solver_job_v1",
    "source_type": "manual_fixture",
    "units": "chips",
    "street": "FLOP",
    "hero_hand": ["Ah", "Kh"],
    "villain_hand": ["Qd", "Qc"],
    "villain_range": None,
    "board": ["2h", "7h", "9d"],
    "pot": 100.0,
    "to_call": 20.0,
    "stack": 1000.0,
    "bet_sizes": [0.33],
    "iterations": 25,
    "timeout_s": 5.0,
    "backend": "rust",
    "label_intent": "solver_smoke",
}
```

`units` is required. If a job says `chips`, then `pot`, `to_call`, and `stack`
are interpreted as chips. Future jobs may use `bb`, but the two must not be
mixed silently.

## Validation

The builder rejects:

- invalid cards;
- duplicate cards across hero, villain, and board;
- board length that does not match the street;
- `pot <= 0`;
- `stack <= 0`;
- `iterations > 100`;
- missing or excessive `timeout_s`;
- `villain_range`, which is not supported by the tiny smoke runner yet.

Public functions return stable dictionaries. They should not leak raw
exceptions to callers.

## Smoke, Candidate, Label

`solver_smoke` means the job is testing technical plumbing only. A smoke solve
answers: can the heavy solver be called, bounded, and logged?

`solver_candidate` is reserved for future jobs that may be considered for
label generation after stronger convergence rules exist.

A true ML label must be produced by a much stricter process: enough iterations,
clear convergence checks, exploitability thresholds, reproducibility rules, and
batch-level audit logs. That is intentionally not implemented yet.

`iterations=10` or `iterations=25` is not strategically reliable. Those values
are useful for fast validation, not for poker advice and not for ML labels.

## Runner Output

`run_solver_job(job)` calls:

```python
solve_tiny_postflop_spot(...)
```

and returns:

```python
{
    "status": "ok" or "failed",
    "solver_job_id": "...",
    "input": {...},
    "output": {...} or None,
    "error": None or "...",
    "duration_ms": 0.0,
    "quality": {
        "iterations": 25,
        "exploitability_last": None,
        "is_label_candidate": False,
        "exclusion_reason": "iterations_too_low",
    },
}
```

For now, `is_label_candidate` is always `False`. Low iterations, missing
exploitability, timeouts, errors, and incomplete outputs all exclude a result
from label use.

## Test Commands

Run the solver job tests:

```powershell
python -m pytest tests/test_solver_jobs.py
```

Run the full test suite:

```powershell
python -m pytest
```

## Current Limits

- Only concrete hero hand vs concrete villain hand jobs are supported.
- `villain_range` is part of the schema but is rejected for now.
- `to_call` is tracked in the job for traceability, but the current tiny smoke
  adapter path does not yet consume it.
- The timeout is a caller-side guard rail, not a hard process kill for native
  computation.
- No labels are generated, no model is trained, and no production-scale solve
  is launched.
