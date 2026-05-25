# Solver Batch Pipeline

The batch runner validates the full technical chain on a small list of
snapshots:

```text
ML snapshot
-> map_snapshot_to_solver_job(...)
-> run_solver_job(...)
-> solver run result
-> JSONL trace
```

This is not a training pipeline. It does not create ML labels, train a model,
or generate a production dataset.

## Batch Result

`run_solver_batch(snapshots)` returns:

```python
{
    "status": "ok" or "partial" or "failed",
    "total": 5,
    "mapped": 2,
    "solved": 2,
    "mapping_failed": 3,
    "solver_failed": 0,
    "failed_total": 3,
    "results": [...],
}
```

Status rules:

- `ok`: every snapshot mapped and solved;
- `partial`: at least one solve succeeded and at least one item failed;
- `failed`: no snapshot solved.

Each row includes:

```python
{
    "source_snapshot_id": "...",
    "mapping_status": "ok" or "failed",
    "solver_status": "ok" or "failed" or "skipped",
    "solver_job": {...} or None,
    "solver_result": {...} or None,
    "error": None or "...",
    "warnings": [],
    "quality": {
        "iterations": 25,
        "exploitability_last": None,
        "is_label_candidate": False,
        "exclusion_reason": "...",
    },
}
```

For this stage, `is_label_candidate` is always `False`.

Snapshots must carry a real decision context before they can reach the solver:
`decision_context_known` must be `True`, and `to_call` must be present. The
batch runner does not default unknown `to_call` values to zero.

## JSONL Persistence

`write_solver_batch_jsonl(...)` writes autonomous records:

```json
{
  "record_type": "solver_run_result",
  "recorded_at": "2026-05-25T00:00:00+00:00",
  "source_snapshot_id": "snapshot_001",
  "mapping_status": "ok",
  "solver_status": "ok",
  "solver_job": {},
  "solver_result": {},
  "quality": {
    "is_label_candidate": false
  },
  "error": null,
  "warnings": [],
  "solver": {
    "solver_name": "PokerSolver",
    "version": "1.7.0",
    "rust_backend_available": true
  }
}
```

The JSONL intentionally does not contain `label_action`, `gto_label`, or
`training_label`. It is a solver run trace, not a training dataset.

The default output path is:

```text
outputs/solver_runs/solver_run_<timestamp>.jsonl
```

Generated `outputs/` files are runtime artifacts. They should not be committed
if you start running local batches repeatedly; a future `.gitignore` entry can
exclude `outputs/` when we begin using this script often.

## Why Villain Hand Is Required

The current mapper only supports concrete heads-up postflop spots. A live
snapshot may not know villain hole cards. When villain cards are absent, the
mapper fails cleanly instead of inventing a range or creating a misleading
solver job.

## Solver Result Vs ML Label

A solver result is the raw output from a bounded technical run. A label ML would
need stronger convergence settings, reproducibility checks, quality thresholds,
and audit rules. That is deliberately not implemented here.

Low-iteration runs such as `iterations=10` or `iterations=25` are not
strategically reliable.

## Commands

Run batch tests:

```powershell
python -m pytest tests/test_solver_batch_runner.py
```

Run all project tests:

```powershell
python -m pytest
```

Run a tiny manual batch:

```powershell
python experiments/run_solver_batch.py
```

## Limits

- Unit tests mock the solver; they do not launch heavy solves.
- The manual experiment uses only a few fixtures.
- Heads-up postflop only.
- No model, no pandas, no scikit-learn, no massive dataset.
- PokerSolver and `aide_decision` are not modified.
