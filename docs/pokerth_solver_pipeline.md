# PokerTH Solver Pipeline

The PokerTH solver pipeline validates the full technical path:

```text
PokerTH history text
-> hand_summary
-> ml_dataset_v1 snapshot
-> solver_job_v1
-> bounded solver batch
-> solver_run_result JSONL
```

This is not an ML training pipeline. It does not create final labels, does not
train a model, and does not write `label_action`, `gto_label`, or
`training_label` into solver run JSONL records.

## Entry Point

```python
from pokerth.pipeline import run_pokerth_solver_pipeline

result = run_pokerth_solver_pipeline(
    text=history_text,
    hero_name="polo",
    max_hands=10,
    street="FLOP",
    to_call_by_street={"FLOP": 0.0},
    iterations=25,
    timeout_s=5,
    output_path="outputs/solver_runs/pokerth_solver_run.jsonl",
)
```

The pipeline also accepts `path="history.txt"` instead of raw text.

## Summary

The return shape is stable:

```python
{
    "status": "ok" or "partial" or "failed",
    "hands_total": 2,
    "hands_parsed": 1,
    "hands_rejected": 1,
    "snapshots_built": 1,
    "snapshots_rejected": 0,
    "jobs_mapped": 1,
    "jobs_solved": 1,
    "solver_failed": 0,
    "output_path": "...",
    "results": [...],
}
```

Rejection reasons are preserved in `results`, including `showdown_missing`,
`multiway_context_not_supported`, `to_call_unknown`,
`side_pot_not_supported`, `invalid_board`, `villain_hand_missing`, and
`pot_reconstruction_failed`.

## To Call

The pipeline never invents `to_call = 0.0`.

You must pass an explicit `to_call_by_street` value for the requested street.
If the decision context is unknown, omit it and the snapshot is rejected with
`to_call_unknown`. Accepted snapshots have:

```python
"decision_context_known": True
"to_call_is_estimated": False
```

## Current Limits

- Only a small number of hands should be processed (`max_hands` defaults to 10).
- Unit tests mock the solver; they do not launch heavy solves.
- Side pots are rejected.
- Multiway flop contexts are rejected.
- Villain hands must be visible at showdown.
- Pot is marked as estimated.
- `is_label_candidate` remains `False`.

## Commands

Run pipeline tests:

```powershell
python -m pytest tests/test_pokerth_pipeline.py
```

Run all project tests:

```powershell
python -m pytest
```

Manual example:

```powershell
python experiments/run_pokerth_solver_pipeline.py history.txt --street FLOP --to-call 0 --output outputs/solver_runs/pokerth_solver_run.jsonl
```
