# Strategy Extraction

This layer inspects `solver_run_result` records and tries to read an explicit
root strategy. It does not solve by itself, train a model, create a dataset,
write `training_label`, or mark `is_label_candidate` true.

## Purpose

The extractor answers one narrow question:

```text
Does this solver result expose root action frequencies that can be inspected?
```

It is not a label-generation step. A successful extraction is only a readable
strategy summary for investigation.

## Output Shape

`solver_jobs.strategy_extractor.extract_root_strategy(...)` returns:

```json
{
  "status": "ok",
  "solver_job_id": "synthetic_solver_job_...",
  "available": true,
  "root_strategy": {},
  "action_frequencies": {
    "CHECK": 0.4,
    "BET_33": 0.6
  },
  "action_evs": null,
  "dominant_action": "BET_33",
  "dominant_action_frequency": 0.6,
  "confidence": "medium",
  "error": null
}
```

If no explicit strategy is present, it returns `status: "failed"` and
`error: "strategy_not_available"`.

## Accepted Strategy Shapes

The extractor accepts explicit action-frequency payloads such as:

- `solver_result.output.root_strategy`;
- `solver_result.output.action_frequencies`;
- `solver_result.output.strategy.root`;
- `solver_result.output.average_strategy.root`, when the value is already a
  mapping of action name to frequency.

It also accepts action rows:

```json
{
  "root_strategy": [
    {"action": "CHECK", "frequency": 0.4, "ev": 1.0},
    {"action": "BET_33", "frequency": 0.6, "ev": 1.2}
  ]
}
```

It rejects a bare `game_value`, a bare `strategy_entry_count`, or an unlabeled
strategy vector because those do not identify root actions safely.

## Validation

- Frequencies must be numeric values between 0 and 1.
- Frequency mass must sum to approximately 1.0.
- Invalid or missing strategy payloads fail cleanly without raw exceptions.
- Dominant-action confidence is:
  - `low` below `0.55`;
  - `medium` from `0.55` to below `0.75`;
  - `high` at `0.75` or above.

## Real Inspection

Run a single stable river job:

```powershell
python experiments/inspect_solver_strategy.py --profile random_river_spot --iterations 25 --timeout-s 5
```

Run a single stable turn job:

```powershell
python experiments/inspect_solver_strategy.py --profile random_turn_spot --iterations 25 --timeout-s 5
```

The experiment prints an abridged `solver_result`, the output keys exposed by
the current adapter, and the extractor result.

## Not A Label

Even if a dominant action is found, this repository still keeps:

- `is_label_candidate: false`;
- no `training_label`;
- no `gto_label`;
- no `label_action`.

A future label-candidate flow would need explicit convergence thresholds,
reproducibility checks, audit metadata, and a separate approval step.
