# Snapshot To Solver Job

This mapper converts one exported ML snapshot into one bounded
`solver_job_v1`. It is a validation bridge only: no model is trained, no batch
dataset is generated, and no solver result is treated as an ML label.

## Input Shape

The mapper expects a snapshot similar to what `aide_decision` can export:

```python
{
    "schema_version": "ml_dataset_v1",
    "snapshot_id": "snapshot_001",
    "metadata": {"street": "FLOP"},
    "features": {
        "hero_cards": ["Ah", "Kh"],
        "villain_hand": ["Qd", "Qc"],
        "board_cards": ["2h", "7h", "9d"],
        "pot": 100.0,
        "to_call": 20.0,
        "to_call_is_estimated": False,
        "decision_context_known": True,
        "stack": 1000.0,
        "active_opponents": 1,
        "hero_position": "BTN",
        "units": "chips",
    },
    "labels": {},
    "confidence": {"overall": 0.95},
    "quality_flags": {"usable_for_training": True},
}
```

The output is always stable:

```python
{
    "status": "ok" or "failed",
    "source_snapshot_id": "...",
    "solver_job": {...} or None,
    "error": None or "...",
    "warnings": [],
}
```

## Current Mapping Rules

Only simple postflop heads-up snapshots are mapped:

- `schema_version` must be `ml_dataset_v1`;
- `quality_flags.usable_for_training` must be `True`, or
  `quality_flags.usable_for_solver` must be `True`;
- street must be `FLOP`, `TURN`, or `RIVER`;
- `active_opponents` must be `1`;
- hero cards, villain hand, and board cards must be present;
- board length must match the street;
- pot must be positive;
- `decision_context_known` must be `True`;
- `to_call` must be present and non-null;
- stack defaults to `1000` if absent and emits a warning;
- units default to `chips` if absent and emit a warning.

The mapper does not invent opponent ranges. If `villain_hand` is missing, the
snapshot is rejected with `villain_hand_missing`. If `villain_range` is present,
the snapshot is rejected with `villain_range_not_supported`.

The mapper also does not invent decision context. Missing `to_call` is rejected
with `to_call_unknown`, and `decision_context_known: False` is rejected with
`decision_context_unknown`.

## Why Heads-Up Only

The current tiny solver adapter validates concrete HUNL postflop spots. Multiway
snapshots need a separate abstraction and game model, so they are rejected
instead of approximated silently.

## Snapshot Live Vs Fixture

A manual fixture may include known villain hole cards because it is constructed
for validation. A live exported snapshot may not know the villain hand. In that
case the mapper fails cleanly instead of fabricating a range or creating a
misleading solver job.

## Smoke Vs Candidate Vs Label

`solver_smoke` jobs prove that the pipeline can transform and call the solver.

`solver_candidate` is reserved for future jobs that might be evaluated under
stronger convergence and audit rules.

A true ML label is not produced here. Low-iteration solver calls such as
`iterations=10` or `iterations=25` are not strategically reliable and must not
be interpreted as GTO labels.

## Tests

Run mapper tests:

```powershell
python -m pytest tests/test_snapshot_mapper.py
```

Run the project tests:

```powershell
python -m pytest
```

## Limits

- No `features/`, `labeling/`, or `models/` pipeline is created.
- No model training is performed.
- No massive dataset is generated.
- PokerSolver and `aide_decision` are not modified.
- `villain_range` remains unsupported in this bridge.
- Confidence handling is conservative: numeric confidence values below the
  configured threshold reject the snapshot.
