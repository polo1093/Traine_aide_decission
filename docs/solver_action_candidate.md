# Solver action candidate

`solver_jobs.action_candidate.build_solver_action_candidate` turns a
hero-aligned root strategy into a guarded action candidate.

It is not a training label, not a GTO label, and not a dataset row. The output
always contains:

```json
{
  "is_training_label": false,
  "label_quality": "solver_candidate_untrusted"
}
```

The function never emits `training_label` or `gto_label`.

## Input

The function accepts either:

- a strategy extraction payload from `extract_root_strategy`, or
- a solver run/result payload containing `root_strategy_raw`.

Required successful signal:

- `solver_status == "ok"`
- root strategy exists
- `root_player_role == "hero"`
- action frequencies are valid and sum to roughly 1.0
- dominant action frequency is at least `0.60`
- iterations are at least `25`

## Refusal reasons

- `solver_not_ok`
- `root_not_hero`
- `strategy_not_available`
- `dominant_action_too_weak`
- `iterations_too_low`
- `invalid_frequencies`

## Output

Successful candidates use this shape:

```json
{
  "status": "ok",
  "solver_job_id": "synthetic_hero_solver_job_...",
  "candidate_action": "CHECK",
  "candidate_frequency": 0.99,
  "candidate_confidence": "high",
  "is_training_label": false,
  "label_quality": "solver_candidate_untrusted",
  "exclusion_reason": null,
  "warnings": []
}
```

Failed candidates keep the same shape with `candidate_action: null` and a
stable `exclusion_reason`.

## Why this is still not an ML label

The candidate layer is only an inspection artifact. It preserves the separation
between solver output and training data. Before any future label production,
the project still needs explicit acceptance thresholds, convergence/quality
policy, provenance metadata, review gates, and a separate writer that cannot be
confused with solver run traces.
