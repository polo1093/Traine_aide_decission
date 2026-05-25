# Solver Job File Runner

This runner loads synthetic `solver_job_v1` records from JSONL, validates each
job, optionally calls `run_solver_job`, and writes `solver_run_result` JSONL.

It does not train a model, does not create labels, does not create `models/`,
and does not modify PokerSolver or `aide_decision`.

## Generation Versus Solve

Synthetic generation creates input jobs:

```text
synthetic spot generator -> solver_job_v1 JSONL
```

The file runner is the next step:

```text
solver_job_v1 JSONL -> validate_solver_job -> run_solver_job -> solver_run_result JSONL
```

The two steps are intentionally separate. Generating 100 or 1000 jobs is cheap.
Solving those jobs may be expensive, so the solver step defaults to only five
jobs.

By default, real solves run through `solver_jobs.subprocess_runner`, which
launches one worker process per job and applies a hard timeout. `--direct-solver`
is a debugging escape hatch for the older in-process path.

## Why `max_jobs` Is Limited

The default `max_jobs` is 5. Runs above 50 jobs are refused unless
`--allow-large-run` is passed explicitly. This prevents accidental large solver
batches while the project is still validating plumbing and quality rules.

The existing solver job validator still enforces:

- `iterations <= 100`;
- `timeout_s <= 10`;
- no duplicate cards;
- board length matching the street;
- valid positive pot and stack values.

Invalid jobs are written as failed `solver_run_result` rows and are not sent to
the solver.

## Not A Dataset ML

The output JSONL is a solver run trace. It is not a training dataset.

Records intentionally do not contain:

- `training_label`;
- `gto_label`;
- `label_action`.

`quality.is_label_candidate` is forced to `false`. A future labeling step will
need convergence thresholds, reproducibility checks, and audit rules before any
solver output can become a candidate ML label.

## Output Record Shape

Each line is a `solver_run_result` record:

```json
{
  "record_type": "solver_run_result",
  "solver_job_id": "synthetic_solver_job_drawy_board_spot_seed_42_000000",
  "source_snapshot_id": "synthetic_snapshot_drawy_board_spot_seed_42_000000",
  "source_type": "synthetic",
  "solver_job": {},
  "solver_result": {},
  "quality": {
    "is_label_candidate": false,
    "exclusion_reason": "iterations_too_low"
  },
  "error": null,
  "warnings": [],
  "recorded_at": "2026-05-25T00:00:00+00:00"
}
```

If validation fails, `solver_job` and `solver_result` can be `null`, and `error`
explains the rejection. If the solver fails, `solver_job` is preserved and the
solver error is written.

## Commands

Generate 100 synthetic jobs:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 100 --seed 42 --profile drawy_board_spot --output outputs/synthetic_jobs_100.jsonl
```

Solve only five jobs:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_jobs_100.jsonl --max-jobs 5 --output outputs/solver_runs/synthetic_solver_results_5.jsonl
```

Validate five jobs without solving:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_jobs_100.jsonl --max-jobs 5 --output outputs/solver_runs/synthetic_solver_results_dry_run.jsonl --dry-run
```

Run tests:

```powershell
python -m pytest tests/test_solver_job_file_runner.py
```

## Limits

- The runner is for synthetic `source_type` jobs only.
- Unit tests mock the solver.
- The default run size is deliberately tiny.
- Solver results are not labels.
- No model training pipeline is introduced.
