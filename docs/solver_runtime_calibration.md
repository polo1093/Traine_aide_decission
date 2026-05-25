# Solver Runtime Calibration

Runtime calibration measures which tiny synthetic `solver_job_v1` spots can run
through the subprocess runner within a bounded timeout.

It does not train a model, does not create labels, does not create `models/`,
and does not modify PokerSolver or `aide_decision`.

## Why This Exists

Some synthetic spots may be much heavier than others. Before running even a
small solver batch, we need a measured answer to questions like:

- Which profiles finish at `iterations=1`?
- Which profiles time out even at tiny settings?
- Are turn or river spots cheaper or more expensive in this local setup?
- Which profiles are safe enough for smoke tests?

The calibration tool generates a very small number of synthetic jobs per
profile and solves them with `run_solver_job_subprocess`, so every solve has a
hard process timeout.

## Not A Labeling Step

A successful calibration row means only: the solver returned before the timeout.
It is not a strategic quality guarantee.

Calibration output is not a training dataset and does not contain:

- `training_label`;
- `gto_label`;
- `label_action`.

Every row keeps `is_label_candidate` false. Future label generation still needs
convergence checks, exploitability thresholds, reproducibility rules, and audit
logs.

## Commands

Minimal calibration:

```powershell
python experiments/calibrate_solver_runtime.py --profiles random_river_spot --jobs-per-profile 1 --iterations 1 --timeout-s 5
```

Short calibration:

```powershell
python experiments/calibrate_solver_runtime.py --jobs-per-profile 1 --iterations 1 5 --timeout-s 5 --output outputs/solver_calibration/calibration_short.jsonl
```

The short calibration plans 16 solves: 8 profiles times 2 iteration settings.
The default maximum plan size is 20 jobs.

Progressive calibration for the currently stable turn and river profiles:

```powershell
python experiments/calibrate_solver_runtime.py --profiles random_turn_spot random_river_spot --jobs-per-profile 3 --iterations 1 5 10 25 --timeout-s 5 --seed 42 --max-total-jobs 24 --output outputs/solver_calibration/progressive_turn_river_24.jsonl --summary-output outputs/solver_calibration/progressive_turn_river_24_summary.json
```

This plans exactly 24 solves: 2 profiles times 4 iteration levels times 3 jobs.

## Garde-Fous

- `jobs_per_profile` defaults to 1.
- `max_total_jobs` defaults to 20.
- Plans above 50 jobs require `--allow-large-run`.
- `iterations` must be `<= 25`.
- `timeout_s` must be `<= 10`.
- Every solve uses the subprocess runner.

## Output

Each JSONL row is a runtime measurement:

```json
{
  "record_type": "solver_runtime_calibration",
  "profile": "drawy_board_spot",
  "street": "FLOP",
  "iterations": 5,
  "job_index": 0,
  "solver_job_id": "synthetic_solver_job_drawy_board_spot_seed_42_000000_iter_5",
  "status": "timeout",
  "solver_status": "timeout",
  "duration_ms": 5000.0,
  "error": "solver_subprocess_timeout:5s",
  "quality": {
    "is_label_candidate": false,
    "exclusion_reason": "timeout"
  },
  "is_label_candidate": false
}
```

The CLI also prints a JSON summary with:

- total jobs run;
- successes;
- timeouts;
- errors;
- average duration for successful solves;
- per-profile counts;
- per-iteration counts;
- per-profile/per-iteration counts;
- success and timeout rates;
- profiles too heavy;
- profiles usable for smoke tests;
- recommended conservative parameters.

## Interpreting Results

Treat a profile as too heavy when it times out or has no successful rows.

Treat a profile as smoke-test usable only when all rows for the tested settings
finish successfully. Start with the lowest successful `iterations` value.

For progressive calibration, recommendation rules are:

- success rate below 80 percent: not recommended;
- timeout rate above 10 percent: not recommended;
- average duration above 3000 ms: avoid for large batches;
- success rate above 90 percent with average duration at or below 3000 ms:
  `stable_for_solver_batch`.

If no profile succeeds, keep calibration at one job, `iterations=1`, and
consider lowering synthetic complexity or investigating solver initialization
cost before any larger run.

## Limits

- The tool measures local runtime behavior, not poker quality.
- One subprocess is launched per calibration row.
- Results may vary by machine, Rust backend availability, and current system
  load.
- A profile passing at `iterations=1` or `5` is still not suitable for ML labels.
