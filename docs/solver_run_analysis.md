# Solver Run Analysis

This step analyzes tiny `solver_run_result` JSONL files produced by controlled
solver smoke runs. It does not train a model, does not create labels, does not
create `models/`, and does not modify PokerSolver or `aide_decision`.

## Generate A Tiny Batch

Use only the currently eligible synthetic profiles:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 5 --seed 42 --profile random_turn_spot --iterations 1 --timeout-s 5 --output outputs/synthetic_turn_5.jsonl
python experiments/generate_synthetic_solver_jobs.py --count 5 --seed 43 --profile random_river_spot --iterations 1 --timeout-s 5 --output outputs/synthetic_river_5.jsonl
```

These commands only write `solver_job_v1` inputs. They do not call the solver.

## Run The Solver

The synthetic runner defaults to the hard-timeout subprocess path. Keep
`--max-jobs 5` so the run stays bounded:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_turn_5.jsonl --max-jobs 5 --output outputs/solver_runs/turn_5_results.jsonl
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_river_5.jsonl --max-jobs 5 --output outputs/solver_runs/river_5_results.jsonl
```

The eligibility filter skips non-safe profiles before solver execution. At this
stage, eligible profiles are `random_turn_spot` and `random_river_spot` with low
iterations and bounded timeout.

## Analyze Results

Run:

```powershell
python experiments/analyze_solver_run_results.py outputs/solver_runs/turn_5_results.jsonl outputs/solver_runs/river_5_results.jsonl
```

The report contains:

```json
{
  "total": 0,
  "solved": 0,
  "skipped": 0,
  "timeouts": 0,
  "errors": 0,
  "avg_duration_ms": null,
  "profiles": {},
  "iterations": {},
  "recommendation": ""
}
```

Invalid JSONL lines are counted as errors instead of crashing the analyzer.

## Why This Is Not A Dataset ML

The output is a solver run trace, not training data. It contains operational
status, duration, input job metadata, solver output, warnings, and quality
exclusion reasons.

It must not contain:

- `training_label`;
- `gto_label`;
- `label_action`.

`quality.is_label_candidate` remains `false`, including successful solves.
Low-iteration runs such as `iterations=1` are plumbing smoke tests only.

## Recommendations

The analyzer applies simple guard rails:

- if more than 30 percent of rows time out, reduce profiles or iterations even
  further;
- if more than 80 percent of rows solve successfully, the profile mix is stable
  enough for smoke runs;
- otherwise, inspect skipped and failed rows before expanding.

These recommendations are about runtime stability only. They are not label
quality claims.

## Future Label Candidates

A later label-candidate workflow would need stricter conditions before any row
could be considered for ML:

- much higher and documented solver iterations;
- convergence or exploitability thresholds;
- reproducibility across seeds and repeated solves;
- audit metadata describing solver version and parameters;
- a separate explicit labeling step that writes candidate labels outside this
  smoke-run JSONL.
