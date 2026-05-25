# Solver Subprocess Runner

The subprocess runner executes one `solver_job_v1` in a separate Python process
and applies a hard timeout to that process.

It does not train a model, does not create labels, does not create `models/`,
and does not modify PokerSolver or `aide_decision`.

## Why A Subprocess Is Needed

The adapter-level timeout uses Python control flow around a solver call. That is
useful for stable return values, but it cannot always interrupt native Rust work
already running inside the process.

The subprocess runner moves each solve into an isolated child process:

```text
parent runner
-> python -m solver_jobs.solver_worker
-> run_solver_job(job)
-> JSON result on stdout
```

If the timeout expires, the parent kills the child process. The batch can then
continue and write a failed `solver_run_result` row instead of hanging.

## Python Timeout Versus Hard Kill

A Python timeout can return control only when Python can observe the timeout.
Native code can keep running underneath.

`subprocess.run(..., timeout=...)` terminates the child process on timeout and
waits for it to exit. That gives the batch runner a real boundary around each
job.

## Result Shape

`run_solver_job_subprocess(job)` returns:

```json
{
  "status": "failed",
  "solver_job_id": "synthetic_solver_job_drawy_board_spot_seed_42_000000",
  "solver_status": "timeout",
  "solver_result": null,
  "error": "solver_subprocess_timeout:5s",
  "duration_ms": 5001.2,
  "quality": {
    "iterations": null,
    "exploitability_last": null,
    "is_label_candidate": false,
    "exclusion_reason": "timeout"
  }
}
```

Successful solves carry the normal `run_solver_job` result in `solver_result`.
`quality.is_label_candidate` is always forced to `false`.

## Bounded Batch Command

Generate jobs:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 100 --seed 42 --profile drawy_board_spot --output outputs/synthetic_jobs_100.jsonl
```

Run only one true subprocess job:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_jobs_100.jsonl --max-jobs 1 --output outputs/solver_runs/synthetic_solver_results_1.jsonl
```

Run five jobs, still bounded:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_jobs_100.jsonl --max-jobs 5 --output outputs/solver_runs/synthetic_solver_results_5.jsonl
```

Validate without solving:

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_jobs_100.jsonl --max-jobs 5 --output outputs/solver_runs/synthetic_solver_results_dry_run.jsonl --dry-run
```

The file runner uses subprocess mode by default. `--direct-solver` exists only
for debugging the old in-process path.

## Why This Is Still Not A Dataset ML

The JSONL output is a solver trace, not a training dataset. It must not contain
`training_label`, `gto_label`, or `label_action`.

Timeouts, low iterations, solver errors, and missing convergence evidence all
remain excluded from label use. A later label-candidate step needs explicit
quality thresholds and audit rules.

## Limits

- One subprocess is launched per solved job.
- `max_jobs` remains 5 by default.
- Runs above 50 jobs require `--allow-large-run`.
- Job validation still enforces `timeout_s <= 10` and `iterations <= 100`.
- A timeout kills the child process, but any native library cleanup inside that
  child is abandoned with the process.
- Tests mock worker behavior; no heavy real solve is required.
