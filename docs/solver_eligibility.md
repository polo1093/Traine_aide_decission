# Solver Eligibility

The eligibility filter decides whether a synthetic `solver_job_v1` is allowed to
reach the solver.

It exists because runtime calibration showed that not every valid synthetic job
is safe to solve in the current setup. A valid job can still be too expensive
for smoke batches.

This layer does not train a model, does not create labels, does not create a
training dataset, and does not modify PokerSolver or `aide_decision`.

## Calibration Basis

Observed short calibration:

- `random_turn_spot`: passed at `iterations=1` and `iterations=5`.
- `random_river_spot`: passed at `iterations=1` and `iterations=5`.
- `random_flop_spot`: timed out.
- `drawy_board_spot`: timed out.
- `paired_board_spot`: timed out.
- `top_pair_spot`: timed out.
- `two_pair_plus_spot`: timed out.
- `made_hand_vs_draw_spot`: timed out.

The current policy is intentionally conservative until more calibration data
exists.

## Current Allowed Jobs

The filter allows only jobs that satisfy all of these rules:

- `validate_solver_job(job)` succeeds;
- `source_type == "synthetic"`;
- `generation_profile` is `random_turn_spot` or `random_river_spot`;
- `street` is `TURN` or `RIVER`;
- `iterations <= 5`;
- `timeout_s <= 5`;
- `bet_sizes` has no more than two values.

## Current Refusals

The filter refuses:

- FLOP jobs, with `flop_solver_timeout_risk`;
- non-safe profiles, with `profile_not_solver_safe`;
- non-safe streets, with `street_not_solver_safe`;
- `iterations > 5`, with `iterations_too_high_for_calibration`;
- `timeout_s > 5`, with `timeout_too_high_for_calibration`;
- invalid jobs, with `invalid_solver_job`.

When the file runner sees a non-eligible job, it does not call the solver. It
still writes a `solver_run_result` row with:

- `solver_status: "skipped"`;
- `solver_result: null`;
- `quality.is_label_candidate: false`;
- `quality.exclusion_reason` set to the eligibility reason.

## Refused Does Not Mean Strategically Bad

A refused job is not being judged as a bad poker spot. It only means the job is
not safe for the current bounded solver path. For example, `top_pair_spot` may
be strategically useful later, but calibration says it is currently too likely
to timeout.

## Still Not ML Labels

Eligibility is only a runtime safety gate. Even eligible jobs produce solver
run traces, not ML labels.

No output from this path should contain:

- `training_label`;
- `gto_label`;
- `label_action`.

`quality.is_label_candidate` remains false. A future label-candidate workflow
must add convergence checks, exploitability thresholds, reproducibility rules,
and audit logs.

## Test Command

```powershell
python -m pytest tests/test_solver_eligibility.py
```
