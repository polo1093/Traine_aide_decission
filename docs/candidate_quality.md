# Candidate quality

`solver_jobs.candidate_quality.evaluate_candidate_quality` checks whether a
non-label solver action candidate is stable across multiple runs.

It does not create `training_label`, does not create `gto_label`, and always
returns:

```json
{
  "is_training_label": false,
  "label_quality": "solver_candidate_untrusted"
}
```

## Candidate vs ML label

A solver action candidate is an inspectable solver artifact. It can say "the
solver repeatedly preferred CHECK in these runs." It is not an ML target and is
not safe to train on until a separate policy approves quality, provenance,
coverage, and conversion rules.

## Why one solve is not enough

Single solves can reflect random seed noise, low iterations, unstable CFR
averages, action abstraction artifacts, or subgame construction quirks. The
quality layer therefore requires multiple observations for the same candidate
analysis and refuses `single_run_only`.

## Stability checks

The provisional checks refuse candidates when:

- root is not hero;
- any run is not solver `ok`;
- candidate is absent;
- dominant action changes between runs;
- minimum dominant frequency is below `0.60`;
- iterations are below `25`;
- exploitability is missing or not numeric;
- any run timed out or returned an error.

The output includes:

- `stable_action`
- `dominant_frequency_avg`
- `dominant_frequency_min`
- `dominant_action_consistency`
- `danger_flags`
- `recommendation`

## ALL_IN handling

`ALL_IN` can be correct in a no-limit subgame, but in these synthetic spots it
can also be an artifact of stack size, pot construction, contributions, or the
action abstraction. For now, `ALL_IN` is always suspicious unless the caller
explicitly marks the scenario as short-stack.

When the stable dominant action is `ALL_IN`, the quality layer adds:

```json
"danger_flags": ["extreme_action_all_in"]
```

This does not make it a label. It remains:

```json
"label_quality": "solver_candidate_untrusted"
```

## Future label candidate requirements

Before any future label-candidate writer exists, the project needs:

- calibrated iteration and exploitability thresholds;
- stability across seeds and action abstractions;
- explicit stack-depth and pot geometry checks;
- handling for extreme actions such as `ALL_IN`;
- provenance and audit metadata;
- a separate review gate that cannot be confused with solver run traces.
