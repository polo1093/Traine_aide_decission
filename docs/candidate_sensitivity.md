# Candidate sensitivity analysis

This experiment studies how hero-oriented solver action candidates react to
changes in stack, pot, `to_call`, and action abstraction.

It exists because some facing-bet toy subgames produce stable `ALL_IN`
candidates. That can be a real equilibrium preference in the constructed
subgame, but it can also be an artifact of:

- stack-to-pot ratio;
- pot and contribution geometry;
- `to_call` size;
- available bet/raise sizes;
- fixed-combo synthetic card selection;
- short bounded solves.

## What It Runs

`experiments/analyze_candidate_sensitivity.py` runs a bounded curated plan:

- contexts:
  - `hero_oop_check_or_bet`
  - `hero_ip_facing_bet`
  - `hero_oop_facing_bet`
- street: `RIVER`
- backend: `rust`
- iterations: `25`, `50`, `100`
- max default solves: `30`

The default plan uses 27 base solves plus 3 extra probes, covering:

- stack values including `200`, `500`, `1000`, `2000`;
- pot values `100`, `300`, `800`;
- facing-bet `to_call` ratios near `10%`, `33%`, `75%`;
- bet sizes `[0.33]`, `[0.5]`, `[0.66]`, `[1.0]`.

It refuses larger plans above the configured cap unless explicitly allowed, and
still refuses more than 50 solves without an override.

## Output

The JSONL rows are candidate-sensitivity records, not training rows. Each row
contains fields such as:

```json
{
  "context": "hero_ip_facing_bet",
  "street": "RIVER",
  "stack": 1000,
  "pot": 300,
  "to_call": 99,
  "spr": 3.333333,
  "bet_size_fractions": [0.66],
  "iterations": 100,
  "dominant_action": "ALL_IN",
  "dominant_frequency": 0.71,
  "danger_flags": ["extreme_action_all_in"],
  "quality_status": "ok",
  "is_training_label": false,
  "label_quality": "solver_candidate_untrusted"
}
```

The summary groups runs by context and scenario, then reports action stability
across 25/50/100 iterations.

## Why ALL_IN Is Suspect

`ALL_IN` is legal in no-limit hold'em, and PokerSolver may prefer it in a
specific subgame. In these toy synthetic spots, though, all-in can be inflated
by coarse action abstractions or unusual stack/pot/contribution geometry. For
that reason, any `ALL_IN` dominant action is marked with:

```json
"danger_flags": ["extreme_action_all_in"]
```

It remains usable for candidate analysis only, not training.

## Interpretation

Useful questions:

- Does `ALL_IN` appear only at low SPR?
- Does it appear when `to_call` is small?
- Does it depend on the single available raise fraction?
- Do `CHECK`, `FOLD`, or `CALL` dominate under other stack/pot settings?
- Is the action stable from 25 to 100 iterations?

Even stable results are not labels. A future label-candidate policy would need
calibrated exploitability thresholds, multi-seed stability, abstraction
sensitivity checks, and explicit handling for extreme actions.
