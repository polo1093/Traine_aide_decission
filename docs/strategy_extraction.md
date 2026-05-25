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

`solve_tiny_postflop_spot(...)` exposes the raw root row as inspection data:

```json
{
  "root_strategy_raw": {
    "infoset_key": "8dJc|TsThTdQc|t|",
    "player": 1,
    "root_player": 1,
    "hero_solver_player": 0,
    "root_matches_hero": false,
    "root_player_role": "villain",
    "decision_actor": "hero",
    "root_must_be_hero": true,
    "action_ids": [1, 3, 13],
    "action_labels": ["CHECK", "BET_66", "ALL_IN"],
    "frequencies": [0.51, 0.47, 0.02],
    "source": "average_strategy",
    "bet_size_fractions": [0.66]
  },
  "root_strategy_error": null
}
```

The source is PokerSolver's `average_strategy`, not a current-iteration
strategy. The row is selected with:

```text
state = game.initial_state()
player = game.current_player(state)
infoset_key = game.infoset_key(state, player)
actions = game.legal_actions(state)
frequencies = result.average_strategy[infoset_key]
```

`solver_jobs.strategy_extractor.extract_root_strategy(...)` returns:

```json
{
  "status": "ok",
  "solver_job_id": "synthetic_solver_job_...",
  "available": true,
  "root_strategy": {},
  "root_player": 0,
  "root_player_role": "hero",
  "action_frequencies": {
    "CHECK": 0.4,
    "BET_66": 0.6
  },
  "action_evs": null,
  "dominant_action": "BET_66",
  "dominant_action_frequency": 0.6,
  "confidence": "medium",
  "error": null
}
```

If no explicit strategy is present, it returns `status: "failed"` and
`error: "strategy_not_available"`.

## Accepted Strategy Shapes

The extractor now accepts only `solver_result.output.root_strategy_raw`, because
that shape carries the root-player alignment metadata. Older shapes such as
`root_strategy`, `action_frequencies`, or a bare `average_strategy` row are not
enough to prove the action belongs to hero.

It rejects a bare `game_value`, a bare `strategy_entry_count`, or an unlabeled
strategy vector because those do not identify root actions safely. It also
rejects a valid root strategy when the root player is not confirmed to be hero.

## Action Mapping

PokerSolver returns probabilities aligned with internal action IDs from
`game.legal_actions(state)`. Bet and raise IDs are slots, so their labels must
be derived from `HUNLConfig.bet_size_fractions`.

For example, if `bet_size_fractions` is `[0.66]`, internal opening-bet slot
`ACTION_BET_33` is exposed as `BET_66`, not `BET_33`. The name reflects the
configured pot fraction, not the historical constant name.

The adapter currently exposes:

- `FOLD`;
- `CHECK`;
- `CALL`;
- `BET_<percent>`;
- `RAISE_<percent>`;
- `ALL_IN`.

## Root Player

`root_strategy_raw` is the strategy for the solver's current player at the
initial solver state. That player is not guaranteed to be the real hand's hero.
The adapter builds PokerSolver configs as `initial_hole_cards=(hero, villain)`,
so `hero_solver_player` is `0` unless explicitly set otherwise.

The adapter reports:

- `root_player`: the current solver player at the root node;
- `hero_solver_player`: the solver player index assigned to the job hero;
- `root_matches_hero`: whether both indexes match;
- `root_player_role`: `hero`, `villain`, or `unknown`;
- `root_must_be_hero`: whether extraction is only valid for hero-root spots.

If `root_player=1` while `hero_solver_player=0`, the root belongs to villain
from the adapter's point of view. In that case extraction fails with
`root_player_not_hero`, even if frequencies are present.

Do not call the dominant action a `hero_action`.

## Validation

- Frequencies must be numeric values between 0 and 1.
- Frequency mass must sum to approximately 1.0.
- `hero_solver_player` must be known.
- `root_player_role` must be `hero`.
- `root_matches_hero` must be `true`.
- Invalid or missing strategy payloads fail cleanly without raw exceptions.
- Dominant-action confidence is:
  - `low` below `0.55`;
  - `medium` from `0.55` through `0.75`;
  - `high` above `0.75`.

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

Minimum future conditions for a `solver_action_candidate` would include:

- `root_matches_hero: true`;
- `root_player_role: "hero"`;
- known `hero_solver_player`;
- enough iterations and convergence evidence;
- repeated-run stability;
- explicit downstream review that still keeps it separate from `training_label`.

Runs at `iterations=25` remain strategically weak. They are useful for plumbing
and schema validation only, not for ML labels.
