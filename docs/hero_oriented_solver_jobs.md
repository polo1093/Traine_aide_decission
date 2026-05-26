# Hero-oriented solver jobs

This module builds bounded `solver_job_v1` records whose solver root node is
validated to be a hero decision before any solve is run.

It does not train a model, create a dataset, create `training_label`, or mark
`is_label_candidate=true`.

## Added job fields

- `hero_solver_player`: `0` or `1`.
- `villain_solver_player`: the opposite solver seat.
- `decision_actor`: `"hero"`, `"villain"`, or `"unknown"`.
- `root_must_be_hero`: boolean guard.
- `hero_position_model`: `"IP"`, `"OOP"`, or `"unknown"`.
- `decision_context_type`: `"hero_check_or_bet"`, `"hero_facing_bet"`, or
  `"unknown"`.
- `initial_hole_cards`: explicit `(P0_cards, P1_cards)` order.
- `initial_contributions`: explicit `(P0_contribution, P1_contribution)` order.

## Supported cases

### Hero OOP check/bet

- `hero_position_model="OOP"`
- `decision_context_type="hero_check_or_bet"`
- `hero_solver_player=1`
- `initial_contributions` are symmetric
- expected root player: `1`
- legal actions include `CHECK`, at least one `BET_*`, and `ALL_IN`

### Hero IP facing bet

- `hero_position_model="IP"`
- `decision_context_type="hero_facing_bet"`
- `hero_solver_player=0`
- `initial_contributions[0] < initial_contributions[1]`
- expected root player: `0`
- legal actions include `FOLD`, `CALL`, at least one `RAISE_*`, and `ALL_IN`

### Hero OOP facing bet

- `hero_position_model="OOP"`
- `decision_context_type="hero_facing_bet"`
- `hero_solver_player=1`
- `initial_contributions[1] < initial_contributions[0]`
- expected root player: `1`
- legal actions include `FOLD`, `CALL`, at least one `RAISE_*`, and `ALL_IN`

## Validation

Before `run_solver_job` calls the adapter, it rebuilds `HUNLPoker(config)` from
the normalized job:

```python
game = HUNLPoker(config)
state = game.initial_state()
root_player = game.current_player(state)
```

If `root_must_be_hero` is true and `root_player != hero_solver_player`, the job
is refused with `root_player_not_hero`.

## Current limits

- The PokerSolver config has no direct `current_player` or initial action
  history field.
- Hero IP check/bet at root is not currently representable as a root hero
  decision with symmetric postflop contributions.
- Facing raise can only be approximated as an asymmetric facing-bet root; a
  true raise-history root needs action-history support.
