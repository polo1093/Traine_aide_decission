# Hero-oriented synthetic generation

The historical synthetic generator produced valid `solver_job_v1` records, but
it did not guarantee that the solver root actor was hero. In symmetric
postflop spots PokerSolver starts with P1, so a job with hero in P0 could expose
a root strategy for villain.

`synthetic.hero_oriented_spot_generator.generate_hero_oriented_solver_jobs`
builds jobs through `build_hero_oriented_solver_job`, then validates both:

- `validate_solver_job(job)`
- `validate_hero_root_alignment(job)`

The root check rebuilds `HUNLPoker(config)`, calls `initial_state()`, and
requires `game.current_player(state) == hero_solver_player`.

## Supported contexts

- `hero_oop_check_or_bet`
  - `hero_solver_player=1`
  - symmetric contributions
  - root actions include `CHECK`, a bet, and `ALL_IN`
- `hero_ip_facing_bet`
  - `hero_solver_player=0`
  - `initial_contributions[0] < initial_contributions[1]`
  - root actions include `FOLD`, `CALL`, a raise, and `ALL_IN`
- `hero_oop_facing_bet`
  - `hero_solver_player=1`
  - `initial_contributions[1] < initial_contributions[0]`
  - root actions include `FOLD`, `CALL`, a raise, and `ALL_IN`

Only `RIVER` and `TURN` are generated for now. `FLOP` is intentionally excluded
because lossless flop solves are too slow for the current bounded smoke path.

## Eligibility

Hero-oriented synthetic jobs are solver-eligible only when:

- `source_type == "synthetic"`
- `generation_profile` is one of the three supported hero contexts
- `root_must_be_hero is True`
- `decision_actor == "hero"`
- root alignment validates
- `street in {"TURN", "RIVER"}`
- `iterations <= 25`
- `timeout_s <= 5`

## Not a label

This flow still produces solver run traces only. It does not train a model,
create an ML dataset, write `training_label`, set `is_label_candidate=true`, or
turn a dominant action into a label.

Before creating a future `solver_action_candidate`, we still need explicit
quality thresholds, enough solver iterations/convergence evidence, audit
metadata, and a separate downstream review step that keeps candidate actions
separate from ML labels.
