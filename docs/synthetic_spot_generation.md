# Synthetic Spot Generation

This layer generates bounded synthetic postflop spots as `solver_job_v1`
records. It does not train a model, does not create ML labels, and does not
call the solver.

## Why Generate Spots Instead Of Full Games

Recording complete PokerTH games by hand is slow and produces a narrow sample
of situations. For offline solver experiments, the immediate need is smaller:
cover specific postflop cases that exercise the solver job contract.

Synthetic spots let us ask for cases such as:

- drawy flops;
- paired boards;
- top-pair spots;
- simple made-hand-versus-draw spots;
- simple two-pair-or-better constructions.

The generator creates a spot that the existing solver runner can consume later.
It does not try to simulate a full betting history or produce a realistic game
tree from PokerTH.

## Synthetic Spot Versus Real PokerTH Hand

A real PokerTH hand has player actions, positions, stacks across streets, and
history-dependent decisions. A synthetic spot is only a compact postflop state:
hero hand, villain hand, board, pot, call amount, stack, bet sizes, and bounded
solver settings.

These spots do not necessarily represent the real PokerTH distribution. They
are coverage fixtures first. They are useful for testing plumbing and creating
controlled offline solver inputs, not for claiming that the generated set is a
natural sample of real play.

Profiles such as `made_hand_vs_draw_spot`, `top_pair_spot`, and
`two_pair_plus_spot` are built from simple card constraints. They are not backed
by a complete poker hand evaluator.

## Why Solver Results Are Not ML Labels Yet

A generated job is only an input. A solver result from a tiny bounded run is
only a technical result. It is not automatically a reliable ML label.

Reliable labels will need stricter rules later:

- enough iterations;
- convergence checks;
- exploitability thresholds;
- reproducible batch metadata;
- audit logs and exclusion criteria.

For now, generated jobs keep `label_intent` at `solver_smoke`. The volume of
jobs does not imply a volume of trustworthy labels.

## Generate Jobs

The experiment script writes JSONL jobs only. It never calls `run_solver_job`.

Generate 10 jobs:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 10 --seed 42 --profile random_flop_spot --output out/synthetic_jobs_10.jsonl
```

Generate 100 jobs:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 100 --seed 42 --profile drawy_board_spot --output out/synthetic_jobs_100.jsonl
```

Generate 1000 jobs:

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 1000 --seed 42 --profile random_river_spot --output out/synthetic_jobs_1000.jsonl
```

The default maximum is 1000 jobs per command. This keeps accidental generation
bounded. Solver settings are also bounded by the existing `solver_job_v1`
validator: `iterations <= 100`, `timeout_s <= 10`, and no more than five bet
sizes.

## Available Profiles

- `random_flop_spot`
- `random_turn_spot`
- `random_river_spot`
- `drawy_board_spot`
- `paired_board_spot`
- `made_hand_vs_draw_spot`
- `top_pair_spot`
- `two_pair_plus_spot`

Every generated job is deterministic for the same `seed`, `profile`, and
`index`. Job IDs are derived from those values.

## Solving A Small Sample Later

Solving is a separate step. Start with a tiny smoke sample, for example 3 to 5
jobs, and write solver outputs as result JSONL through the existing runner.

Do not treat a 1000-job file as permission to launch 1000 heavy solves. The
generation step is cheap and safe; the solver step is intentionally separate and
must stay bounded until convergence and labeling rules exist.
