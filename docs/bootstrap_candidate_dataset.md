# Bootstrap Candidate Dataset

This export is an MVP bridge between solver experiments and a weak ML pipeline
smoke test. It is not a GTO dataset and it is not production data.

The exporter reads guarded solver run results or candidate sensitivity JSONL
records and writes a flat JSONL/CSV table. Accepted rows get:

- `bootstrap_label`
- `raw_action`
- `normalized_action`
- `label_source = "solver_candidate"`
- `label_quality = "bootstrap_solver_untrusted"`

It never writes `gto_label` or `training_label`.

## Inclusion Rules

A row is kept only when all of these are true:

- `root_matches_hero = true`
- `root_player_role = "hero"`
- `solver_status = "ok"`
- dominant action frequency is at least `0.70`
- action is stable across the grouped runs
- iterations are at least `25`
- there is no timeout
- there is no blocking danger flag

Allowed actions are `CHECK`, `FOLD`, `CALL`, `BET_x`, and `RAISE_x`.
`ALL_IN` is always excluded for this MVP.

For model training, labels are normalized:

- `CHECK` -> `CHECK`
- `FOLD` -> `FOLD`
- `CALL` -> `CALL`
- `BET_33`, `BET_50`, `BET_66`, `RAISE_33`, `RAISE_66` -> `RAISE`
- `ALL_IN` -> excluded

## Weak Rule Bootstrap

When `--include-weak-rules` is enabled, the exporter keeps the accepted solver
candidates and adds synthetic weak-rule rows to make the pipeline exercise at
least `CHECK`, `FOLD`, and `RAISE`.

Weak-rule rows are clearly marked:

- `label_source = "weak_rule_bootstrap"`
- `label_quality = "bootstrap_weak_rule_untrusted"`
- `weak_rule_reason` explains the simple rule

These rows are not solver candidates and are not GTO labels.

The export also writes:

- `dataset_report.json`
- `dataset_report.md`

The report includes source distribution, label distribution, exclusion reasons,
context/street distribution, missing fields, weak-rule rate, solver-candidate
rate, and explicit warnings.

## Output Fields

The flat export contains:

- `source_id`
- `street`
- `hero_cards`
- `villain_hand`
- `board_cards`
- `pot`
- `to_call`
- `stack`
- `spr`
- `position_model`
- `decision_context_type`
- `action_frequencies`
- `dominant_action`
- `dominant_action_frequency`
- `iterations`
- `exploitability_last`
- `candidate_confidence`
- `raw_action`
- `normalized_action`
- `bootstrap_label`
- `label_source`
- `label_quality`
- `weak_rule_reason`
- `board_card_count`
- `is_river`
- `is_turn`
- `is_check_or_bet_context`
- `is_facing_bet_context`
- `to_call_ratio`
- `stack_to_pot_ratio`
- `excluded`
- `exclusion_reason`

Excluded rows remain in the export for auditability, with `bootstrap_label`
set to `null`.

## Usage

```powershell
python datasets/export_candidate_dataset.py outputs/candidate_sensitivity/results.jsonl `
  --output-jsonl outputs/bootstrap_candidate_dataset/candidates.jsonl `
  --output-csv outputs/bootstrap_candidate_dataset/candidates.csv
```

Weak-rule export:

```powershell
python datasets/export_candidate_dataset.py outputs/candidate_sensitivity/results.jsonl `
  --output-jsonl outputs/bootstrap_candidate_dataset/candidates.jsonl `
  --output-csv outputs/bootstrap_candidate_dataset/candidates.csv `
  --include-weak-rules `
  --min-usable-rows 100 `
  --class-floor 10
```

## Limits

This dataset is intentionally untrusted. It exists to test the mechanics of
feature extraction, loading, and a first bootstrap model. The labels are not
solver-certified, not GTO, and should not be used in production decisions.
