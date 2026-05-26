# Bootstrap Candidate Dataset Report

This dataset is bootstrap-only. It is not GTO, not production data, and not a reliable poker strategy.

- input_records_count: `30`
- rows_total: `120`
- candidates_exported: `102`
- excluded_count: `18`
- class_count: `3`
- weak_rule_row_rate: `0.882353`
- solver_candidate_row_rate: `0.117647`

## Label Distribution

```json
{
  "CHECK": 34,
  "FOLD": 34,
  "RAISE": 34
}
```

## Label Sources

```json
{
  "solver_candidate": 12,
  "weak_rule_bootstrap": 90
}
```

## Exclusions

```json
{
  "all_in_excluded": 18
}
```

## Warnings

- `all_in_excluded`
- `bootstrap_solver_untrusted`
- `call_class_absent`
- `card_fields_missing_or_partial`
- `dataset_contains_weak_rule_labels`
- `not_for_production`
- `not_gto`
- `small_dataset`
- `synthetic_distribution_bias`

## Missing Fields

```json
{
  "board_cards": 12,
  "exploitability_last": 90,
  "hero_cards": 12,
  "iterations": 90,
  "villain_hand": 102
}
```
