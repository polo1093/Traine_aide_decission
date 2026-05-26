# Bootstrap Action Model Card

This model is trained from bootstrap candidate rows produced by solver candidates plus optional weak-rule rows.
It is not GTO, not production-ready, and must not be connected to live decisions.

## Data Origin

- input_path: `outputs/bootstrap_candidate_dataset_v3/candidates.csv`
- rows_used: `102`
- contains_weak_rule_labels: `True`
- model_may_learn_synthetic_rules: `True`

## Classes

```json
[
  "CHECK",
  "FOLD",
  "RAISE"
]
```

## Features

```json
{
  "card_features": [
    "hero_cards",
    "villain_hand",
    "board_cards"
  ],
  "card_presence_counts": {
    "board_cards": 90,
    "hero_cards": 90,
    "villain_hand": 0
  },
  "categorical_features": [
    "street",
    "position_model",
    "decision_context_type",
    "label_source",
    "label_quality",
    "candidate_confidence"
  ],
  "categorical_values": {
    "candidate_confidence": [
      "high",
      "weak_rule"
    ],
    "decision_context_type": [
      "hero_check_or_bet",
      "hero_facing_bet"
    ],
    "label_quality": [
      "bootstrap_solver_untrusted",
      "bootstrap_weak_rule_untrusted"
    ],
    "label_source": [
      "solver_candidate",
      "weak_rule_bootstrap"
    ],
    "position_model": [
      "IP",
      "OOP"
    ],
    "street": [
      "RIVER"
    ]
  },
  "forbidden_targets": [
    "ALL_IN",
    "gto_label",
    "training_label"
  ],
  "labels": [
    "CHECK",
    "FOLD",
    "RAISE"
  ],
  "not_for_production": true,
  "numeric_features": [
    "pot",
    "to_call",
    "stack",
    "spr",
    "dominant_action_frequency",
    "iterations",
    "exploitability_last",
    "board_card_count",
    "is_river",
    "is_turn",
    "is_check_or_bet_context",
    "is_facing_bet_context",
    "to_call_ratio",
    "stack_to_pot_ratio"
  ],
  "target": "bootstrap_label",
  "training_quality": "pipeline_smoke_only"
}
```

## Limits

- Labels are weak bootstrap labels, not verified strategy labels.
- Weak-rule rows can create synthetic distribution bias.
- Metrics can be artificially high when the model learns generated rules.
- CALL may be absent in the current dataset.
- Card features are partial and should not be treated as a complete hand representation.

## Next Steps

- Increase solver-aligned real candidates.
- Add CALL coverage.
- Separate weak-rule validation from solver candidate validation.
- Keep all evaluation offline until data quality is materially better.
