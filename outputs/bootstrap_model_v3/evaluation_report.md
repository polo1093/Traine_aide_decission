# Bootstrap Model Evaluation

- selected_model: `logistic_regression`
- selection_metric: `macro_f1`
- training_quality: `pipeline_smoke_only`
- not_for_production: `True`
- contains_weak_rule_labels: `True`
- model_may_learn_synthetic_rules: `True`
- accuracy: `1.0`
- macro_f1: `1.0`
- weighted_f1: `1.0`

## Label Distribution

```json
{
  "CHECK": 34,
  "FOLD": 34,
  "RAISE": 34
}
```

## Model Comparison

```json
{
  "dummy": {
    "accuracy": 0.307692,
    "classification_report": {
      "CHECK": {
        "f1-score": 0.47058823529411764,
        "precision": 0.3076923076923077,
        "recall": 1.0,
        "support": 8.0
      },
      "FOLD": {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 9.0
      },
      "RAISE": {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 9.0
      },
      "accuracy": 0.3076923076923077,
      "macro avg": {
        "f1-score": 0.1568627450980392,
        "precision": 0.10256410256410257,
        "recall": 0.3333333333333333,
        "support": 26.0
      },
      "weighted avg": {
        "f1-score": 0.14479638009049772,
        "precision": 0.09467455621301776,
        "recall": 0.3076923076923077,
        "support": 26.0
      }
    },
    "confusion_matrix": {
      "CHECK": {
        "CHECK": 8,
        "FOLD": 0,
        "RAISE": 0
      },
      "FOLD": {
        "CHECK": 9,
        "FOLD": 0,
        "RAISE": 0
      },
      "RAISE": {
        "CHECK": 9,
        "FOLD": 0,
        "RAISE": 0
      }
    },
    "macro_f1": 0.156863,
    "model_name": "dummy",
    "weighted_f1": 0.144796
  },
  "extra_trees": {
    "accuracy": 1.0,
    "classification_report": {
      "CHECK": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 8.0
      },
      "FOLD": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "RAISE": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "accuracy": 1.0,
      "macro avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      },
      "weighted avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      }
    },
    "confusion_matrix": {
      "CHECK": {
        "CHECK": 8,
        "FOLD": 0,
        "RAISE": 0
      },
      "FOLD": {
        "CHECK": 0,
        "FOLD": 9,
        "RAISE": 0
      },
      "RAISE": {
        "CHECK": 0,
        "FOLD": 0,
        "RAISE": 9
      }
    },
    "macro_f1": 1.0,
    "model_name": "extra_trees",
    "weighted_f1": 1.0
  },
  "logistic_regression": {
    "accuracy": 1.0,
    "classification_report": {
      "CHECK": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 8.0
      },
      "FOLD": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "RAISE": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "accuracy": 1.0,
      "macro avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      },
      "weighted avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      }
    },
    "confusion_matrix": {
      "CHECK": {
        "CHECK": 8,
        "FOLD": 0,
        "RAISE": 0
      },
      "FOLD": {
        "CHECK": 0,
        "FOLD": 9,
        "RAISE": 0
      },
      "RAISE": {
        "CHECK": 0,
        "FOLD": 0,
        "RAISE": 9
      }
    },
    "macro_f1": 1.0,
    "model_name": "logistic_regression",
    "weighted_f1": 1.0
  },
  "random_forest": {
    "accuracy": 1.0,
    "classification_report": {
      "CHECK": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 8.0
      },
      "FOLD": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "RAISE": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 9.0
      },
      "accuracy": 1.0,
      "macro avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      },
      "weighted avg": {
        "f1-score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 26.0
      }
    },
    "confusion_matrix": {
      "CHECK": {
        "CHECK": 8,
        "FOLD": 0,
        "RAISE": 0
      },
      "FOLD": {
        "CHECK": 0,
        "FOLD": 9,
        "RAISE": 0
      },
      "RAISE": {
        "CHECK": 0,
        "FOLD": 0,
        "RAISE": 9
      }
    },
    "macro_f1": 1.0,
    "model_name": "random_forest",
    "weighted_f1": 1.0
  }
}
```

## Warnings

- `all_in_absent_because_excluded`
- `board_cards_missing_or_null`
- `call_class_absent`
- `contains_weak_rule_labels`
- `dataset_used_for_pipeline_smoke_test_only`
- `hero_cards_missing_or_null`
- `label_quality_bootstrap_solver_untrusted`
- `label_quality_bootstrap_weak_rule_untrusted`
- `metrics_may_be_artificial_due_to_weak_rules`
- `model_may_learn_synthetic_rules`
- `not_for_production`
- `not_gto`
- `small_dataset`
