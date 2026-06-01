# Bootstrap Training

`models/train_bootstrap_model.py` trains and compares deliberately weak
bootstrap sklearn models from a candidate CSV.

This is only a pipeline smoke test. The source labels are
`bootstrap_solver_untrusted`, not GTO labels, and the resulting model must not
be used in production or connected to `aide_decision`.

## Command

```powershell
python models/train_bootstrap_model.py `
  --input outputs/bootstrap_candidate_dataset/candidates.csv `
  --output-dir outputs/bootstrap_model `
  --model-type random_forest `
  --min-rows 10
```

Supported model types are:

- `auto`
- `dummy`
- `random_forest`
- `logistic_regression`
- `extra_trees`

With `--model-type auto`, the script compares:

- `DummyClassifier`
- `LogisticRegression(class_weight="balanced")`
- `RandomForestClassifier(class_weight="balanced")`
- `ExtraTreesClassifier(class_weight="balanced")`

Selection is based on macro F1. The report also includes accuracy, weighted F1,
confusion matrix, classification report, train/test label distribution, and a
comparison against the dummy baseline.

## Features

Numeric features:

- `pot`
- `to_call`
- `stack`
- `spr`
- `dominant_action_frequency`
- `iterations`
- `exploitability_last`
- `board_card_count`
- `is_river`
- `is_turn`
- `is_check_or_bet_context`
- `is_facing_bet_context`
- `to_call_ratio`
- `stack_to_pot_ratio`

Categorical features:

- `street`
- `position_model`
- `decision_context_type`
- `label_source`
- `label_quality`
- `candidate_confidence`

Target:

- `bootstrap_label`

## Strict Refusals

Training fails if:

- `bootstrap_label` is missing
- `gto_label` is present
- `training_label` is present
- `ALL_IN` appears as a label
- the usable rows contain only one class
- usable rows are below `--min-rows`

## Outputs

The script writes:

- `model.joblib`
- `preprocessing.joblib`
- `feature_schema.json`
- `label_mapping.json`
- `evaluation_report.json`
- `evaluation_report.md`
- `model_card.md`

Every report includes `training_quality = "pipeline_smoke_only"` and warnings
that the dataset is small, untrusted, and unsuitable for production.

For datasets that include weak-rule rows, reports also include:

- `contains_weak_rule_labels = true`
- `not_for_production = true`
- warning `contains_weak_rule_labels`

The model may learn the weak rules directly. That is acceptable for the smoke
test, but it is not evidence of strategic quality. If metrics look very strong
on the current dataset, interpret that as evidence that the pipeline can
train and evaluate, not that the model plays well.

## Offline Prediction

Use:

```powershell
python models/predict_bootstrap_model.py `
  --model-dir outputs/bootstrap_model `
  --input-json "{\"pot\":100,\"to_call\":0,\"stack\":1000,\"spr\":10,\"dominant_action_frequency\":0.8,\"iterations\":25,\"street\":\"RIVER\",\"position_model\":\"OOP\",\"decision_context_type\":\"hero_check_or_bet\",\"label_source\":\"weak_rule_bootstrap\",\"label_quality\":\"bootstrap_weak_rule_untrusted\",\"candidate_confidence\":\"high\"}"
```

The prediction script is offline-only and always emits
`pipeline_smoke_only_not_for_production`.
