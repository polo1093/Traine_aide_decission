"""Train and compare sklearn bootstrap models from candidate datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


NUMERIC_FEATURES = (
    "features.pot",
    "features.to_call",
    "features.to_call_pot_ratio",
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_known",
    "features.equity_required",
    "features.equity_gap",
    "features.ev",
    "features.call_max",
    "features.player_start",
    "features.player_active",
    "features.active_opponents",
    "features.board_card_count",
    "features.hero_cards_known",
    "features.opponent_looseness_avg",
    "features.opponent_aggression_avg",
    "features.opponent_confidence_avg",
    "features.hero_stack",
    "features.effective_stack",
    "features.stack_to_pot_ratio",
    "features.has_check",
    "features.has_call",
    "features.has_raise",
)
CATEGORICAL_FEATURES = (
    "metadata.street",
    "features.hero_position",
)
CARD_FEATURES: tuple[str, ...] = ()
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + CARD_FEATURES
FEATURE_ALIASES = {
    "metadata.street": ("street",),
    "features.hero_position": ("hero_position", "position_model"),
    "features.pot": ("pot",),
    "features.to_call": ("to_call",),
    "features.to_call_pot_ratio": ("to_call_pot_ratio", "to_call_ratio"),
    "features.equity_table": ("equity_table",),
    "features.equity_1v1": ("equity_1v1",),
    "features.equity_known": ("equity_known",),
    "features.ev": ("ev",),
    "features.call_max": ("call_max",),
    "features.board_card_count": ("board_card_count",),
    "features.player_start": ("player_start",),
    "features.player_active": ("player_active",),
    "features.hero_stack": ("stack",),
    "features.effective_stack": ("stack",),
    "features.has_check": ("has_check", "is_check_or_bet_context"),
    "features.has_call": ("has_call",),
    "features.has_raise": ("has_raise",),
}
ALLOWED_MODEL_TYPES = {"auto", "logistic_regression", "random_forest", "extra_trees", "dummy"}
TRAINING_QUALITY = "pipeline_smoke_only"
WEAK_RULE_LABEL_SOURCE = "weak_rule_bootstrap"
WEAK_RULE_LABEL_QUALITY = "bootstrap_weak_rule_untrusted"
SOLVER_LABEL_QUALITY = "bootstrap_solver_untrusted"
STRICT_FORBIDDEN_COLUMNS = {"gto_label", "training_label"}
RANDOM_STATE = 17


class BootstrapTrainingError(ValueError):
    """Raised when a bootstrap dataset violates a strict guard."""


def train_bootstrap_model(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_type: str,
    min_rows: int = 50,
    random_seed: int = RANDOM_STATE,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    if model_type not in ALLOWED_MODEL_TYPES:
        raise BootstrapTrainingError(f"unsupported_model_type:{model_type}")
    selected_features = tuple(feature_columns or FEATURE_COLUMNS)
    unknown_features = [feature for feature in selected_features if feature not in FEATURE_COLUMNS]
    if unknown_features:
        raise BootstrapTrainingError(f"unknown_feature_column:{unknown_features[0]}")

    rows, fieldnames = load_candidate_csv(input_path)
    validate_schema(fieldnames)
    train_rows = select_training_rows(rows)
    validate_training_rows(train_rows, min_rows=min_rows)

    warnings = build_reliability_warnings(rows, train_rows)
    train_split, test_split, split_warning = stratified_split(train_rows, random_seed=random_seed)
    if split_warning:
        warnings.append(split_warning)

    labels = sorted({str(row["bootstrap_label"]) for row in train_rows})
    candidates = candidate_model_names(model_type)
    comparisons = {
        name: train_and_evaluate_model(name, train_split, test_split, labels, feature_columns=selected_features)
        for name in candidates
    }
    best_model_name = select_best_model(comparisons, requested_model_type=model_type)
    best = comparisons[best_model_name]
    dummy = comparisons["dummy"]
    if best["macro_f1"] <= dummy["macro_f1"] + 0.02:
        warnings.append("best_model_does_not_clearly_beat_dummy")

    contains_weak_rule_labels = any(_label_source(row) == WEAK_RULE_LABEL_SOURCE for row in train_rows)
    report = {
        "status": "ok",
        "requested_model_type": model_type,
        "model_type": best_model_name,
        "selected_model": best_model_name,
        "selection_metric": "macro_f1",
        "training_quality": TRAINING_QUALITY,
        "not_for_production": True,
        "contains_weak_rule_labels": contains_weak_rule_labels,
        "model_may_learn_synthetic_rules": contains_weak_rule_labels,
        "input_path": str(input_path),
        "rows_total": len(rows),
        "rows_used": len(train_rows),
        "rows_excluded": len(rows) - len(train_rows),
        "train_size": len(train_split),
        "test_size": len(test_split),
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in train_rows).items())),
        "train_label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in train_split).items())),
        "test_label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in test_split).items())),
        "warnings": sorted(set(warnings)),
        "model_feature_columns": list(selected_features),
        "model_comparison": comparisons,
        "dummy_comparison": dummy,
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
    }

    feature_schema = build_feature_schema(train_rows, labels, feature_columns=selected_features)
    label_mapping = build_label_mapping(labels)
    paths = write_training_outputs(
        output_dir=output_dir,
        model=best["model"],
        feature_schema=feature_schema,
        label_mapping=label_mapping,
        report=report,
    )
    return {**without_model_objects(report), "output_files": paths}


def load_candidate_csv(input_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [normalize_csv_row(row) for row in reader]
    return rows, fieldnames


def normalize_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for feature in NUMERIC_FEATURES:
        normalized[feature] = _float_or_none(_feature_value(normalized, feature))
    normalized["excluded"] = _bool(normalized.get("excluded"))
    normalized["bootstrap_label"] = _text(normalized.get("bootstrap_label"))
    return normalized


def validate_schema(fieldnames: Sequence[str]) -> None:
    columns = set(fieldnames)
    forbidden = sorted(columns & STRICT_FORBIDDEN_COLUMNS)
    if forbidden:
        raise BootstrapTrainingError(f"forbidden_column_present:{forbidden[0]}")
    if "bootstrap_label" not in columns:
        raise BootstrapTrainingError("bootstrap_label_missing")


def select_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("excluded") is True:
            continue
        label = _text(row.get("bootstrap_label"))
        if label:
            selected.append(dict(row))
    return selected


def validate_training_rows(rows: Sequence[Mapping[str, Any]], *, min_rows: int) -> None:
    if len(rows) < min_rows:
        raise BootstrapTrainingError(f"not_enough_rows:{len(rows)}<min_rows:{min_rows}")
    labels = [str(row["bootstrap_label"]) for row in rows]
    if "ALL_IN" in labels:
        raise BootstrapTrainingError("all_in_label_present")
    if len(set(labels)) < 2:
        raise BootstrapTrainingError("single_class_dataset")


def build_reliability_warnings(all_rows: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings = ["dataset_used_for_pipeline_smoke_test_only", "not_for_production", "not_gto"]
    if len(train_rows) < 50:
        warnings.append("dataset_has_less_than_50_training_rows")
    if len(train_rows) < 500:
        warnings.append("small_dataset")
    labels = Counter(str(row["bootstrap_label"]) for row in train_rows)
    if len(labels) < 3:
        warnings.append("dataset_has_less_than_3_classes")
    if "CALL" not in labels:
        warnings.append("call_class_absent")
    if any(_label_source(row) == WEAK_RULE_LABEL_SOURCE for row in train_rows):
        warnings.extend(["contains_weak_rule_labels", "model_may_learn_synthetic_rules", "metrics_may_be_artificial_due_to_weak_rules"])
    if any(_label_quality(row) == WEAK_RULE_LABEL_QUALITY for row in train_rows):
        warnings.append("label_quality_bootstrap_weak_rule_untrusted")
    if any(_label_quality(row) == SOLVER_LABEL_QUALITY for row in train_rows):
        warnings.append("label_quality_bootstrap_solver_untrusted")
    if any(_is_nullish(_feature_value(row, "features.hero_cards")) for row in train_rows):
        warnings.append("hero_cards_missing_or_null")
    if any(_is_nullish(_feature_value(row, "features.board_cards")) for row in train_rows):
        warnings.append("board_cards_missing_or_null")
    if "ALL_IN" not in {_text(row.get("bootstrap_label")) for row in all_rows if _text(row.get("bootstrap_label"))}:
        warnings.append("all_in_absent_because_excluded")
    return warnings


def stratified_split(rows: Sequence[Mapping[str, Any]], *, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    y = [str(row["bootstrap_label"]) for row in rows]
    try:
        train, test = train_test_split(
            list(rows),
            test_size=0.25,
            random_state=random_seed,
            stratify=y,
        )
        return list(train), list(test), None
    except ValueError:
        train, test = train_test_split(list(rows), test_size=0.25, random_state=random_seed)
        return list(train), list(test), "stratified_split_unavailable_used_random_split"


def candidate_model_names(model_type: str) -> list[str]:
    if model_type == "auto":
        return ["dummy", "logistic_regression", "random_forest", "extra_trees"]
    if model_type == "dummy":
        return ["dummy"]
    return ["dummy", model_type]


def train_and_evaluate_model(
    name: str,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    labels: list[str],
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    model = make_model(name)
    x_train = [feature_payload(row, feature_columns=feature_columns) for row in train_rows]
    y_train = [str(row["bootstrap_label"]) for row in train_rows]
    x_test = [feature_payload(row, feature_columns=feature_columns) for row in test_rows]
    y_test = [str(row["bootstrap_label"]) for row in test_rows]
    model.fit(x_train, y_train)
    predictions = [str(value) for value in model.predict(x_test)]
    return {
        "model_name": name,
        "model": model,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 6),
        "confusion_matrix": matrix_as_dict(y_test, predictions, labels),
        "classification_report": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
    }


def make_model(name: str) -> Pipeline:
    if name == "dummy":
        classifier = DummyClassifier(strategy="most_frequent")
    elif name == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    elif name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "extra_trees":
        classifier = ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise BootstrapTrainingError(f"unsupported_model_type:{name}")
    return Pipeline([("vectorizer", DictVectorizer(sparse=False)), ("classifier", classifier)])


def select_best_model(comparisons: Mapping[str, Mapping[str, Any]], *, requested_model_type: str) -> str:
    if requested_model_type not in {"auto", "dummy"}:
        return requested_model_type
    non_dummy = {name: report for name, report in comparisons.items() if name != "dummy"}
    if not non_dummy:
        return "dummy"
    return max(non_dummy.items(), key=lambda item: (item[1]["macro_f1"], item[1]["weighted_f1"], item[1]["accuracy"]))[0]


def feature_payload(row: Mapping[str, Any], *, feature_columns: Sequence[str] = FEATURE_COLUMNS) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for feature in feature_columns:
        if feature in NUMERIC_FEATURES:
            payload[feature] = _float_or_zero(_feature_value(row, feature))
        elif feature in CATEGORICAL_FEATURES:
            payload[feature] = _text(_feature_value(row, feature)) or "UNKNOWN"
        elif feature in CARD_FEATURES:
            payload[feature] = _canonical_text(_feature_value(row, feature)) or "UNKNOWN"
        else:
            raise BootstrapTrainingError(f"unknown_feature_column:{feature}")
    return payload


def matrix_as_dict(truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predictions, labels=labels)
    return {
        expected: {predicted: int(matrix[row_index][col_index]) for col_index, predicted in enumerate(labels)}
        for row_index, expected in enumerate(labels)
    }


def build_feature_schema(
    rows: Sequence[Mapping[str, Any]],
    labels: list[str],
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    numeric_features = [feature for feature in feature_columns if feature in NUMERIC_FEATURES]
    categorical_features = [feature for feature in feature_columns if feature in CATEGORICAL_FEATURES]
    card_features = [feature for feature in feature_columns if feature in CARD_FEATURES]
    categorical_values = {
        feature: sorted({_text(_feature_value(row, feature)) or "UNKNOWN" for row in rows})
        for feature in categorical_features
    }
    card_presence = {
        feature: sum(0 if _is_nullish(_feature_value(row, feature)) else 1 for row in rows)
        for feature in card_features
    }
    return {
        "training_quality": TRAINING_QUALITY,
        "feature_order": list(feature_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "card_features": card_features,
        "categorical_values": categorical_values,
        "card_presence_counts": card_presence,
        "target": "bootstrap_label",
        "labels": labels,
        "forbidden_targets": sorted(STRICT_FORBIDDEN_COLUMNS | {"ALL_IN"}),
        "not_for_production": True,
    }


def build_label_mapping(labels: list[str]) -> dict[str, Any]:
    return {
        "labels": labels,
        "label_to_id": {label: index for index, label in enumerate(labels)},
        "id_to_label": {str(index): label for index, label in enumerate(labels)},
    }


def write_training_outputs(
    *,
    output_dir: str | Path,
    model: Pipeline,
    feature_schema: Mapping[str, Any],
    label_mapping: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": str(path / "model.joblib"),
        "preprocessing": str(path / "preprocessing.joblib"),
        "feature_schema": str(path / "feature_schema.json"),
        "label_mapping": str(path / "label_mapping.json"),
        "evaluation_report": str(path / "evaluation_report.json"),
        "evaluation_report_md": str(path / "evaluation_report.md"),
        "model_card": str(path / "model_card.md"),
    }
    joblib.dump(model, paths["model"])
    joblib.dump({"feature_schema": dict(feature_schema), "label_mapping": dict(label_mapping)}, paths["preprocessing"])
    write_json(feature_schema, paths["feature_schema"])
    write_json(label_mapping, paths["label_mapping"])
    safe_report = without_model_objects(report)
    write_json(safe_report, paths["evaluation_report"])
    Path(paths["evaluation_report_md"]).write_text(render_markdown_report(safe_report), encoding="utf-8")
    Path(paths["model_card"]).write_text(render_model_card(safe_report, feature_schema, label_mapping), encoding="utf-8")
    return paths


def without_model_objects(report: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    comparisons = {}
    for name, value in dict(cleaned.get("model_comparison", {})).items():
        comparisons[name] = {key: item for key, item in value.items() if key != "model"}
    cleaned["model_comparison"] = comparisons
    if "dummy_comparison" in cleaned:
        cleaned["dummy_comparison"] = {key: item for key, item in dict(cleaned["dummy_comparison"]).items() if key != "model"}
    return cleaned


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _feature_value(row: Mapping[str, Any], feature: str) -> Any:
    derived = _derived_feature_value(row, feature)
    if derived is not None:
        return derived
    if feature in row:
        return row.get(feature)
    for alias in FEATURE_ALIASES.get(feature, ()):
        if alias in row:
            return row.get(alias)
    return None


def _derived_feature_value(row: Mapping[str, Any], feature: str) -> Any:
    if feature == "features.active_opponents":
        active = _float_or_none(_raw_feature_value(row, "features.player_active"))
        return None if active is None else max(0.0, active - 1.0)
    if feature == "features.hero_cards_known":
        raw = _raw_feature_value(row, "features.hero_cards") or row.get("hero_cards")
        return 0.0 if _is_nullish(raw) else 1.0
    if feature == "features.equity_gap":
        equity = _float_or_none(_raw_feature_value(row, "features.equity_table"))
        if equity is None:
            equity = _float_or_none(_raw_feature_value(row, "features.equity_1v1"))
        required = _float_or_none(_raw_feature_value(row, "features.equity_required"))
        if equity is None or required is None:
            return None
        return equity - required
    if feature == "features.stack_to_pot_ratio":
        stack = _float_or_none(_raw_feature_value(row, "features.effective_stack"))
        if stack is None:
            stack = _float_or_none(_raw_feature_value(row, "features.hero_stack"))
        if stack is None:
            stack = _float_or_none(row.get("stack"))
        pot = _float_or_none(_raw_feature_value(row, "features.pot"))
        if stack is None or pot is None or pot <= 0:
            return None
        return stack / pot
    if feature.startswith("features.opponent_") and feature.endswith("_avg"):
        metric = feature.removeprefix("features.opponent_").removesuffix("_avg")
        return _opponent_profile_average(row, metric)
    if feature in {"features.hero_stack", "features.effective_stack"}:
        return row.get(feature) or row.get("stack")
    return None


def _raw_feature_value(row: Mapping[str, Any], feature: str) -> Any:
    if feature in row:
        return row.get(feature)
    for alias in FEATURE_ALIASES.get(feature, ()):
        if alias in row:
            return row.get(alias)
    return None


def _opponent_profile_average(row: Mapping[str, Any], metric: str) -> float | None:
    raw = row.get("features.opponent_profiles") or row.get("opponent_profiles")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    values = [_float_or_none(item.get(metric)) for item in raw if isinstance(item, Mapping)]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _label_source(row: Mapping[str, Any]) -> str | None:
    return _text(row.get("label_source") or row.get("audit.label_source") or row.get("metadata.label_source"))


def _label_quality(row: Mapping[str, Any]) -> str | None:
    return _text(row.get("label_quality") or row.get("audit.label_quality"))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bootstrap Model Evaluation",
        "",
        f"- selected_model: `{report['selected_model']}`",
        f"- selection_metric: `{report['selection_metric']}`",
        f"- training_quality: `{report['training_quality']}`",
        f"- not_for_production: `{report['not_for_production']}`",
        f"- contains_weak_rule_labels: `{report['contains_weak_rule_labels']}`",
        f"- model_may_learn_synthetic_rules: `{report['model_may_learn_synthetic_rules']}`",
        f"- accuracy: `{report['accuracy']}`",
        f"- macro_f1: `{report['macro_f1']}`",
        f"- weighted_f1: `{report['weighted_f1']}`",
        "",
        "## Label Distribution",
        "",
        "```json",
        json.dumps(report["label_distribution"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Model Comparison",
        "",
        "```json",
        json.dumps(report["model_comparison"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- `{warning}`" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def render_model_card(report: Mapping[str, Any], feature_schema: Mapping[str, Any], label_mapping: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Bootstrap Action Model Card",
            "",
            "This model is trained from bootstrap candidate rows produced by solver candidates plus optional weak-rule rows.",
            "It is not GTO, not production-ready, and must not be connected to live decisions.",
            "",
            "## Data Origin",
            "",
            f"- input_path: `{report['input_path']}`",
            f"- rows_used: `{report['rows_used']}`",
            f"- contains_weak_rule_labels: `{report['contains_weak_rule_labels']}`",
            f"- model_may_learn_synthetic_rules: `{report['model_may_learn_synthetic_rules']}`",
            "",
            "## Classes",
            "",
            "```json",
            json.dumps(label_mapping["labels"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Features",
            "",
            "```json",
            json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Limits",
            "",
            "- Labels are weak bootstrap labels, not verified strategy labels.",
            "- Weak-rule rows can create synthetic distribution bias.",
            "- Metrics can be artificially high when the model learns generated rules.",
            "- CALL may be absent in the current dataset.",
            "- Raw card and board strings are audit data only; model inputs use equity/context summaries.",
            "",
            "## Next Steps",
            "",
            "- Increase solver-aligned real candidates.",
            "- Add CALL coverage.",
            "- Separate weak-rule validation from solver candidate validation.",
            "- Keep all evaluation offline until data quality is materially better.",
            "",
        ]
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_text(value: Any) -> str | None:
    text = _text(value)
    if text is None or text.lower() in {"null", "none", "[]"}:
        return None
    return text


def _is_nullish(value: Any) -> bool:
    text = _text(value)
    return text is None or text.lower() in {"null", "none", "[]"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input candidate dataset CSV")
    parser.add_argument("--output-dir", default="outputs/bootstrap_model")
    parser.add_argument("--model-type", choices=sorted(ALLOWED_MODEL_TYPES), default="auto")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train_bootstrap_model(
            input_path=args.input,
            output_dir=args.output_dir,
            model_type=args.model_type,
            min_rows=args.min_rows,
            random_seed=args.random_seed,
        )
    except BootstrapTrainingError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
