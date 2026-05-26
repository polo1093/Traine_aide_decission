"""Train the offline bootstrap v5 model from the dist-aligned candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_bootstrap_model import (
    CARD_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    candidate_model_names,
    feature_payload,
    load_candidate_csv,
    select_training_rows,
    select_best_model,
    stratified_split,
    train_and_evaluate_model,
    train_bootstrap_model,
    without_model_objects,
)


DEFAULT_CANDIDATES = Path("outputs/readiness/bootstrap_candidate_dataset_v5/candidates.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/readiness/bootstrap_model_v5")
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "training_report_v5.json"
TARGET_COLUMNS = {"bootstrap_label", "excluded", "exclusion_reason"}
LEAKAGE_PREFIXES = ("audit.", "debug.", "labels.")
LEAKAGE_NAMES = {
    "metadata.label_source",
    "label_source",
    "label_quality",
    "dominant_action",
    "dominant_action_frequency",
    "action_frequencies",
    "raw_action",
    "normalized_action",
    "weak_rule_reason",
    "candidate_confidence",
    "iterations",
    "exploitability_last",
    "villain_hand",
}
SPECIFIC_AUDIT_FEATURES = (
    "features.hero_cards",
    "features.board_cards",
    "features.equity_required",
    "metadata.street",
    "features.hero_position",
)
DERIVED_POKER_FEATURES = {
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_required",
    "features.equity_gap",
    "features.ev",
    "features.call_max",
}
LEAKAGE_SOURCE_FIELDS = {
    "bootstrap_label",
    "labels.final_action",
    "audit.dominant_action",
    "audit.normalized_action",
    "audit.raw_action",
    "audit.action_frequencies",
    "dominant_action",
    "normalized_action",
    "raw_action",
    "action_frequencies",
    "solver_action_candidate",
    "average_strategy",
}
LABELS = ("CHECK", "FOLD", "RAISE")
NEAR_ZERO_STD = 1e-12
ZERO_DOMINANCE_THRESHOLD = 0.995
SUSPECT_LABEL_SPREAD_RATIO = 0.8
SUSPECT_BINARY_SPREAD = 0.75
ACTION_GATING_FEATURES = {"features.has_check", "features.has_call", "features.has_raise"}
AUDIT_ONLY_PATTERNS = (
    "features.hero_cards",
    "features.board_cards",
    "features.players",
    "features.opponent_profiles",
    "features.buttons",
    "features.buttons_active",
    "features.hero_hand",
    "features.board",
    "villain_hand",
    "hero_cards",
    "board_cards",
)


def train_bootstrap_v5(
    *,
    input_csv: str | Path = DEFAULT_CANDIDATES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_type: str = "auto",
    min_rows: int = 100,
    random_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    rows, fieldnames = load_candidate_csv(input_path)
    usable_rows = select_training_rows(rows)
    feature_bias_audit = build_feature_bias_audit(usable_rows, feature_columns=FEATURE_COLUMNS)
    features_model_used = [
        feature for feature in FEATURE_COLUMNS
        if feature not in set(feature_bias_audit["constant_or_useless_features"])
    ]
    training = train_bootstrap_model(
        input_path=input_path,
        output_dir=output_path,
        model_type=model_type,
        min_rows=min_rows,
        random_seed=random_seed,
        feature_columns=features_model_used,
    )
    ignored_columns = sorted(column for column in fieldnames if column not in FEATURE_COLUMNS and column not in TARGET_COLUMNS)
    leakage_columns = sorted(column for column in fieldnames if is_leakage_column(column))
    leakage_used = sorted(column for column in leakage_columns if column in features_model_used)

    ablation_report = build_ablation_report(
        usable_rows,
        feature_bias_audit=feature_bias_audit,
        features_model_used=features_model_used,
        random_seed=random_seed,
    )
    feature_contract = build_feature_contract(
        fieldnames=fieldnames,
        features_model_used=features_model_used,
        feature_bias_audit=feature_bias_audit,
    )
    preprocessing_schema = build_preprocessing_schema(usable_rows, feature_columns=features_model_used)
    heatmaps = write_v5_feature_heatmaps(usable_rows, output_path)
    report = {
        **training,
        "training_report_version": "v5",
        "dataset_schema": "dist_aligned_flat_csv",
        "train_test_split_seed": random_seed,
        "dataset_size": {
            "total": len(rows),
            "usable": len(usable_rows),
            "excluded": len(rows) - len(usable_rows),
        },
        "model_feature_columns": list(features_model_used),
        "candidate_feature_columns": list(FEATURE_COLUMNS),
        "constant_or_useless_features": feature_bias_audit["constant_or_useless_features"],
        "suspect_features": feature_bias_audit["suspect_features"],
        "ignored_columns": ignored_columns,
        "leakage_columns_excluded": leakage_columns,
        "leakage_columns_used_by_model": leakage_used,
        "null_counts_by_feature": null_counts_by_feature(usable_rows),
        "categorical_cardinality": categorical_cardinality(usable_rows),
        "specific_feature_audit": specific_feature_audit(usable_rows),
        "heatmaps": heatmaps,
        "not_gto": True,
        "not_for_production": True,
        "call_added": False,
        "bot_live_connection": "not_modified",
    }

    write_json(feature_contract, output_path / "feature_contract.json")
    write_json(preprocessing_schema, output_path / "preprocessing_schema.json")
    write_json(feature_bias_audit, output_path / "feature_bias_audit.json")
    write_json(ablation_report, output_path / "ablation_report.json")
    write_json(report, output_path / "training_report_v5.json")
    write_json(report, output_path / "training_report.json")
    return report


def write_v5_feature_heatmaps(rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    numeric = build_v5_numeric_heatmap(rows)
    categorical = build_v5_categorical_heatmap(rows)

    numeric_svg = output_path / "v5_input_numeric_by_output_heatmap.svg"
    numeric_json = output_path / "v5_input_numeric_by_output_heatmap.json"
    categorical_svg = output_path / "v5_input_category_by_output_heatmap.svg"
    categorical_json = output_path / "v5_input_category_by_output_heatmap.json"

    write_v5_matrix_heatmap_svg(
        numeric["matrix"],
        numeric_svg,
        title="V5 Inputs numeriques normalises -> sortie",
        subtitle="Valeurs affichees = moyenne par label, normalisee min-max par feature entre 0 et 1.",
    )
    write_json(numeric, numeric_json)
    write_v5_matrix_heatmap_svg(
        categorical["matrix"],
        categorical_svg,
        title="V5 Inputs categoriels normalises -> sortie",
        subtitle="Valeurs affichees = proportion du label ayant cette valeur d entree, entre 0 et 1.",
    )
    write_json(categorical, categorical_json)
    return {
        "v5_input_numeric_by_output_heatmap": str(numeric_svg),
        "v5_input_numeric_by_output_heatmap_data": str(numeric_json),
        "v5_input_category_by_output_heatmap": str(categorical_svg),
        "v5_input_category_by_output_heatmap_data": str(categorical_json),
    }


def build_v5_numeric_heatmap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ["CHECK", "FOLD", "RAISE"]
    matrix = []
    raw_means: dict[str, dict[str, float | None]] = {}
    for feature in NUMERIC_FEATURES:
        means = {
            label: mean_or_none(
                float_or_none(feature_payload(row).get(feature))
                for row in rows
                if str(row.get("bootstrap_label")) == label
            )
            for label in labels
        }
        raw_means[feature] = means
        values = [value for value in means.values() if value is not None]
        row_min = min(values) if values else 0.0
        row_max = max(values) if values else 1.0
        normalized = {
            label: normalize_01(value, row_min=row_min, row_max=row_max)
            for label, value in means.items()
        }
        matrix.append({"feature": feature, "values": normalized})
    return {
        "schema": "v5_input_numeric_by_output_heatmap",
        "normalization": "per_feature_min_max_across_output_labels",
        "columns": labels,
        "features": list(NUMERIC_FEATURES),
        "matrix": matrix,
        "raw_mean_by_label": raw_means,
    }


def build_v5_categorical_heatmap(rows: Sequence[Mapping[str, Any]], *, max_values_per_feature: int = 8) -> dict[str, Any]:
    labels = ["CHECK", "FOLD", "RAISE"]
    features = list(CATEGORICAL_FEATURES) + list(CARD_FEATURES)
    label_totals = {label: sum(1 for row in rows if str(row.get("bootstrap_label")) == label) for label in labels}
    matrix = []
    selected_values: dict[str, list[str]] = {}
    for feature in features:
        counts = Counter(str(feature_payload(row).get(feature) or "UNKNOWN") for row in rows)
        values = [value for value, _ in counts.most_common(max_values_per_feature)]
        selected_values[feature] = values
        for feature_value in values:
            normalized = {}
            for label in labels:
                denominator = label_totals[label]
                numerator = sum(
                    1
                    for row in rows
                    if str(row.get("bootstrap_label")) == label
                    and str(feature_payload(row).get(feature) or "UNKNOWN") == feature_value
                )
                normalized[label] = round(numerator / denominator, 6) if denominator else None
            matrix.append({"feature": f"{feature}={short_heatmap_label(feature_value)}", "values": normalized})
    return {
        "schema": "v5_input_category_by_output_heatmap",
        "normalization": "per_output_label_proportion",
        "columns": labels,
        "features": features,
        "selected_values_by_feature": selected_values,
        "matrix": matrix,
    }


def mean_or_none(values: Sequence[float | None] | Any) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 6)


def normalize_01(value: float | None, *, row_min: float, row_max: float) -> float | None:
    if value is None:
        return None
    if row_max == row_min:
        return 0.0
    return round((value - row_min) / (row_max - row_min), 6)


def write_v5_matrix_heatmap_svg(matrix: Sequence[Mapping[str, Any]], output_path: Path, *, title: str, subtitle: str) -> None:
    labels = ["CHECK", "FOLD", "RAISE"]
    cell_w = 88
    cell_h = 34
    left = 315
    top = 82
    width = left + cell_w * len(labels) + 40
    height = top + cell_h * len(matrix) + 46
    lines = [
        svg_header(width, height),
        f'<text x="20" y="30" font-size="18" font-family="Arial">{escape_xml(title)}</text>',
        f'<text x="20" y="54" font-size="12" font-family="Arial" fill="#555">{escape_xml(subtitle)}</text>',
    ]
    for index, label in enumerate(labels):
        x = left + index * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="74" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
    for y_index, row in enumerate(matrix):
        y = top + y_index * cell_h
        feature = str(row["feature"])
        values = row["values"]
        lines.append(
            f'<text x="{left - 12}" y="{y + 22}" text-anchor="end" font-size="12" '
            f'font-family="Arial">{escape_xml(feature)}</text>'
        )
        for x_index, label in enumerate(labels):
            x = left + x_index * cell_w
            value = values.get(label)
            ratio = 0.0 if value is None else max(0.0, min(1.0, float(value)))
            text = "NA" if value is None else f"{ratio:.3f}".rstrip("0").rstrip(".")
            fill = heat_color(ratio)
            text_color = "#ffffff" if ratio > 0.55 else "#111111"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(
                f'<text x="{x + cell_w / 2}" y="{y + 22}" text-anchor="middle" font-size="11" '
                f'font-family="Arial" fill="{text_color}">{escape_xml(text)}</text>'
            )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def short_heatmap_label(value: str, *, limit: int = 42) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if text.lower() in {"", "none", "null", "nan", "unknown"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def build_feature_bias_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    feature_statistics = {
        feature: feature_bias_statistics(rows, feature)
        for feature in feature_columns
    }
    constant_or_useless = sorted(
        feature
        for feature, stats in feature_statistics.items()
        if is_constant_or_useless(stats)
    )
    suspect = detect_suspect_features(feature_statistics)
    leakage_check = derived_feature_dependency_check()
    features_model_candidate = [
        feature for feature in feature_columns
        if feature not in set(constant_or_useless)
    ]
    return {
        "schema": "bootstrap_v5_feature_bias_audit",
        "row_count": len(rows),
        "labels": list(LABELS),
        "label_distribution": dict(sorted(Counter(str(row.get("bootstrap_label")) for row in rows).items())),
        "feature_columns_audited": list(feature_columns),
        "feature_statistics": feature_statistics,
        "constant_or_useless_features": constant_or_useless,
        "suspect_features": suspect,
        "features_model_candidate_after_constant_exclusion": features_model_candidate,
        "features_audit_only": [],
        "derived_feature_dependency_check": leakage_check,
        "leakage_used_by_model": [],
        "normalization": "none_raw_sklearn_feature_payload_values",
        "not_for_production": True,
    }


def feature_bias_statistics(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    values = [feature_payload(row).get(feature) for row in rows]
    numbers = [float_or_none(value) for value in values]
    numeric_values = [value for value in numbers if value is not None]
    by_label: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        label_values = [
            feature_payload(row).get(feature)
            for row in rows
            if str(row.get("bootstrap_label")) == label
        ]
        label_numbers = [float_or_none(value) for value in label_values]
        label_numeric = [value for value in label_numbers if value is not None]
        if feature in NUMERIC_FEATURES:
            by_label[label] = {
                "count": len(label_values),
                "mean": rounded_mean(label_numeric),
                "std": rounded_std(label_numeric),
            }
        else:
            counts = Counter(str(value) for value in label_values)
            by_label[label] = {
                "count": len(label_values),
                "mean": None,
                "std": None,
                "top_values": dict(counts.most_common(10)),
            }

    counter = Counter(str(value) for value in values)
    stats: dict[str, Any] = {
        "type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
        "null_count": sum(1 for value in values if is_null_feature_value(value)),
        "raw_null_count": sum(1 for row in rows if is_null_feature_value(row.get(feature))),
        "nunique": len(counter),
        "min": round(min(numeric_values), 6) if numeric_values else None,
        "max": round(max(numeric_values), 6) if numeric_values else None,
        "mean": rounded_mean(numeric_values),
        "std": rounded_std(numeric_values),
        "zero_proportion": round(sum(1 for value in numeric_values if abs(value) <= NEAR_ZERO_STD) / len(values), 6) if values else 0.0,
        "mean_by_label": {label: by_label[label]["mean"] for label in LABELS},
        "std_by_label": {label: by_label[label]["std"] for label in LABELS},
        "counts_by_label": {label: by_label[label]["count"] for label in LABELS},
    }
    if feature not in NUMERIC_FEATURES:
        stats["top_values"] = dict(counter.most_common(20))
        stats["zero_proportion"] = round(sum(1 for value in values if str(value) in {"0", "0.0", "False", "false"}) / len(values), 6) if values else 0.0
        stats["values_by_label"] = {
            label: by_label[label].get("top_values", {})
            for label in LABELS
        }
    return stats


def rounded_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def rounded_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 6)


def is_constant_or_useless(stats: Mapping[str, Any]) -> bool:
    if int(stats.get("nunique") or 0) <= 1:
        return True
    std = stats.get("std")
    if isinstance(std, (int, float)) and abs(float(std)) <= NEAR_ZERO_STD:
        return True
    zero_proportion = stats.get("zero_proportion")
    return bool(isinstance(zero_proportion, (int, float)) and float(zero_proportion) >= ZERO_DOMINANCE_THRESHOLD)


def detect_suspect_features(feature_statistics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    suspects: list[dict[str, Any]] = []
    focus = {
        "features.pot",
        "features.to_call",
        "features.to_call_pot_ratio",
        "features.equity_required",
        "features.equity_gap",
        "features.ev",
        "features.call_max",
        "features.equity_table",
        "features.equity_1v1",
        *ACTION_GATING_FEATURES,
    }
    for feature, stats in feature_statistics.items():
        means = {
            label: value
            for label, value in dict(stats.get("mean_by_label", {})).items()
            if isinstance(value, (int, float))
        }
        if len(means) < 2:
            continue
        global_min = stats.get("min")
        global_max = stats.get("max")
        if not isinstance(global_min, (int, float)) or not isinstance(global_max, (int, float)):
            continue
        feature_range = float(global_max) - float(global_min)
        if feature_range <= NEAR_ZERO_STD:
            continue
        spread = max(means.values()) - min(means.values())
        ratio = spread / feature_range
        binary_like = set(float(value) for value in means.values()).issubset({0.0, 1.0})
        threshold_hit = ratio >= SUSPECT_LABEL_SPREAD_RATIO or (binary_like and spread >= SUSPECT_BINARY_SPREAD)
        if not threshold_hit and feature not in focus:
            continue
        if not threshold_hit:
            continue
        dominant_label = max(means.items(), key=lambda item: item[1])[0]
        reason = "large_label_mean_spread"
        if feature in ACTION_GATING_FEATURES:
            reason = "possible_action_gating_feature_strongly_label_separating"
        elif feature in {"features.pot", "features.to_call", "features.to_call_pot_ratio", "features.equity_required"}:
            reason = "pot_call_required_equity_strongly_linked_to_label"
        elif feature in {"features.equity_gap", "features.ev", "features.call_max", "features.equity_table", "features.equity_1v1"}:
            reason = "equity_ev_callmax_strongly_linked_to_label"
        suspects.append(
            {
                "feature": feature,
                "suspected_label": dominant_label,
                "reason": reason,
                "label_means": means,
                "spread": round(spread, 6),
                "global_range": round(feature_range, 6),
                "spread_to_range_ratio": round(ratio, 6),
            }
        )
    return sorted(suspects, key=lambda item: (item["reason"], item["feature"]))


def derived_feature_dependency_check() -> dict[str, Any]:
    formula_inputs = {
        "features.equity_table": ["features.hero_cards", "villain_hand", "features.board_cards", "features.player_active"],
        "features.equity_1v1": ["features.hero_cards", "villain_hand", "features.board_cards"],
        "features.equity_required": ["features.pot", "features.to_call"],
        "features.equity_gap": ["features.equity_table", "features.equity_1v1", "features.equity_required"],
        "features.ev": ["features.equity_table", "features.equity_1v1", "features.pot", "features.to_call"],
        "features.call_max": ["features.equity_table", "features.equity_1v1", "features.pot"],
    }
    features: dict[str, Any] = {}
    for feature in sorted(DERIVED_POKER_FEATURES):
        inputs = formula_inputs[feature]
        forbidden = sorted(set(inputs) & LEAKAGE_SOURCE_FIELDS)
        features[feature] = {
            "formula_inputs": inputs,
            "forbidden_source_fields_checked": sorted(LEAKAGE_SOURCE_FIELDS),
            "forbidden_source_fields_used": forbidden,
            "depends_on_label_or_solver_output": bool(forbidden),
            "status": "ok" if not forbidden else "leakage_risk",
        }
    statuses = {item["status"] for item in features.values()}
    return {
        "status": "ok" if statuses == {"ok"} else "leakage_risk",
        "features": features,
    }


def build_ablation_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_bias_audit: Mapping[str, Any],
    features_model_used: Sequence[str],
    random_seed: int,
) -> dict[str, Any]:
    constants = set(feature_bias_audit.get("constant_or_useless_features", []))
    variants = [
        ("full_current", list(FEATURE_COLUMNS), None),
        ("without_constant_features", [feature for feature in FEATURE_COLUMNS if feature not in constants], None),
        (
            "without_equity_gap_ev_call_max",
            [feature for feature in features_model_used if feature not in {"features.equity_gap", "features.ev", "features.call_max"}],
            None,
        ),
        (
            "without_equity_table_equity_1v1",
            [feature for feature in features_model_used if feature not in {"features.equity_table", "features.equity_1v1"}],
            None,
        ),
        (
            "minimal_pot_call_stack_actions_position",
            [
                feature for feature in features_model_used
                if feature in {
                    "features.pot",
                    "features.to_call",
                    "features.to_call_pot_ratio",
                    "features.hero_stack",
                    "features.effective_stack",
                    "features.stack_to_pot_ratio",
                    "features.has_check",
                    "features.has_call",
                    "features.has_raise",
                    "metadata.street",
                    "features.hero_position",
                }
            ],
            None,
        ),
        ("solver_candidate_only", list(features_model_used), "solver_candidate"),
        ("weak_rule_bootstrap_only", list(features_model_used), "weak_rule_bootstrap"),
    ]
    results = [
        evaluate_ablation_variant(
            name=name,
            rows=rows,
            feature_columns=feature_columns,
            random_seed=random_seed,
            label_source=label_source,
        )
        for name, feature_columns, label_source in variants
    ]
    return {
        "schema": "bootstrap_v5_ablation_report",
        "row_count": len(rows),
        "random_seed": random_seed,
        "variants": results,
        "not_for_production": True,
        "bot_live_connection": "not_modified",
        "call_added": False,
    }


def evaluate_ablation_variant(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    random_seed: int,
    label_source: str | None,
) -> dict[str, Any]:
    filtered_rows = [
        row for row in rows
        if label_source is None or row_label_source(row) == label_source
    ]
    labels = sorted({str(row.get("bootstrap_label")) for row in filtered_rows if row.get("bootstrap_label")})
    base = {
        "name": name,
        "label_source_filter": label_source,
        "rows_used": len(filtered_rows),
        "label_distribution": dict(sorted(Counter(str(row.get("bootstrap_label")) for row in filtered_rows).items())),
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
    }
    if len(filtered_rows) < 4:
        return {**base, "status": "skipped", "reason": "not_enough_rows"}
    if len(labels) < 2:
        return {**base, "status": "skipped", "reason": "single_class_dataset"}
    if not feature_columns:
        return {**base, "status": "skipped", "reason": "no_features"}
    try:
        train_rows, test_rows, split_warning = stratified_split(filtered_rows, random_seed=random_seed)
        comparisons = {
            model_name: train_and_evaluate_model(
                model_name,
                train_rows,
                test_rows,
                labels,
                feature_columns=feature_columns,
            )
            for model_name in candidate_model_names("auto")
        }
        best_name = select_best_model(comparisons, requested_model_type="auto")
        best = comparisons[best_name]
        cleaned = without_model_objects({"model_comparison": comparisons})["model_comparison"]
        return {
            **base,
            "status": "ok",
            "train_size": len(train_rows),
            "test_size": len(test_rows),
            "split_warning": split_warning,
            "selected_model": best_name,
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "weighted_f1": best["weighted_f1"],
            "confusion_matrix": best["confusion_matrix"],
            "model_comparison": cleaned,
        }
    except Exception as exc:  # pragma: no cover - report should survive audit edge cases.
        return {**base, "status": "failed", "reason": str(exc)}


def row_label_source(row: Mapping[str, Any]) -> str:
    return str(row.get("label_source") or row.get("audit.label_source") or row.get("metadata.label_source") or "")


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def heat_color(ratio: float) -> str:
    safe = max(0.0, min(1.0, float(ratio)))
    red = int(245 - safe * 180)
    green = int(247 - safe * 120)
    blue = int(250 - safe * 40)
    return f"#{red:02x}{green:02x}{blue:02x}"


def escape_xml(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dist_sample_prediction_report(*, model_dir: str | Path, dist_sample: str | Path) -> dict[str, Any]:
    from models.predict_bootstrap_model import predict_bootstrap_model

    snapshot = load_first_dist_snapshot(dist_sample)
    payload = flatten_dist_snapshot(snapshot)
    result = predict_bootstrap_model(model_dir=model_dir, input_payload=payload)
    feature_order = prediction_feature_order(model_dir)
    features = feature_payload(payload, feature_columns=feature_order)
    return {
        "status": "ok" if result.get("status") == "ok" else "failed",
        "dist_sample": str(dist_sample),
        "snapshot_id": snapshot.get("snapshot_id"),
        "features_extracted": list(features),
        "feature_order_matches_contract": list(features) == list(feature_order),
        "prediction": result.get("prediction"),
        "prediction_status": result.get("status"),
        "warnings": result.get("warnings", []),
        "audit_debug_label_features_used": [
            feature for feature in feature_order
            if feature.startswith(("audit.", "debug.", "labels."))
        ],
    }


def prediction_feature_order(model_dir: str | Path) -> list[str]:
    schema_path = Path(model_dir) / "feature_schema.json"
    if not schema_path.exists():
        return list(FEATURE_COLUMNS)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return list(schema.get("feature_order") or FEATURE_COLUMNS)


def build_feature_contract(
    *,
    fieldnames: Sequence[str],
    features_model_used: Sequence[str] | None = None,
    feature_bias_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_features = list(features_model_used or FEATURE_COLUMNS)
    constants = sorted(feature_bias_audit.get("constant_or_useless_features", [])) if feature_bias_audit else []
    feature_stats = dict(feature_bias_audit.get("feature_statistics", {})) if feature_bias_audit else {}
    missing_or_unavailable = sorted(
        feature
        for feature, stats in feature_stats.items()
        if int(stats.get("raw_null_count") or 0) >= int(feature_bias_audit.get("row_count") or 0)
    ) if feature_bias_audit else []
    leakage_columns = sorted(column for column in fieldnames if is_leakage_column(column))
    audit_only = sorted(
        column for column in fieldnames
        if column not in model_features
        and (
            column not in TARGET_COLUMNS
            or any(column.startswith(pattern) or column == pattern for pattern in AUDIT_ONLY_PATTERNS)
        )
        and (
            is_audit_only_column(column)
            or column in leakage_columns
            or column not in FEATURE_COLUMNS
        )
    )
    leakage_used = sorted(column for column in leakage_columns if column in model_features)
    return {
        "schema": "bootstrap_v5_dist_aligned_flat_csv",
        "feature_order": model_features,
        "features_model_used": model_features,
        "features_candidate_order": list(FEATURE_COLUMNS),
        "features_audit_only": audit_only,
        "features_constant_excluded": constants,
        "features_leakage_excluded": leakage_columns,
        "features_not_available_yet": missing_or_unavailable,
        "numeric_features": [feature for feature in model_features if feature in NUMERIC_FEATURES],
        "categorical_features": [feature for feature in model_features if feature in CATEGORICAL_FEATURES],
        "card_features": [feature for feature in model_features if feature in CARD_FEATURES],
        "target": "bootstrap_label",
        "allowed_predictions": ["CHECK", "FOLD", "RAISE"],
        "not_allowed_yet": ["CALL"],
        "ignored_columns": sorted(column for column in fieldnames if column not in model_features and column not in TARGET_COLUMNS),
        "leakage_columns_excluded": leakage_columns,
        "leakage_columns_used_by_model": leakage_used,
        "derived_feature_dependency_check": feature_bias_audit.get("derived_feature_dependency_check") if feature_bias_audit else derived_feature_dependency_check(),
        "not_gto": True,
        "not_for_production": True,
    }


def is_audit_only_column(column: str) -> bool:
    return any(column == pattern or column.startswith(pattern + ".") for pattern in AUDIT_ONLY_PATTERNS)


def build_preprocessing_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    categorical_features = [feature for feature in feature_columns if feature in CATEGORICAL_FEATURES]
    card_features = [feature for feature in feature_columns if feature in CARD_FEATURES]
    return {
        "feature_order": list(feature_columns),
        "dict_vectorizer_input": "ordered_feature_payload",
        "null_policy": {
            "numeric": "missing_or_invalid_to_0.0",
            "categorical": "missing_to_UNKNOWN",
            "card": "missing_to_UNKNOWN",
        },
        "categorical_values": {
            feature: sorted({str(feature_payload(row)[feature]) for row in rows})
            for feature in categorical_features
        },
        "card_presence_counts": {
            feature: sum(1 for row in rows if str(feature_payload(row)[feature]) != "UNKNOWN")
            for feature in card_features
        },
    }


def null_counts_by_feature(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in FEATURE_COLUMNS:
        counts[feature] = sum(1 for row in rows if is_null_feature_value(feature_payload(row).get(feature)))
    return counts


def categorical_cardinality(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    features = list(CATEGORICAL_FEATURES) + list(CARD_FEATURES)
    return {
        feature: len({str(feature_payload(row)[feature]) for row in rows})
        for feature in features
    }


def specific_feature_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for feature in SPECIFIC_AUDIT_FEATURES:
        values = [
            row.get(feature) if feature in row else feature_payload(row).get(feature)
            for row in rows
        ]
        counter = Counter(str(value) for value in values)
        audit[feature] = {
            "null_count": sum(1 for value in values if is_null_feature_value(value)),
            "cardinality": len(counter),
            "top_values": dict(counter.most_common(10)),
        }
    return audit


def is_leakage_column(column: str) -> bool:
    return column.startswith(LEAKAGE_PREFIXES) or column in LEAKAGE_NAMES


def is_null_feature_value(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"", "none", "null", "unknown", "[]"}


def flatten_dist_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                flat[f"{key}.{child_key}"] = child_value
        else:
            flat[key] = value
    return flat


def load_first_dist_snapshot(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                value = json.loads(text)
                if isinstance(value, dict):
                    return value
    raise ValueError(f"dist_snapshot_missing:{path}")


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-type", default="auto")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_bootstrap_v5(
        input_csv=args.input,
        output_dir=args.output_dir,
        model_type=args.model_type,
        min_rows=args.min_rows,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
