"""Advanced diagnostics and BB-normalized experimental training for bootstrap v5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.report_bootstrap_v5_feature_views import md
from experiments.train_bootstrap_v5 import LABELS, escape_xml, float_or_none, heat_color, is_leakage_column, svg_header
from models.train_bootstrap_model import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_label_mapping,
    candidate_model_names,
    feature_payload,
    load_candidate_csv,
    make_model,
    matrix_as_dict,
    select_best_model,
    select_training_rows,
    stratified_split,
    without_model_objects,
    write_json,
)


DEFAULT_INPUT = Path("outputs/readiness/bootstrap_candidate_dataset_v5_4000/candidates.csv")
DEFAULT_CURRENT_MODEL_DIR = Path("outputs/readiness/bootstrap_model_v5")
DEFAULT_BB_DATASET = Path("outputs/readiness/bootstrap_candidate_dataset_v5_4000_bb/candidates.csv")
DEFAULT_BB_MODEL_DIR = Path("outputs/readiness/bootstrap_model_v5_bb")
EQUITY_DERIVED_FEATURES = {
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_required",
    "features.equity_gap",
    "features.ev",
    "features.call_max",
}
POT_STACK_ACTION_POSITION_FEATURES = {
    "features.pot",
    "features.to_call",
    "features.to_call_pot_ratio",
    "features.hero_stack",
    "features.effective_stack",
    "features.stack_to_pot_ratio",
    "features.has_check",
    "features.has_call",
    "features.hero_position",
}
BB_NUMERIC_FEATURES = (
    "features.pot_bb",
    "features.to_call_bb",
    "features.to_call_pot_ratio",
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_required",
    "features.equity_gap",
    "features.ev_bb",
    "features.call_max_bb",
    "features.hero_stack_bb",
    "features.effective_stack_bb",
    "features.stack_to_pot_ratio",
    "features.effective_stack_bb_to_pot_bb",
    "features.has_check",
    "features.has_call",
)
BB_CATEGORICAL_FEATURES = ("features.hero_position",)
BB_FEATURE_COLUMNS = BB_NUMERIC_FEATURES + BB_CATEGORICAL_FEATURES
RAW_AMOUNT_COLUMNS = (
    "features.pot",
    "features.to_call",
    "features.hero_stack",
    "features.effective_stack",
    "features.ev",
    "features.call_max",
)
WATCHED_PAIRS = {
    tuple(sorted(pair))
    for pair in [
        ("features.equity_table", "features.equity_1v1"),
        ("features.equity_gap", "features.ev"),
        ("features.equity_gap", "features.call_max"),
        ("features.ev", "features.call_max"),
        ("features.pot", "features.to_call"),
        ("features.hero_stack", "features.effective_stack"),
        ("features.pot", "features.stack_to_pot_ratio"),
        ("features.to_call", "features.equity_required"),
    ]
}


def run_advanced_diagnostics(
    *,
    input_csv: str | Path = DEFAULT_INPUT,
    current_model_dir: str | Path = DEFAULT_CURRENT_MODEL_DIR,
    bb_dataset_csv: str | Path = DEFAULT_BB_DATASET,
    bb_model_dir: str | Path = DEFAULT_BB_MODEL_DIR,
    random_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    input_path = Path(input_csv)
    current_dir = Path(current_model_dir)
    bb_dataset_path = Path(bb_dataset_csv)
    bb_dir = Path(bb_model_dir)
    before_hash = file_sha256(current_dir / "model.joblib")

    rows, _ = load_candidate_csv(input_path)
    usable_rows = select_training_rows(rows)
    current_contract = read_json(current_dir / "feature_contract.json")
    current_report = read_json(current_dir / "training_report_v5.json")
    current_features = list(current_contract.get("features_model_used", []))

    correlation = build_correlation_report(usable_rows, current_features)
    write_json(correlation["json"], current_dir / "input_feature_correlation_matrix.json")
    write_correlation_svg(correlation["json"], current_dir / "input_feature_correlation_matrix.svg")
    (current_dir / "high_correlation_pairs.md").write_text(render_high_correlation_pairs(correlation), encoding="utf-8")

    multicollinearity = build_multicollinearity_report(correlation)
    write_json(multicollinearity, current_dir / "multicollinearity_report.json")
    (current_dir / "multicollinearity_report.md").write_text(render_multicollinearity_report(multicollinearity), encoding="utf-8")

    learning_curve = build_learning_curve_report(usable_rows, current_features, model_name=current_report.get("model_type", "random_forest"), random_seed=random_seed)
    write_json(learning_curve, current_dir / "learning_curve_report.json")
    write_learning_curve_svg(learning_curve, current_dir / "learning_curve.svg")
    (current_dir / "learning_curve_report.md").write_text(render_learning_curve_report(learning_curve), encoding="utf-8")

    robust = build_robust_generalization_report(usable_rows, current_features, random_seed=random_seed)
    write_json(robust, current_dir / "robust_generalization_report.json")
    (current_dir / "robust_generalization_report.md").write_text(render_robust_report(robust), encoding="utf-8")

    bb_export = export_bb_dataset(input_path=input_path, output_csv=bb_dataset_path)
    bb_training = train_bb_model(input_csv=bb_dataset_path, output_dir=bb_dir, random_seed=random_seed)
    bb_rows, _ = load_candidate_csv(bb_dataset_path)
    bb_usable_rows = select_training_rows(bb_rows)
    bb_learning_curve = build_learning_curve_report(
        bb_usable_rows,
        list(bb_training["model_feature_columns"]),
        model_name=bb_training["model_type"],
        random_seed=random_seed,
        payload_fn=bb_feature_payload,
    )
    bb_correlation = build_correlation_report(bb_usable_rows, list(bb_training["model_feature_columns"]), payload_fn=bb_feature_payload)
    bb_robust = build_robust_generalization_report(
        bb_usable_rows,
        list(bb_training["model_feature_columns"]),
        random_seed=random_seed,
        payload_fn=bb_feature_payload,
        equity_features={
            "features.equity_table",
            "features.equity_1v1",
            "features.equity_required",
            "features.equity_gap",
            "features.ev_bb",
            "features.call_max_bb",
        },
        pot_stack_features={
            "features.pot_bb",
            "features.to_call_bb",
            "features.to_call_pot_ratio",
            "features.hero_stack_bb",
            "features.effective_stack_bb",
            "features.stack_to_pot_ratio",
            "features.has_check",
            "features.has_call",
            "features.hero_position",
        },
    )
    current_coefficients = logistic_top_coefficients(usable_rows, current_features, payload_fn=feature_payload)
    bb_coefficients = logistic_top_coefficients(bb_usable_rows, list(bb_training["model_feature_columns"]), payload_fn=bb_feature_payload)
    comparison = build_current_vs_bb_comparison(
        current_report=current_report,
        bb_report=bb_training,
        current_learning_curve=learning_curve,
        bb_learning_curve=bb_learning_curve,
        current_correlation=correlation["json"],
        bb_correlation=bb_correlation["json"],
        current_robust=robust,
        bb_robust=bb_robust,
        current_coefficients=current_coefficients,
        bb_coefficients=bb_coefficients,
    )
    write_json(comparison, bb_dir / "comparison_current_vs_bb.json")
    (bb_dir / "comparison_current_vs_bb.md").write_text(render_current_vs_bb_comparison(comparison), encoding="utf-8")

    after_hash = file_sha256(current_dir / "model.joblib")
    summary = {
        "status": "ok",
        "current_model_unchanged": before_hash == after_hash,
        "current_model_hash_before": before_hash,
        "current_model_hash_after": after_hash,
        "bb_export": bb_export,
        "outputs": {
            "input_feature_correlation_matrix_svg": str(current_dir / "input_feature_correlation_matrix.svg"),
            "input_feature_correlation_matrix_json": str(current_dir / "input_feature_correlation_matrix.json"),
            "high_correlation_pairs": str(current_dir / "high_correlation_pairs.md"),
            "multicollinearity_report_json": str(current_dir / "multicollinearity_report.json"),
            "multicollinearity_report_md": str(current_dir / "multicollinearity_report.md"),
            "learning_curve_svg": str(current_dir / "learning_curve.svg"),
            "learning_curve_report_json": str(current_dir / "learning_curve_report.json"),
            "learning_curve_report_md": str(current_dir / "learning_curve_report.md"),
            "robust_generalization_report_json": str(current_dir / "robust_generalization_report.json"),
            "robust_generalization_report_md": str(current_dir / "robust_generalization_report.md"),
            "bb_dataset": str(bb_dataset_path),
            "bb_model": str(bb_dir / "model.joblib"),
            "bb_training_report": str(bb_dir / "training_report.json"),
            "bb_feature_contract": str(bb_dir / "feature_contract.json"),
            "bb_preprocessing_schema": str(bb_dir / "preprocessing_schema.json"),
            "comparison_current_vs_bb_json": str(bb_dir / "comparison_current_vs_bb.json"),
            "comparison_current_vs_bb_md": str(bb_dir / "comparison_current_vs_bb.md"),
        },
        "call_added": False,
        "bot_live_connection": "not_modified",
    }
    write_json(summary, current_dir / "advanced_diagnostics_summary.json")
    return summary


def build_correlation_report(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    payload_fn: Any = feature_payload,
) -> dict[str, Any]:
    encoded_names, matrix = encoded_feature_matrix(rows, feature_columns, payload_fn=payload_fn)
    corr = np.corrcoef(matrix, rowvar=False) if matrix.shape[1] else np.zeros((0, 0))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    pairs = high_correlation_pairs(encoded_names, corr)
    return {
        "json": {
            "schema": "bootstrap_v5_input_feature_correlation_matrix",
            "row_count": len(rows),
            "feature_order": list(feature_columns),
            "encoded_feature_order": encoded_names,
            "matrix": [[round(float(value), 6) for value in row] for row in corr.tolist()],
            "high_correlation_pairs_abs_gt_0_80": [pair for pair in pairs if pair["abs_correlation"] > 0.80],
            "high_correlation_pairs_abs_gt_0_90": [pair for pair in pairs if pair["abs_correlation"] > 0.90],
            "watched_pairs": watched_pair_findings(encoded_names, corr),
        },
        "pairs": pairs,
    }


def encoded_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    payload_fn: Any,
) -> tuple[list[str], np.ndarray]:
    payloads = [payload_fn(row, feature_columns=feature_columns) for row in rows]
    encoded_values: dict[str, list[float]] = {}
    for feature in feature_columns:
        values = [payload.get(feature) for payload in payloads]
        if all(float_or_none(value) is not None for value in values):
            encoded_values[feature] = [float(float_or_none(value) or 0.0) for value in values]
        else:
            categories = sorted({str(value or "UNKNOWN") for value in values})
            for category in categories:
                encoded_values[f"{feature}={category}"] = [1.0 if str(value or "UNKNOWN") == category else 0.0 for value in values]
    names = list(encoded_values)
    matrix = np.array([encoded_values[name] for name in names], dtype=float).T if names else np.zeros((len(rows), 0))
    return names, matrix


def high_correlation_pairs(names: Sequence[str], corr: np.ndarray) -> list[dict[str, Any]]:
    pairs = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            value = float(corr[i, j])
            abs_value = abs(value)
            if abs_value > 0.80:
                right = names[j]
                pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": round(value, 6),
                        "abs_correlation": round(abs_value, 6),
                        "comment": correlation_comment(left, right, abs_value),
                    }
                )
    return sorted(pairs, key=lambda item: (-item["abs_correlation"], item["feature_a"], item["feature_b"]))


def watched_pair_findings(names: Sequence[str], corr: np.ndarray) -> list[dict[str, Any]]:
    index = {name: pos for pos, name in enumerate(names)}
    findings = []
    for left, right in sorted(WATCHED_PAIRS):
        if left in index and right in index:
            value = float(corr[index[left], index[right]])
            findings.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": round(value, 6),
                    "abs_correlation": round(abs(value), 6),
                    "comment": correlation_comment(left, right, abs(value)),
                }
            )
        else:
            findings.append({"feature_a": left, "feature_b": right, "status": "not_both_present"})
    return findings


def correlation_comment(left: str, right: str, abs_value: float) -> str:
    base_pair = tuple(sorted((left.split("=")[0], right.split("=")[0])))
    if abs_value > 0.95 or base_pair in {
        tuple(sorted(("features.equity_table", "features.equity_1v1"))),
        tuple(sorted(("features.hero_stack", "features.effective_stack"))),
        tuple(sorted(("features.to_call", "features.equity_required"))),
    }:
        return "suspect_redundant"
    if abs_value > 0.90:
        return "redondant"
    return "acceptable_monitor"


def write_correlation_svg(report: Mapping[str, Any], output_path: Path) -> None:
    labels = list(report["encoded_feature_order"])
    matrix = report["matrix"]
    cell = 24
    left = 260
    top = 165
    width = left + cell * len(labels) + 30
    height = top + cell * len(labels) + 30
    lines = [
        svg_header(width, height),
        '<text x="20" y="30" font-size="18" font-family="Arial">V5_4000 input feature correlation matrix</text>',
        '<text x="20" y="54" font-size="12" font-family="Arial" fill="#555">Pearson correlation on encoded features actually used by the model.</text>',
    ]
    for idx, label in enumerate(labels):
        x = left + idx * cell + cell / 2
        y = top + idx * cell + cell / 2
        short = short_label(label)
        lines.append(f'<text x="{x}" y="{top - 8}" text-anchor="start" font-size="9" font-family="Arial" transform="rotate(-55 {x} {top - 8})">{escape_xml(short)}</text>')
        lines.append(f'<text x="{left - 8}" y="{y + 3}" text-anchor="end" font-size="9" font-family="Arial">{escape_xml(short)}</text>')
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            y = top + row_index * cell
            fill = diverging_color(float(value))
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="0.5"/>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def diverging_color(value: float) -> str:
    safe = max(-1.0, min(1.0, value))
    if safe >= 0:
        ratio = safe
        red = int(245 - 35 * ratio)
        green = int(247 - 130 * ratio)
        blue = int(250 - 175 * ratio)
    else:
        ratio = abs(safe)
        red = int(245 - 170 * ratio)
        green = int(247 - 95 * ratio)
        blue = int(250 - 20 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def render_high_correlation_pairs(correlation: Mapping[str, Any]) -> str:
    report = correlation["json"]
    lines = [
        "# High Correlation Pairs",
        "",
        "Only encoded features actually used by the model are included.",
        "",
        "## Absolute Correlation > 0.90",
        "",
        "| feature A | feature B | correlation | comment |",
        "|---|---|---:|---|",
    ]
    for pair in report["high_correlation_pairs_abs_gt_0_90"]:
        lines.append(correlation_pair_row(pair))
    lines.extend(["", "## Absolute Correlation > 0.80", "", "| feature A | feature B | correlation | comment |", "|---|---|---:|---|"])
    for pair in report["high_correlation_pairs_abs_gt_0_80"]:
        lines.append(correlation_pair_row(pair))
    lines.extend(["", "## Watched Pairs", "", "| feature A | feature B | correlation | comment |", "|---|---|---:|---|"])
    for pair in report["watched_pairs"]:
        if pair.get("status"):
            lines.append(f"| {md(pair['feature_a'])} | {md(pair['feature_b'])} | NA | `{pair['status']}` |")
        else:
            lines.append(correlation_pair_row(pair))
    lines.append("")
    return "\n".join(lines)


def correlation_pair_row(pair: Mapping[str, Any]) -> str:
    return f"| {md(pair['feature_a'])} | {md(pair['feature_b'])} | {pair['correlation']:.6f} | `{pair['comment']}` |"


def build_multicollinearity_report(correlation: Mapping[str, Any]) -> dict[str, Any]:
    encoded_names = list(correlation["json"]["encoded_feature_order"])
    corr = np.array(correlation["json"]["matrix"], dtype=float)
    vif = variance_inflation_factors(encoded_names, corr)
    strong_pairs = correlation["json"]["high_correlation_pairs_abs_gt_0_80"]
    redundant_candidates = sorted(
        {
            item["feature_b"]
            for item in strong_pairs
            if item["comment"] in {"suspect_redundant", "redondant"}
        }
    )
    recommendations = {
        feature: feature_recommendation(feature, vif.get(feature), strong_pairs)
        for feature in encoded_names
    }
    return {
        "schema": "bootstrap_v5_multicollinearity_report",
        "encoded_feature_order": encoded_names,
        "strong_correlation_pairs": strong_pairs,
        "vif": {feature: format_json_number(value) for feature, value in vif.items()},
        "redundant_candidate_features": redundant_candidates,
        "recommendations": recommendations,
    }


def variance_inflation_factors(names: Sequence[str], corr: np.ndarray) -> dict[str, float]:
    if len(names) == 0:
        return {}
    try:
        inverse = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(corr)
    vifs = {}
    for index, name in enumerate(names):
        if any(index != other and abs(float(corr[index, other])) >= 0.999999 for other in range(len(names))):
            vifs[name] = math.inf
        else:
            vifs[name] = max(1.0, float(inverse[index, index]))
    return vifs


def feature_recommendation(feature: str, vif: float | None, strong_pairs: Sequence[Mapping[str, Any]]) -> str:
    related = [pair for pair in strong_pairs if feature in {pair["feature_a"], pair["feature_b"]}]
    if feature.endswith("=IP") or feature.endswith("=OOP"):
        return "fusionner"
    if vif is math.inf or (isinstance(vif, float) and vif >= 10):
        return "supprimer"
    if any(pair["comment"] == "suspect_redundant" for pair in related):
        return "fusionner"
    if related:
        return "garder"
    return "garder"


def render_multicollinearity_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multicollinearity Report",
        "",
        "## Strong Correlations",
        "",
        "| feature A | feature B | correlation | comment |",
        "|---|---|---:|---|",
    ]
    for pair in report["strong_correlation_pairs"]:
        lines.append(correlation_pair_row(pair))
    lines.extend(["", "## Variance Inflation Factor", "", "| feature | VIF | recommendation |", "|---|---:|---|"])
    for feature, vif in report["vif"].items():
        lines.append(f"| {md(feature)} | {vif} | `{report['recommendations'].get(feature, 'garder')}` |")
    lines.extend(["", "## Redundant Candidate Features", ""])
    lines.extend(f"- {md(feature)}" for feature in report["redundant_candidate_features"])
    lines.append("")
    return "\n".join(lines)


def build_learning_curve_report(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    model_name: str,
    random_seed: int,
    payload_fn: Any = feature_payload,
) -> dict[str, Any]:
    train_rows, validation_rows, split_warning = stratified_split(rows, random_seed=random_seed)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    points = []
    for fraction in fractions:
        subset = stratified_sample(train_rows, fraction=fraction, random_seed=random_seed)
        if len({str(row["bootstrap_label"]) for row in subset}) < 2:
            continue
        model = make_model(model_name if model_name in {"dummy", "logistic_regression", "random_forest", "extra_trees"} else "random_forest")
        x_train = [payload_fn(row, feature_columns=feature_columns) for row in subset]
        y_train = [str(row["bootstrap_label"]) for row in subset]
        x_validation = [payload_fn(row, feature_columns=feature_columns) for row in validation_rows]
        y_validation = [str(row["bootstrap_label"]) for row in validation_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        train_predictions = [str(value) for value in model.predict(x_train)]
        validation_predictions = [str(value) for value in model.predict(x_validation)]
        train_macro = f1_score(y_train, train_predictions, average="macro", zero_division=0)
        validation_macro = f1_score(y_validation, validation_predictions, average="macro", zero_division=0)
        points.append(
            {
                "train_size": len(subset),
                "train_fraction": fraction,
                "train_accuracy": round(float(accuracy_score(y_train, train_predictions)), 6),
                "validation_accuracy": round(float(accuracy_score(y_validation, validation_predictions)), 6),
                "train_macro_f1": round(float(train_macro), 6),
                "validation_macro_f1": round(float(validation_macro), 6),
                "macro_f1_gap": round(float(train_macro - validation_macro), 6),
            }
        )
    return {
        "schema": "bootstrap_v5_learning_curve_report",
        "model_name": model_name,
        "row_count": len(rows),
        "validation_size": len(validation_rows),
        "feature_columns": list(feature_columns),
        "split_warning": split_warning,
        "points": points,
        "diagnostic": learning_curve_diagnostic(points),
    }


def stratified_sample(rows: Sequence[Mapping[str, Any]], *, fraction: float, random_seed: int) -> list[dict[str, Any]]:
    if fraction >= 1.0:
        return [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["bootstrap_label"]), []).append(row)
    rng = np.random.default_rng(random_seed + int(fraction * 1000))
    for label_rows in by_label.values():
        count = max(1, int(round(len(label_rows) * fraction)))
        indexes = rng.choice(len(label_rows), size=min(count, len(label_rows)), replace=False)
        selected.extend(dict(label_rows[int(index)]) for index in indexes)
    return selected


def learning_curve_diagnostic(points: Sequence[Mapping[str, Any]]) -> str:
    if not points:
        return "insufficient_data"
    last = points[-1]
    train = float(last.get("train_macro_f1") or 0.0)
    validation = float(last.get("validation_macro_f1") or 0.0)
    gap = train - validation
    if train > 0.95 and gap > 0.08:
        return "overfit_probable"
    if train < 0.65 and validation < 0.65:
        return "underfit_probable"
    if train > 0.98 and validation > 0.98:
        return "dataset_too_simple_or_leakage_possible"
    return "acceptable"


def write_learning_curve_svg(report: Mapping[str, Any], output_path: Path) -> None:
    points = list(report.get("points", []))
    width = 760
    height = 360
    left = 70
    top = 45
    plot_w = 610
    plot_h = 245
    lines = [
        svg_header(width, height),
        '<text x="20" y="28" font-size="18" font-family="Arial">V5_4000 learning curve</text>',
        f'<text x="20" y="348" font-size="12" font-family="Arial" fill="#555">Diagnostic: {escape_xml(str(report.get("diagnostic")))}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cccccc"/>',
    ]
    for tick in range(6):
        y = top + plot_h - tick * plot_h / 5
        label = tick / 5
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-size="10" font-family="Arial">{label:.1f}</text>')
    if points:
        xs = [point["train_size"] for point in points]
        min_x = min(xs)
        max_x = max(xs)
        draw_polyline(lines, points, "train_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#2957a4")
        draw_polyline(lines, points, "validation_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#c44733")
        for point in points:
            x = scale_x(point["train_size"], min_x, max_x, left, plot_w)
            lines.append(f'<text x="{x}" y="{top + plot_h + 18}" text-anchor="middle" font-size="9" font-family="Arial">{point["train_size"]}</text>')
    lines.append('<circle cx="705" cy="62" r="5" fill="#2957a4"/><text x="716" y="66" font-size="11" font-family="Arial">train macro F1</text>')
    lines.append('<circle cx="705" cy="82" r="5" fill="#c44733"/><text x="716" y="86" font-size="11" font-family="Arial">validation macro F1</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def draw_polyline(lines: list[str], points: Sequence[Mapping[str, Any]], key: str, min_x: int, max_x: int, left: int, top: int, plot_w: int, plot_h: int, color: str) -> None:
    coords = []
    for point in points:
        x = scale_x(point["train_size"], min_x, max_x, left, plot_w)
        y = top + plot_h - max(0.0, min(1.0, float(point[key]))) * plot_h
        coords.append(f"{x},{y}")
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')


def scale_x(value: int, min_x: int, max_x: int, left: int, plot_w: int) -> float:
    if max_x == min_x:
        return left + plot_w / 2
    return left + (value - min_x) / (max_x - min_x) * plot_w


def render_learning_curve_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Learning Curve Report",
        "",
        f"Diagnostic: `{report.get('diagnostic')}`",
        "",
        "| train size | train accuracy | validation accuracy | train macro F1 | validation macro F1 | gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in report.get("points", []):
        lines.append(
            f"| {point['train_size']} | {point['train_accuracy']} | {point['validation_accuracy']} | "
            f"{point['train_macro_f1']} | {point['validation_macro_f1']} | {point['macro_f1_gap']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_robust_generalization_report(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    random_seed: int,
    payload_fn: Any = feature_payload,
    equity_features: set[str] = EQUITY_DERIVED_FEATURES,
    pot_stack_features: set[str] = POT_STACK_ACTION_POSITION_FEATURES,
) -> dict[str, Any]:
    variants = [
        ("random_split", rows, None, list(feature_columns)),
        ("solver_candidate_only", source_rows(rows, "solver_candidate"), None, list(feature_columns)),
        ("weak_rule_bootstrap_only", source_rows(rows, "weak_rule_bootstrap"), None, list(feature_columns)),
        ("equity_derived_features_only", rows, None, [feature for feature in feature_columns if feature in equity_features]),
        ("no_equity_derived_features", rows, None, [feature for feature in feature_columns if feature not in equity_features]),
        ("pot_stack_actions_position_only", rows, None, [feature for feature in feature_columns if feature in pot_stack_features]),
    ]
    results = [
        evaluate_variant(
            name=name,
            rows=variant_rows,
            fixed_split=fixed_split,
            feature_columns=variant_features,
            random_seed=random_seed,
            payload_fn=payload_fn,
        )
        for name, variant_rows, fixed_split, variant_features in variants
    ]
    if source_rows(rows, "solver_candidate") and source_rows(rows, "weak_rule_bootstrap"):
        results.append(
            evaluate_variant(
                name="train_solver_candidate_validate_weak_rule_bootstrap",
                rows=[],
                fixed_split=(source_rows(rows, "solver_candidate"), source_rows(rows, "weak_rule_bootstrap")),
                feature_columns=list(feature_columns),
                random_seed=random_seed,
                payload_fn=payload_fn,
            )
        )
        results.append(
            evaluate_variant(
                name="train_weak_rule_bootstrap_validate_solver_candidate",
                rows=[],
                fixed_split=(source_rows(rows, "weak_rule_bootstrap"), source_rows(rows, "solver_candidate")),
                feature_columns=list(feature_columns),
                random_seed=random_seed,
                payload_fn=payload_fn,
            )
        )
    return {
        "schema": "bootstrap_v5_robust_generalization_report",
        "row_count": len(rows),
        "feature_columns": list(feature_columns),
        "variants": results,
    }


def evaluate_variant(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fixed_split: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None,
    feature_columns: Sequence[str],
    random_seed: int,
    payload_fn: Any,
) -> dict[str, Any]:
    if not feature_columns:
        return {"name": name, "status": "skipped", "reason": "no_features", "rows_used": len(rows), "feature_count": 0}
    if fixed_split is None:
        if len(rows) < 4 or len({str(row["bootstrap_label"]) for row in rows}) < 2:
            return {"name": name, "status": "skipped", "reason": "not_enough_rows_or_classes", "rows_used": len(rows), "feature_count": len(feature_columns)}
        train_rows, test_rows, split_warning = stratified_split(rows, random_seed=random_seed)
    else:
        train_rows = [dict(row) for row in fixed_split[0]]
        test_rows = [dict(row) for row in fixed_split[1]]
        split_warning = None
        if len(train_rows) < 2 or len(test_rows) < 1 or len({str(row["bootstrap_label"]) for row in train_rows}) < 2:
            return {"name": name, "status": "skipped", "reason": "fixed_split_not_trainable", "rows_used": len(train_rows) + len(test_rows), "feature_count": len(feature_columns)}
    labels = sorted({str(row["bootstrap_label"]) for row in list(train_rows) + list(test_rows)})
    comparisons = {}
    for model_name in candidate_model_names("auto"):
        comparisons[model_name] = fit_and_score(model_name, train_rows, test_rows, labels, feature_columns, payload_fn)
    best_name = select_best_model(comparisons, requested_model_type="auto")
    best = comparisons[best_name]
    return {
        "name": name,
        "status": "ok",
        "rows_used": len(train_rows) + len(test_rows),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "label_distribution": dict(sorted(Counter(str(row["bootstrap_label"]) for row in list(train_rows) + list(test_rows)).items())),
        "selected_model": best_name,
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "split_warning": split_warning,
        "comment": generalization_comment(best["macro_f1"], best["accuracy"]),
    }


def fit_and_score(
    model_name: str,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    feature_columns: Sequence[str],
    payload_fn: Any,
) -> dict[str, Any]:
    model = make_model(model_name)
    x_train = [payload_fn(row, feature_columns=feature_columns) for row in train_rows]
    y_train = [str(row["bootstrap_label"]) for row in train_rows]
    x_test = [payload_fn(row, feature_columns=feature_columns) for row in test_rows]
    y_test = [str(row["bootstrap_label"]) for row in test_rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    predictions = [str(value) for value in model.predict(x_test)]
    return {
        "model_name": model_name,
        "model": model,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="weighted", zero_division=0)), 6),
        "confusion_matrix": matrix_as_dict(y_test, predictions, labels),
    }


def source_rows(rows: Sequence[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row_label_source(row) == source]


def row_label_source(row: Mapping[str, Any]) -> str:
    return str(row.get("label_source") or row.get("audit.label_source") or row.get("metadata.label_source") or "")


def generalization_comment(macro_f1: float, accuracy: float) -> str:
    if macro_f1 >= 0.98 and accuracy >= 0.98:
        return "very_high_score_check_for_synthetic_simplicity_or_leakage"
    if macro_f1 < 0.65:
        return "weak_generalization"
    return "acceptable"


def render_robust_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Robust Generalization Report",
        "",
        "| variant | status | rows | features | selected model | accuracy | macro F1 | comment |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for variant in report.get("variants", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    md(variant.get("name")),
                    md(variant.get("status")),
                    str(variant.get("rows_used", 0)),
                    str(variant.get("feature_count", 0)),
                    md(variant.get("selected_model", "")),
                    format_value(variant.get("accuracy")),
                    format_value(variant.get("macro_f1")),
                    md(variant.get("comment") or variant.get("reason", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def export_bb_dataset(*, input_path: Path, output_csv: Path) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    added = [
        "features.big_blind",
        "features.pot_bb",
        "features.to_call_bb",
        "features.hero_stack_bb",
        "features.effective_stack_bb",
        "features.call_max_bb",
        "features.ev_bb",
        "features.effective_stack_bb_to_pot_bb",
        "big_blind_missing_or_inferred",
        "audit.big_blind_inference_rule",
    ]
    output_fields = fieldnames + [field for field in added if field not in fieldnames]
    warnings_seen = Counter()
    exported = []
    for row in rows:
        normalized = dict(row)
        big_blind, warning, rule = infer_big_blind(normalized)
        warnings_seen[warning] += 1
        normalized["features.big_blind"] = big_blind
        normalized["big_blind_missing_or_inferred"] = warning
        normalized["audit.big_blind_inference_rule"] = rule
        for source, target in [
            ("features.pot", "features.pot_bb"),
            ("features.to_call", "features.to_call_bb"),
            ("features.hero_stack", "features.hero_stack_bb"),
            ("features.effective_stack", "features.effective_stack_bb"),
            ("features.call_max", "features.call_max_bb"),
            ("features.ev", "features.ev_bb"),
        ]:
            normalized[target] = divide_or_empty(normalized.get(source), big_blind)
        pot_bb = float_or_none(normalized.get("features.pot_bb"))
        effective_stack_bb = float_or_none(normalized.get("features.effective_stack_bb"))
        normalized["features.effective_stack_bb_to_pot_bb"] = (
            effective_stack_bb / pot_bb if pot_bb and pot_bb > 0 and effective_stack_bb is not None else ""
        )
        exported.append(normalized)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(exported)
    return {
        "status": "ok",
        "output_csv": str(output_csv),
        "rows_total": len(exported),
        "big_blind_warning_counts": dict(sorted(warnings_seen.items())),
        "raw_amount_columns_kept_for_audit": list(RAW_AMOUNT_COLUMNS),
        "normalization_rule": "big_blind inferred as 1.0 from offline solver bootstrap convention units=bb when no explicit blind column exists",
    }


def infer_big_blind(row: Mapping[str, Any]) -> tuple[float, str, str]:
    for field in ("features.big_blind", "big_blind", "bb", "metadata.big_blind"):
        value = float_or_none(row.get(field))
        if value is not None and value > 0:
            return value, "explicit_big_blind_available", f"read:{field}"
    return 1.0, "big_blind_missing_or_inferred", "inferred:offline_solver_bootstrap_jobs_are_declared_units_bb"


def divide_or_empty(value: Any, denominator: float) -> float | str:
    number = float_or_none(value)
    if number is None or denominator <= 0:
        return ""
    return number / denominator


def train_bb_model(*, input_csv: Path, output_dir: Path, random_seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, fieldnames = load_candidate_csv(input_csv)
    train_rows = select_training_rows(rows)
    feature_columns = list(BB_FEATURE_COLUMNS)
    labels = sorted({str(row["bootstrap_label"]) for row in train_rows})
    train_split, test_split, split_warning = stratified_split(train_rows, random_seed=random_seed)
    comparisons = {
        name: fit_and_score(name, train_split, test_split, labels, feature_columns, bb_feature_payload)
        for name in candidate_model_names("auto")
    }
    best_name = select_best_model(comparisons, requested_model_type="auto")
    best = comparisons[best_name]
    model = best["model"]
    feature_schema = build_bb_feature_schema(train_rows, labels, feature_columns)
    label_mapping = build_label_mapping(labels)
    paths = {
        "model": str(output_dir / "model.joblib"),
        "preprocessing": str(output_dir / "preprocessing.joblib"),
        "feature_schema": str(output_dir / "feature_schema.json"),
        "label_mapping": str(output_dir / "label_mapping.json"),
    }
    joblib.dump(model, paths["model"])
    joblib.dump({"feature_schema": feature_schema, "label_mapping": label_mapping}, paths["preprocessing"])
    write_json(feature_schema, paths["feature_schema"])
    write_json(label_mapping, paths["label_mapping"])
    report = {
        "status": "ok",
        "model_type": best_name,
        "selected_model": best_name,
        "input_path": str(input_csv),
        "rows_total": len(rows),
        "rows_used": len(train_rows),
        "train_size": len(train_split),
        "test_size": len(test_split),
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in train_rows).items())),
        "model_feature_columns": feature_columns,
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "model_comparison": without_model_objects({"model_comparison": comparisons})["model_comparison"],
        "split_warning": split_warning,
        "output_files": paths,
        "not_for_production": True,
        "bot_live_connection": "not_modified",
        "call_added": False,
    }
    write_json(report, output_dir / "training_report.json")
    write_json(build_bb_feature_contract(fieldnames, feature_columns), output_dir / "feature_contract.json")
    write_json(build_bb_preprocessing_schema(train_rows, feature_columns), output_dir / "preprocessing_schema.json")
    return report


def bb_feature_payload(row: Mapping[str, Any], *, feature_columns: Sequence[str] = BB_FEATURE_COLUMNS) -> dict[str, Any]:
    payload = {}
    for feature in feature_columns:
        if feature in BB_NUMERIC_FEATURES:
            payload[feature] = float_or_none(row.get(feature)) or 0.0
        elif feature in BB_CATEGORICAL_FEATURES:
            payload[feature] = str(row.get(feature) or "UNKNOWN")
        else:
            raise ValueError(f"unknown_bb_feature:{feature}")
    return payload


def build_bb_feature_schema(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], feature_columns: Sequence[str]) -> dict[str, Any]:
    categorical = [feature for feature in feature_columns if feature in BB_CATEGORICAL_FEATURES]
    return {
        "training_quality": "pipeline_smoke_only_bb_experimental",
        "feature_order": list(feature_columns),
        "numeric_features": [feature for feature in feature_columns if feature in BB_NUMERIC_FEATURES],
        "categorical_features": categorical,
        "card_features": [],
        "categorical_values": {feature: sorted({str(row.get(feature) or "UNKNOWN") for row in rows}) for feature in categorical},
        "target": "bootstrap_label",
        "labels": list(labels),
        "not_for_production": True,
        "bot_live_connection": "not_modified",
    }


def build_bb_feature_contract(fieldnames: Sequence[str], feature_columns: Sequence[str]) -> dict[str, Any]:
    leakage = sorted(column for column in fieldnames if is_leakage_column(column))
    raw_audit = [column for column in RAW_AMOUNT_COLUMNS if column in fieldnames]
    return {
        "schema": "bootstrap_v5_4000_bb_experimental",
        "features_model_used": list(feature_columns),
        "feature_order": list(feature_columns),
        "features_audit_only": sorted(set(raw_audit + [column for column in fieldnames if column not in feature_columns and column not in {"bootstrap_label", "excluded", "exclusion_reason"}])),
        "features_raw_amount_audit_only": raw_audit,
        "features_leakage_excluded": leakage,
        "leakage_columns_used_by_model": sorted(column for column in leakage if column in feature_columns),
        "features_constant_excluded": [],
        "features_not_available_yet": [],
        "allowed_predictions": ["CHECK", "FOLD", "RAISE"],
        "not_allowed_yet": ["CALL"],
        "big_blind_policy": "explicit if present, otherwise inferred as 1.0 from solver bootstrap units=bb with warning column",
        "not_for_production": True,
    }


def build_bb_preprocessing_schema(rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> dict[str, Any]:
    categorical = [feature for feature in feature_columns if feature in BB_CATEGORICAL_FEATURES]
    return {
        "feature_order": list(feature_columns),
        "dict_vectorizer_input": "bb_ordered_feature_payload",
        "null_policy": {"numeric": "missing_or_invalid_to_0.0", "categorical": "missing_to_UNKNOWN"},
        "categorical_values": {feature: sorted({str(row.get(feature) or "UNKNOWN") for row in rows}) for feature in categorical},
    }


def build_current_vs_bb_comparison(
    *,
    current_report: Mapping[str, Any],
    bb_report: Mapping[str, Any],
    current_learning_curve: Mapping[str, Any],
    bb_learning_curve: Mapping[str, Any],
    current_correlation: Mapping[str, Any],
    bb_correlation: Mapping[str, Any],
    current_robust: Mapping[str, Any],
    bb_robust: Mapping[str, Any],
    current_coefficients: Mapping[str, Any],
    bb_coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "bootstrap_v5_current_vs_bb_comparison",
        "current_v5_4000": model_comparison_summary(current_report, current_learning_curve, current_correlation, current_robust, current_coefficients),
        "bb_normalized_experimental": model_comparison_summary(bb_report, bb_learning_curve, bb_correlation, bb_robust, bb_coefficients),
        "delta": {
            "accuracy": round(float(bb_report.get("accuracy", 0.0)) - float(current_report.get("accuracy", 0.0)), 6),
            "macro_f1": round(float(bb_report.get("macro_f1", 0.0)) - float(current_report.get("macro_f1", 0.0)), 6),
            "feature_count": len(bb_report.get("model_feature_columns", [])) - len(current_report.get("model_feature_columns", [])),
        },
        "recommendation": comparison_recommendation(current_report, bb_report, current_learning_curve, bb_learning_curve),
        "not_for_production": True,
        "bot_live_connection": "not_modified",
        "call_added": False,
    }


def model_comparison_summary(
    report: Mapping[str, Any],
    learning_curve: Mapping[str, Any],
    correlation: Mapping[str, Any],
    robust: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": report.get("model_type") or report.get("selected_model"),
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro_f1"),
        "confusion_matrix": report.get("confusion_matrix"),
        "feature_count": len(report.get("model_feature_columns", [])),
        "features": report.get("model_feature_columns", []),
        "learning_curve_diagnostic": learning_curve.get("diagnostic"),
        "learning_curve_last_point": (learning_curve.get("points") or [{}])[-1],
        "high_correlation_pair_count_abs_gt_0_90": len(correlation.get("high_correlation_pairs_abs_gt_0_90", [])),
        "robust_variants": [
            {
                "name": variant.get("name"),
                "status": variant.get("status"),
                "accuracy": variant.get("accuracy"),
                "macro_f1": variant.get("macro_f1"),
                "comment": variant.get("comment") or variant.get("reason"),
            }
            for variant in robust.get("variants", [])
        ],
        "logistic_top_coefficients": coefficients,
    }


def comparison_recommendation(
    current_report: Mapping[str, Any],
    bb_report: Mapping[str, Any],
    current_learning_curve: Mapping[str, Any],
    bb_learning_curve: Mapping[str, Any],
) -> str:
    delta = float(bb_report.get("macro_f1", 0.0)) - float(current_report.get("macro_f1", 0.0))
    if bb_learning_curve.get("diagnostic") == "overfit_probable":
        return "keep_current_for_now_bb_overfit_needs_review"
    if delta >= -0.02:
        return "bb_model_is_viable_experimental_candidate"
    return "keep_current_for_now_bb_score_drop"


def render_current_vs_bb_comparison(comparison: Mapping[str, Any]) -> str:
    current = comparison["current_v5_4000"]
    bb = comparison["bb_normalized_experimental"]
    lines = [
        "# Current vs BB-Normalized Comparison",
        "",
        f"Recommendation: `{comparison['recommendation']}`",
        "",
        "| model | accuracy | macro F1 | features | learning diagnostic | high corr pairs > .90 |",
        "|---|---:|---:|---:|---|---:|",
        f"| current_v5_4000 | {format_value(current.get('accuracy'))} | {format_value(current.get('macro_f1'))} | {current.get('feature_count')} | `{current.get('learning_curve_diagnostic')}` | {current.get('high_correlation_pair_count_abs_gt_0_90')} |",
        f"| bb_normalized_experimental | {format_value(bb.get('accuracy'))} | {format_value(bb.get('macro_f1'))} | {bb.get('feature_count')} | `{bb.get('learning_curve_diagnostic')}` | {bb.get('high_correlation_pair_count_abs_gt_0_90')} |",
        "",
        "## Delta",
        "",
        f"- accuracy: `{comparison['delta']['accuracy']}`",
        f"- macro_f1: `{comparison['delta']['macro_f1']}`",
        f"- feature_count: `{comparison['delta']['feature_count']}`",
        "",
        "## Robustness Snapshot",
        "",
        "| model | variant | status | macro F1 | comment |",
        "|---|---|---|---:|---|",
    ]
    for model_name, model in [("current_v5_4000", current), ("bb_normalized_experimental", bb)]:
        for variant in model.get("robust_variants", []):
            lines.append(
                f"| `{model_name}` | {md(variant.get('name'))} | {md(variant.get('status'))} | "
                f"{format_value(variant.get('macro_f1'))} | {md(variant.get('comment', ''))} |"
            )
    lines.extend(["", "## Coefficient Snapshot", ""])
    for model_name, model in [("current_v5_4000", current), ("bb_normalized_experimental", bb)]:
        lines.extend([f"### {model_name}", "", "| class | top positive coefficient | top negative coefficient |", "|---|---|---|"])
        for label, payload in dict(model.get("logistic_top_coefficients", {})).items():
            positive = payload.get("top_positive", [{}])[0]
            negative = payload.get("top_negative", [{}])[0]
            lines.append(
                f"| `{label}` | {md(positive.get('feature', ''))} `{format_value(positive.get('coefficient'))}` | "
                f"{md(negative.get('feature', ''))} `{format_value(negative.get('coefficient'))}` |"
            )
        lines.append("")
    lines.extend(["", "The BB model is experimental and saved in a separate folder. The current model artifact was not overwritten.", ""])
    return "\n".join(lines)


def logistic_top_coefficients(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    payload_fn: Any,
) -> dict[str, Any]:
    model = make_model("logistic_regression")
    x = [payload_fn(row, feature_columns=feature_columns) for row in rows]
    y = [str(row["bootstrap_label"]) for row in rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    result = {}
    for class_index, label in enumerate(classifier.classes_):
        weights = [
            {"feature": names[index], "coefficient": round(float(classifier.coef_[class_index][index]), 6)}
            for index in range(len(names))
        ]
        result[str(label)] = {
            "top_positive": sorted(weights, key=lambda item: item["coefficient"], reverse=True)[:8],
            "top_negative": sorted(weights, key=lambda item: item["coefficient"])[:8],
        }
    return result


def short_label(value: str, *, limit: int = 34) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_json_number(value: float) -> float | str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return round(float(value), 6)


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--current-model-dir", default=str(DEFAULT_CURRENT_MODEL_DIR))
    parser.add_argument("--bb-dataset", default=str(DEFAULT_BB_DATASET))
    parser.add_argument("--bb-model-dir", default=str(DEFAULT_BB_MODEL_DIR))
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_advanced_diagnostics(
        input_csv=args.input,
        current_model_dir=args.current_model_dir,
        bb_dataset_csv=args.bb_dataset,
        bb_model_dir=args.bb_model_dir,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
