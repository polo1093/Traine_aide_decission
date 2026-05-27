"""Train a three-intent PokerBench oracle model.

This experiment keeps the PokerBench four-class oracle baseline intact, then
maps CHECK/FOLD into a shared NO_INVEST intent:

CHECK -> NO_INVEST
FOLD  -> NO_INVEST
CALL  -> CALL
RAISE -> RAISE
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pokerbench_oracle_baseline_v1 import (
    AUDIT_ONLY_COLUMNS,
    DEFAULT_DATA_DIR,
    FEATURE_COLUMNS,
    LIVE_BB_MODEL,
    NUMERIC_FEATURES,
    build_candidates as build_four_class_candidates,
    deterministic_sample,
    feature_payload,
    file_sha256,
    is_leakage_column,
    load_pokerbench_rows,
    make_model,
    number_or_zero,
    prepare_pokerbench_sources,
    preflop_postflop_distribution,
    read_json,
    safe_corrcoef,
    write_bar_chart,
    write_feature_importance_svg,
    write_json,
    svg_header,
    svg_text,
    blue_scale,
    diverging_color,
    short_label,
)


DEFAULT_OUTPUT_DIR = Path("outputs/readiness/pokerbench_oracle_3intent_v1")
FOUR_CLASS_BASELINE_REPORT = Path("outputs/readiness/pokerbench_oracle_baseline_v1/training_report.json")
INTENT_LABELS = ("NO_INVEST", "CALL", "RAISE")
INTENT_MAPPING = {
    "CHECK": "NO_INVEST",
    "FOLD": "NO_INVEST",
    "CALL": "CALL",
    "RAISE": "RAISE",
}


def run_pokerbench_oracle_3intent_v1(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    download: bool = True,
    max_rows: int | None = None,
    random_seed: int = 17,
    fast_mode: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    live_hash_before = file_sha256(LIVE_BB_MODEL)

    source_paths = prepare_pokerbench_sources(data_path, download=download)
    loaded_rows = load_pokerbench_rows(source_paths, max_rows=max_rows)
    four_class_candidates, parse_report = build_four_class_candidates(loaded_rows)
    candidates = map_candidates_to_intents(four_class_candidates)
    if len(candidates) < 20:
        raise ValueError(f"not_enough_usable_pokerbench_rows:{len(candidates)}")

    candidates_csv = output_path / "candidates.csv"
    write_candidates_csv(candidates, candidates_csv)
    training = train_intent_model(candidates, output_path=output_path, random_seed=random_seed)
    prediction = predict_first_row(output_path, candidates[0])
    comparison = build_comparison_with_four_class_baseline(training, rows_used=len(candidates))
    (output_path / "comparison_with_pokerbench_oracle_baseline_v1.md").write_text(comparison, encoding="utf-8")
    graphical = write_graphical_study(
        candidates,
        output_path=output_path,
        training=training,
        random_seed=random_seed,
        fast_mode=fast_mode,
    )
    live_hash_after = file_sha256(LIVE_BB_MODEL)

    report = {
        **training,
        "schema": "pokerbench_oracle_3intent_v1",
        "dataset_source": "RZ412/PokerBench",
        "input_files": [str(path) for path in source_paths],
        "rows_loaded": len(loaded_rows),
        "rows_usable": len(candidates),
        "candidates_csv": str(candidates_csv),
        "label_source": "pokerbench_solver_oracle",
        "intent_mapping": dict(INTENT_MAPPING),
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in candidates).items())),
        "original_label_distribution": dict(sorted(Counter(row["pokerbench.original_four_class_label"] for row in candidates).items())),
        "street_distribution": dict(sorted(Counter(row["metadata.street"] for row in candidates).items())),
        "preflop_postflop_distribution": preflop_postflop_distribution(candidates),
        "unmapped_outputs": parse_report["unmapped_outputs"],
        "unmapped_count": parse_report["unmapped_count"],
        "offline_prediction": prediction,
        "comparison_with_pokerbench_oracle_baseline_v1": "comparison_with_pokerbench_oracle_baseline_v1.md",
        "graphical_study": graphical,
        "live_bb_baseline_v1_overwritten": live_hash_before != live_hash_after,
        "live_bb_baseline_v1_hash_before": live_hash_before,
        "live_bb_baseline_v1_hash_after": live_hash_after,
        "known_limits": [
            "NO_INVEST intentionally merges CHECK and FOLD, then requires legal-action context to resolve the final action.",
            "This model is an offline experiment and does not connect to the live bot.",
            "Raw prompt text is audit-only and not used as a model feature.",
        ],
        "bot_live_connection": "not_modified",
        "not_for_production": True,
    }
    write_json(report, output_path / "training_report.json")
    return report


def map_candidates_to_intents(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        original_label = str(row["bootstrap_label"])
        intent = map_action_to_intent(original_label)
        if intent not in INTENT_LABELS:
            continue
        candidate = dict(row)
        candidate["pokerbench.original_four_class_label"] = original_label
        candidate["bootstrap_label"] = intent
        candidate["metadata.label_source"] = "pokerbench_solver_oracle_3intent"
        candidates.append(candidate)
    return candidates


def map_action_to_intent(action: str) -> str:
    return INTENT_MAPPING.get(str(action).upper(), str(action).upper())


def resolve_no_invest_action(intent: str, *, check_possible: bool) -> str:
    normalized = str(intent).upper()
    if normalized == "NO_INVEST":
        return "CHECK" if check_possible else "FOLD"
    if normalized in {"CALL", "RAISE"}:
        return normalized
    raise ValueError(f"unsupported_intent:{intent}")


def train_intent_model(rows: Sequence[Mapping[str, Any]], *, output_path: Path, random_seed: int) -> dict[str, Any]:
    y = [str(row["bootstrap_label"]) for row in rows]
    train_rows, test_rows = train_test_split(list(rows), test_size=0.20, random_state=random_seed, stratify=y)
    labels = list(INTENT_LABELS)
    comparisons = {
        name: fit_and_score(name, train_rows, test_rows, labels)
        for name in ("logistic_regression", "extra_trees")
    }
    best_name = max(comparisons.items(), key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]))[0]
    best = comparisons[best_name]
    model = best.pop("model")
    joblib.dump(model, output_path / "model.joblib")
    write_json(build_feature_contract(), output_path / "feature_contract.json")
    write_json({"feature_order": list(FEATURE_COLUMNS), "labels": labels}, output_path / "preprocessing_schema.json")
    return {
        "status": "ok",
        "model_type": best_name,
        "model_feature_columns": list(FEATURE_COLUMNS),
        "allowed_predictions": labels,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
        "performance_by_street": best["performance_by_street"],
        "model_comparison": {name: without_model(value) for name, value in comparisons.items()},
    }


def fit_and_score(name: str, train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    model = make_model(name)
    x_train = [feature_payload(row) for row in train_rows]
    y_train = [str(row["bootstrap_label"]) for row in train_rows]
    x_test = [feature_payload(row) for row in test_rows]
    y_test = [str(row["bootstrap_label"]) for row in test_rows]
    model.fit(x_train, y_train)
    predictions = [str(value) for value in model.predict(x_test)]
    return {
        "model": model,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="weighted", zero_division=0)), 6),
        "confusion_matrix": matrix_as_dict(y_test, predictions, labels),
        "classification_report": classification_report(y_test, predictions, labels=list(labels), output_dict=True, zero_division=0),
        "performance_by_street": performance_by_street(test_rows, y_test, predictions, labels),
    }


def performance_by_street(
    rows: Sequence[Mapping[str, Any]],
    truth: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    streets = sorted({str(row.get("metadata.street") or "UNKNOWN") for row in rows})
    for street in streets:
        indexes = [index for index, row in enumerate(rows) if str(row.get("metadata.street") or "UNKNOWN") == street]
        if not indexes:
            continue
        y_true = [truth[index] for index in indexes]
        y_pred = [predictions[index] for index in indexes]
        result[street] = {
            "rows": len(indexes),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)), 6),
        }
    return result


def predict_first_row(model_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    model = joblib.load(model_dir / "model.joblib")
    features = feature_payload(row)
    intent = str(model.predict([features])[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = {
            str(label): round(float(probability), 6)
            for label, probability in zip(model.classes_, model.predict_proba([features])[0])
        }
    check_possible = bool(number_or_zero(row.get("features.has_check")))
    return {
        "status": "ok",
        "prediction": intent,
        "probabilities": probabilities,
        "resolved_action": resolve_no_invest_action(intent, check_possible=check_possible),
        "resolved_action_if_check_possible": resolve_no_invest_action(intent, check_possible=True),
        "resolved_action_if_check_impossible": resolve_no_invest_action(intent, check_possible=False),
    }


def write_graphical_study(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    training: Mapping[str, Any],
    random_seed: int,
    fast_mode: bool,
) -> dict[str, str]:
    label_svg = output_path / "eda_intent_distribution.svg"
    original_label_svg = output_path / "eda_original_label_distribution.svg"
    street_svg = output_path / "eda_street_distribution.svg"
    label_by_street_svg = output_path / "eda_intent_by_street.svg"
    confusion_svg = output_path / "confusion_matrix.svg"
    importance_svg = output_path / "feature_importance.svg"
    correlation_svg = output_path / "feature_correlation.svg"
    learning_svg = output_path / "learning_curve.svg"
    study_md = output_path / "graphical_study.md"

    write_bar_chart(Counter(row["bootstrap_label"] for row in rows), label_svg, title="PokerBench intent distribution")
    write_bar_chart(Counter(row["pokerbench.original_four_class_label"] for row in rows), original_label_svg, title="Original four-class labels")
    write_bar_chart(Counter(row["metadata.street"] for row in rows), street_svg, title="PokerBench street distribution")
    write_label_by_street(rows, label_by_street_svg)
    write_confusion_matrix_svg(training["confusion_matrix"], confusion_svg)
    write_feature_importance_svg(output_path / "model.joblib", importance_svg)
    write_feature_correlation_svg(rows, correlation_svg, sample_size=12000 if fast_mode else 40000, random_seed=random_seed)
    learning_report = build_learning_curve(rows, random_seed=random_seed, fast_mode=fast_mode)
    write_learning_curve_svg(learning_report, learning_svg)
    study_md.write_text(render_graphical_study(training, rows, learning_report), encoding="utf-8")
    return {
        "eda_intent_distribution": str(label_svg),
        "eda_original_label_distribution": str(original_label_svg),
        "eda_street_distribution": str(street_svg),
        "eda_intent_by_street": str(label_by_street_svg),
        "confusion_matrix": str(confusion_svg),
        "feature_importance": str(importance_svg),
        "feature_correlation": str(correlation_svg),
        "learning_curve": str(learning_svg),
        "graphical_study_md": str(study_md),
    }


def write_label_by_street(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    streets = sorted({str(row["metadata.street"]) for row in rows})
    matrix = {
        street: Counter(row["bootstrap_label"] for row in rows if row["metadata.street"] == street)
        for street in streets
    }
    cell = 64
    left = 120
    top = 80
    width = left + cell * len(INTENT_LABELS) + 60
    height = top + cell * len(streets) + 50
    max_value = max([matrix[street][label] for street in streets for label in INTENT_LABELS] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, "Intents by street", size=18)]
    for col, label in enumerate(INTENT_LABELS):
        lines.append(svg_text(left + col * cell + cell / 2, top - 14, label, size=10, anchor="middle"))
    for row_index, street in enumerate(streets):
        lines.append(svg_text(left - 12, top + row_index * cell + cell / 2 + 4, street, size=10, anchor="end"))
        for col, label in enumerate(INTENT_LABELS):
            value = matrix[street][label]
            x = left + col * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{blue_scale(value / max_value if max_value else 0)}" stroke="#ffffff"/>')
            lines.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), size=9, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_matrix_svg(matrix: Mapping[str, Mapping[str, int]], output_path: Path) -> None:
    cell = 78
    left = 135
    top = 80
    width = left + cell * len(INTENT_LABELS) + 80
    height = top + cell * len(INTENT_LABELS) + 70
    max_value = max([int(matrix.get(row, {}).get(col, 0)) for row in INTENT_LABELS for col in INTENT_LABELS] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, "Confusion matrix", size=18)]
    for col, label in enumerate(INTENT_LABELS):
        lines.append(svg_text(left + col * cell + cell / 2, top - 14, f"pred {label}", size=10, anchor="middle"))
    for row_index, label in enumerate(INTENT_LABELS):
        lines.append(svg_text(left - 12, top + row_index * cell + cell / 2 + 4, f"true {label}", size=10, anchor="end"))
        for col, pred in enumerate(INTENT_LABELS):
            value = int(matrix.get(label, {}).get(pred, 0))
            x = left + col * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{blue_scale(value / max_value if max_value else 0)}" stroke="#ffffff"/>')
            lines.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), size=10, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_correlation_svg(rows: Sequence[Mapping[str, Any]], output_path: Path, *, sample_size: int, random_seed: int) -> None:
    sampled = deterministic_sample(rows, sample_size=sample_size, random_seed=random_seed)
    import numpy as np

    matrix = np.array([[number_or_zero(row.get(feature)) for feature in NUMERIC_FEATURES] for row in sampled], dtype=float)
    corr = np.eye(len(NUMERIC_FEATURES)) if matrix.shape[0] < 2 else safe_corrcoef(matrix)
    cell = 28
    left = 265
    top = 180
    width = left + cell * len(NUMERIC_FEATURES) + 40
    height = top + cell * len(NUMERIC_FEATURES) + 40
    lines = [svg_header(width, height), svg_text(20, 30, "Numeric feature correlation", size=18), svg_text(20, 52, f"sample rows: {len(sampled)}", size=11)]
    for index, name in enumerate(NUMERIC_FEATURES):
        x = left + index * cell + cell / 2
        y = top + index * cell + cell / 2
        lines.append(svg_text(x, top - 8, short_label(name.replace("features.", ""), 24), size=8, anchor="start", transform=f"rotate(-55 {x} {top - 8})"))
        lines.append(svg_text(left - 8, y + 3, short_label(name.replace("features.", ""), 28), size=8, anchor="end"))
    for row_index, row in enumerate(corr.tolist()):
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{diverging_color(float(value))}" stroke="#ffffff" stroke-width="0.5"/>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_curve(rows: Sequence[Mapping[str, Any]], *, random_seed: int, fast_mode: bool) -> dict[str, Any]:
    sample_size = 9000 if fast_mode else 24000
    sampled = deterministic_sample(rows, sample_size=sample_size, random_seed=random_seed)
    y = [str(row["bootstrap_label"]) for row in sampled]
    train_rows, test_rows = train_test_split(list(sampled), test_size=0.25, random_state=random_seed, stratify=y)
    points = []
    for fraction in (0.15, 0.3, 0.5, 0.75, 1.0):
        subset = stratified_fraction(train_rows, fraction=fraction, random_seed=random_seed)
        model = make_model("extra_trees")
        x_train = [feature_payload(row) for row in subset]
        y_train = [row["bootstrap_label"] for row in subset]
        x_val = [feature_payload(row) for row in test_rows]
        y_val = [row["bootstrap_label"] for row in test_rows]
        model.fit(x_train, y_train)
        train_pred = [str(value) for value in model.predict(x_train)]
        val_pred = [str(value) for value in model.predict(x_val)]
        points.append(
            {
                "train_size": len(subset),
                "train_macro_f1": round(float(f1_score(y_train, train_pred, labels=list(INTENT_LABELS), average="macro", zero_division=0)), 6),
                "validation_macro_f1": round(float(f1_score(y_val, val_pred, labels=list(INTENT_LABELS), average="macro", zero_division=0)), 6),
                "gap": round(
                    float(
                        f1_score(y_train, train_pred, labels=list(INTENT_LABELS), average="macro", zero_division=0)
                        - f1_score(y_val, val_pred, labels=list(INTENT_LABELS), average="macro", zero_division=0)
                    ),
                    6,
                ),
            }
        )
    return {"points": points, "diagnostic": learning_diagnostic(points), "sample_rows": len(sampled)}


def stratified_fraction(rows: Sequence[Mapping[str, Any]], *, fraction: float, random_seed: int) -> list[Mapping[str, Any]]:
    if fraction >= 1.0:
        return list(rows)
    import numpy as np

    rng = np.random.default_rng(random_seed + int(fraction * 1000))
    selected = []
    for label in INTENT_LABELS:
        label_rows = [row for row in rows if row["bootstrap_label"] == label]
        if not label_rows:
            continue
        count = max(1, int(round(len(label_rows) * fraction)))
        indexes = rng.choice(len(label_rows), size=min(count, len(label_rows)), replace=False)
        selected.extend(label_rows[int(index)] for index in indexes)
    return selected


def write_learning_curve_svg(report: Mapping[str, Any], output_path: Path) -> None:
    points = list(report.get("points") or [])
    width = 760
    height = 360
    left = 70
    top = 45
    plot_w = 610
    plot_h = 245
    lines = [svg_header(width, height), svg_text(20, 28, "Learning curve", size=18), svg_text(20, 348, f"Diagnostic: {report.get('diagnostic')}", size=12)]
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cccccc"/>')
    for tick in range(6):
        y = top + plot_h - tick * plot_h / 5
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#eeeeee"/>')
        lines.append(svg_text(left - 10, y + 4, f"{tick / 5:.1f}", size=10, anchor="end"))
    if points:
        min_x = min(point["train_size"] for point in points)
        max_x = max(point["train_size"] for point in points)
        add_curve(lines, points, "train_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#2957a4")
        add_curve(lines, points, "validation_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#c44733")
    lines.append('<circle cx="700" cy="62" r="5" fill="#2957a4"/><text x="711" y="66" font-size="11" font-family="Arial">train</text>')
    lines.append('<circle cx="700" cy="82" r="5" fill="#c44733"/><text x="711" y="86" font-size="11" font-family="Arial">validation</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def add_curve(
    lines: list[str],
    points: Sequence[Mapping[str, Any]],
    key: str,
    min_x: int,
    max_x: int,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    color: str,
) -> None:
    coords = []
    for point in points:
        x = left + (0.5 * plot_w if max_x == min_x else (point["train_size"] - min_x) / (max_x - min_x) * plot_w)
        y = top + plot_h - max(0.0, min(1.0, float(point[key]))) * plot_h
        coords.append(f"{x:.2f},{y:.2f}")
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')


def render_graphical_study(training: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], learning: Mapping[str, Any]) -> str:
    intent_counts = Counter(row["bootstrap_label"] for row in rows)
    original_counts = Counter(row["pokerbench.original_four_class_label"] for row in rows)
    return "\n".join(
        [
            "# Graphical Study",
            "",
            "PokerBench oracle 3-intent experiment: CHECK and FOLD are merged into NO_INVEST.",
            "",
            "## What The Graphs Show",
            "",
            f"- Intents: `{dict(sorted(intent_counts.items()))}`.",
            f"- Original labels: `{dict(sorted(original_counts.items()))}`.",
            f"- Global accuracy: `{training.get('accuracy')}`, macro F1: `{training.get('macro_f1')}`.",
            f"- Learning diagnostic: `{learning.get('diagnostic')}`.",
            "",
            "## Intent Resolution",
            "",
            "- NO_INVEST + check possible resolves to CHECK.",
            "- NO_INVEST + check impossible resolves to FOLD.",
            "- CALL and RAISE remain direct actions.",
            "",
            "## Guardrails",
            "",
            "- This model is offline only.",
            "- PokerBench remains the solver-oracle label source.",
            "- Raw prompt/cards/history text are audit-only and not used as direct model features.",
            "",
        ]
    )


def learning_diagnostic(points: Sequence[Mapping[str, Any]]) -> str:
    if not points:
        return "insufficient_data"
    last = points[-1]
    train = float(last.get("train_macro_f1") or 0.0)
    validation = float(last.get("validation_macro_f1") or 0.0)
    gap = train - validation
    if train > 0.9 and gap > 0.12:
        return "overfit_probable"
    if train < 0.6 and validation < 0.6:
        return "underfit_probable"
    if validation < 0.75:
        return "dataset_difficult_or_features_underpowered"
    return "correct"


def build_feature_contract() -> dict[str, Any]:
    return {
        "schema": "pokerbench_oracle_3intent_v1",
        "features_model_used": list(FEATURE_COLUMNS),
        "features_audit_only": [*AUDIT_ONLY_COLUMNS, "pokerbench.original_four_class_label"],
        "features_leakage_excluded": [
            "pokerbench.correct_decision_raw",
            "pokerbench.original_four_class_label",
            "bootstrap_label",
            "labels.*",
            "debug.*",
            "audit.*",
        ],
        "leakage_columns_used_by_model": [feature for feature in FEATURE_COLUMNS if is_leakage_column(feature)],
        "label_source": "pokerbench_solver_oracle",
        "allowed_predictions": list(INTENT_LABELS),
        "intent_mapping": dict(INTENT_MAPPING),
        "no_invest_resolution": {
            "check_possible": "CHECK",
            "check_impossible": "FOLD",
        },
        "raw_text_direct_features": False,
        "bot_live_connection": "not_modified",
    }


def build_comparison_with_four_class_baseline(training: Mapping[str, Any], *, rows_used: int) -> str:
    baseline = read_json(FOUR_CLASS_BASELINE_REPORT)
    lines = [
        "# pokerbench_oracle_3intent_v1 vs pokerbench_oracle_baseline_v1",
        "",
        "| model | label source | rows | classes | accuracy | macro F1 |",
        "|---|---|---:|---|---:|---:|",
        f"| pokerbench_oracle_3intent_v1 | solver oracle mapped to intents | {rows_used} | NO_INVEST/CALL/RAISE | {training['accuracy']} | {training['macro_f1']} |",
        f"| pokerbench_oracle_baseline_v1 | solver oracle | {baseline.get('rows_usable')} | CHECK/FOLD/CALL/RAISE | {baseline.get('accuracy')} | {baseline.get('macro_f1')} |",
        "",
        "- The scores are related but not equivalent because CHECK and FOLD are merged in the intent model.",
        "- Neither model is connected to the live bot.",
        "",
    ]
    return "\n".join(lines)


def write_candidates_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "snapshot_id",
        *FEATURE_COLUMNS,
        *AUDIT_ONLY_COLUMNS,
        "pokerbench.original_four_class_label",
        "metadata.label_source",
        "bootstrap_label",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix_as_dict(truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predictions, labels=list(labels))
    return {
        expected: {predicted: int(matrix[row_index][col_index]) for col_index, predicted in enumerate(labels)}
        for row_index, expected in enumerate(labels)
    }


def without_model(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "model"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--full-diagnostics", action="store_true", help="Use larger samples for graphical diagnostics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pokerbench_oracle_3intent_v1(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        download=not args.no_download,
        max_rows=args.max_rows,
        random_seed=args.random_seed,
        fast_mode=not args.full_diagnostics,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
