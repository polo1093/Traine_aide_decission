"""Write separated feature-status heatmaps and markdown reports for bootstrap v5."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sklearn.exceptions import ConvergenceWarning
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_bootstrap_v5 import (
    LABELS,
    escape_xml,
    float_or_none,
    heat_color,
    is_leakage_column,
    is_null_feature_value,
    load_candidate_csv,
    normalize_01,
    select_training_rows,
    short_heatmap_label,
    svg_header,
)
from models.train_bootstrap_model import FEATURE_COLUMNS, NUMERIC_FEATURES, feature_payload, make_model


DEFAULT_INPUT = Path("outputs/readiness/bootstrap_candidate_dataset_v5_4000/candidates.csv")
DEFAULT_MODEL_DIR = Path("outputs/readiness/bootstrap_model_v5_4000")


def write_feature_view_reports(
    *,
    input_csv: str | Path = DEFAULT_INPUT,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, str]:
    model_path = Path(model_dir)
    rows, _ = load_candidate_csv(input_csv)
    usable_rows = select_training_rows(rows)
    contract = read_json(model_path / "feature_contract.json")
    audit = read_json(model_path / "feature_bias_audit.json")
    ablation = read_json(model_path / "ablation_report.json")

    used = list(contract.get("features_model_used", []))
    constants = list(contract.get("features_constant_excluded", []))
    leakage = list(contract.get("features_leakage_excluded", []))
    not_available = list(contract.get("features_not_available_yet", []))
    audit_only = [
        feature for feature in contract.get("features_audit_only", [])
        if feature not in set(leakage)
        and feature not in set(used)
        and feature not in set(constants)
        and feature not in set(not_available)
    ]

    all_stats = {
        feature: generic_feature_stats(usable_rows, feature)
        for feature in sorted(set(used) | set(constants) | set(audit_only) | set(leakage) | set(not_available))
    }
    audit_stats = dict(audit.get("feature_statistics", {}))
    all_stats.update({feature: {**all_stats.get(feature, {}), **stats} for feature, stats in audit_stats.items()})

    paths = {
        "heatmap_model_used_features": model_path / "heatmap_model_used_features.svg",
        "heatmap_excluded_constant_features": model_path / "heatmap_excluded_constant_features.svg",
        "heatmap_audit_only_features": model_path / "heatmap_audit_only_features.svg",
        "feature_status_summary": model_path / "feature_status_summary.md",
        "model_coefficients": model_path / "model_coefficients.md",
        "ablation_summary": model_path / "ablation_summary.md",
    }

    write_status_heatmap(
        usable_rows,
        used,
        paths["heatmap_model_used_features"],
        title="V5_4000 - features vraiment utilisees par le modele",
        subtitle="Moyennes numeriques normalisees par feature; categoriels = proportion non nulle par label.",
    )
    write_status_heatmap(
        usable_rows,
        constants,
        paths["heatmap_excluded_constant_features"],
        title="V5_4000 - features exclues car constantes",
        subtitle="Ces colonnes restent auditables mais ne sont plus passees a X_train.",
    )
    write_status_heatmap(
        usable_rows,
        audit_only,
        paths["heatmap_audit_only_features"],
        title="V5_4000 - colonnes audit-only",
        subtitle="Colonnes presentes dans le CSV pour inspection seulement; elles ne sont pas features sklearn.",
    )
    paths["feature_status_summary"].write_text(
        render_feature_status_summary(
            all_stats=all_stats,
            used=used,
            constants=constants,
            audit_only=audit_only,
            leakage=leakage,
            not_available=not_available,
        ),
        encoding="utf-8",
    )
    paths["model_coefficients"].write_text(
        render_logistic_coefficients(usable_rows, used),
        encoding="utf-8",
    )
    paths["ablation_summary"].write_text(render_ablation_summary(ablation), encoding="utf-8")

    return {key: str(path) for key, path in paths.items()}


def generic_feature_stats(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    values = [feature_value(row, feature) for row in rows]
    numbers = [float_or_none(value) for value in values]
    numeric_values = [value for value in numbers if value is not None]
    mean_by_label = {}
    for label in LABELS:
        label_values = [feature_value(row, feature) for row in rows if str(row.get("bootstrap_label")) == label]
        label_numbers = [float_or_none(value) for value in label_values]
        label_numeric = [value for value in label_numbers if value is not None]
        mean_by_label[label] = round(sum(label_numeric) / len(label_numeric), 6) if label_numeric else None
    return {
        "nunique": len(Counter(canonical_stats_value(value) for value in values)),
        "zero_proportion": zero_ratio(values),
        "mean_by_label": mean_by_label,
        "null_count": sum(1 for value in values if is_null_feature_value(value)),
        "type": "numeric" if numeric_values and len(numeric_values) >= max(1, len(values) // 2) else "categorical",
    }


def feature_value(row: Mapping[str, Any], feature: str) -> Any:
    if feature in FEATURE_COLUMNS:
        return feature_payload(row).get(feature)
    return row.get(feature)


def canonical_stats_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def zero_ratio(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    zeros = 0
    for value in values:
        number = float_or_none(value)
        if number is not None:
            zeros += int(abs(number) <= 1e-12)
        else:
            zeros += int(str(value).strip().lower() in {"false", "0", "0.0"})
    return round(zeros / len(values), 6)


def write_status_heatmap(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    matrix = [heatmap_row(rows, feature) for feature in features]
    write_matrix_heatmap_svg(matrix, output_path, title=title, subtitle=subtitle)


def heatmap_row(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    raw_values = {}
    numeric_labels = {}
    for label in LABELS:
        values = [feature_value(row, feature) for row in rows if str(row.get("bootstrap_label")) == label]
        numbers = [float_or_none(value) for value in values]
        numeric = [value for value in numbers if value is not None]
        if numeric and len(numeric) >= max(1, len(values) // 2):
            raw_values[label] = round(sum(numeric) / len(numeric), 6)
            numeric_labels[label] = True
        else:
            raw_values[label] = round(sum(0 if is_null_feature_value(value) else 1 for value in values) / len(values), 6) if values else None
            numeric_labels[label] = False
    numeric_means = [value for label, value in raw_values.items() if numeric_labels[label] and value is not None]
    if numeric_means:
        row_min = min(numeric_means)
        row_max = max(numeric_means)
        values = {
            label: normalize_01(value, row_min=row_min, row_max=row_max) if numeric_labels[label] else value
            for label, value in raw_values.items()
        }
    else:
        values = raw_values
    return {"feature": feature, "values": values, "raw_values": raw_values}


def write_matrix_heatmap_svg(matrix: Sequence[Mapping[str, Any]], output_path: Path, *, title: str, subtitle: str) -> None:
    cell_w = 92
    cell_h = 32
    left = 390
    top = 82
    width = left + cell_w * len(LABELS) + 40
    height = max(150, top + cell_h * max(1, len(matrix)) + 46)
    lines = [
        svg_header(width, height),
        f'<text x="20" y="30" font-size="18" font-family="Arial">{escape_xml(title)}</text>',
        f'<text x="20" y="54" font-size="12" font-family="Arial" fill="#555">{escape_xml(subtitle)}</text>',
    ]
    for index, label in enumerate(LABELS):
        x = left + index * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="74" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
    if not matrix:
        lines.append('<text x="20" y="110" font-size="13" font-family="Arial">Aucune feature dans ce groupe.</text>')
    for y_index, row in enumerate(matrix):
        y = top + y_index * cell_h
        lines.append(
            f'<text x="{left - 12}" y="{y + 21}" text-anchor="end" font-size="11" '
            f'font-family="Arial">{escape_xml(short_heatmap_label(str(row["feature"]), limit=58))}</text>'
        )
        for x_index, label in enumerate(LABELS):
            x = left + x_index * cell_w
            value = row["values"].get(label)
            ratio = 0.0 if value is None or (isinstance(value, float) and math.isnan(value)) else max(0.0, min(1.0, float(value)))
            text = "NA" if value is None else f"{ratio:.3f}".rstrip("0").rstrip(".")
            fill = heat_color(ratio)
            text_color = "#ffffff" if ratio > 0.55 else "#111111"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(
                f'<text x="{x + cell_w / 2}" y="{y + 21}" text-anchor="middle" font-size="11" '
                f'font-family="Arial" fill="{text_color}">{escape_xml(text)}</text>'
            )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_feature_status_summary(
    *,
    all_stats: Mapping[str, Mapping[str, Any]],
    used: Sequence[str],
    constants: Sequence[str],
    audit_only: Sequence[str],
    leakage: Sequence[str],
    not_available: Sequence[str],
) -> str:
    ordered = []
    for status_features in (used, constants, audit_only, leakage, not_available):
        for feature in status_features:
            if feature not in ordered:
                ordered.append(feature)
    lines = [
        "# Feature Status Summary",
        "",
        "| feature name | status | reason | nunique | zero_ratio | mean CHECK | mean FOLD | mean RAISE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for feature in ordered:
        status = feature_status(feature, used=used, constants=constants, audit_only=audit_only, leakage=leakage, not_available=not_available)
        stats = all_stats.get(feature, {})
        means = dict(stats.get("mean_by_label", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    md(feature),
                    status,
                    md(feature_reason(feature, status)),
                    format_stat(stats.get("nunique")),
                    format_stat(stats.get("zero_proportion")),
                    format_stat(means.get("CHECK")),
                    format_stat(means.get("FOLD")),
                    format_stat(means.get("RAISE")),
                ]
            )
            + " |"
        )
    lines.extend(["", "Leakage used by model: `[]`.", "CALL was not added. Bot live connection was not modified.", ""])
    return "\n".join(lines)


def feature_status(
    feature: str,
    *,
    used: Sequence[str],
    constants: Sequence[str],
    audit_only: Sequence[str],
    leakage: Sequence[str],
    not_available: Sequence[str],
) -> str:
    if feature in set(used):
        return "used_by_model"
    if feature in set(constants):
        return "constant_excluded"
    if feature in set(leakage) or is_leakage_column(feature):
        return "leakage_excluded"
    if feature in set(not_available):
        return "not_available_yet"
    if feature in set(audit_only):
        return "audit_only"
    return "audit_only"


def feature_reason(feature: str, status: str) -> str:
    if status == "used_by_model":
        return "Present in feature_contract.features_model_used and passed to sklearn."
    if status == "constant_excluded":
        return "Constant or quasi-constant in this v5_4000 CSV; excluded from X_train."
    if status == "leakage_excluded":
        return "Audit/debug/label/solver-derived target field; never passed to sklearn."
    if status == "not_available_yet":
        return "Expected by contract but not available in the current dataset."
    return "Kept in the CSV for audit and traceability only; not passed to sklearn."


def render_logistic_coefficients(rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> str:
    x = [feature_payload(row, feature_columns=feature_columns) for row in rows]
    y = [str(row["bootstrap_label"]) for row in rows]
    model = make_model("logistic_regression")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    lines = [
        "# Logistic Regression Coefficients",
        "",
        "Transient logistic_regression fit for interpretability only. The saved production/offline model artifact was not changed.",
        "",
    ]
    for class_index, label in enumerate(classifier.classes_):
        weights = [(names[index], float(classifier.coef_[class_index][index])) for index in range(len(names))]
        top_positive = sorted(weights, key=lambda item: item[1], reverse=True)[:12]
        top_negative = sorted(weights, key=lambda item: item[1])[:12]
        lines.extend([
            f"## Class {label}",
            "",
            "| positive feature | coefficient | negative feature | coefficient |",
            "|---|---:|---|---:|",
        ])
        for positive, negative in zip(top_positive, top_negative):
            lines.append(
                f"| {md(positive[0])} | {positive[1]:.6f} | {md(negative[0])} | {negative[1]:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_ablation_summary(ablation: Mapping[str, Any]) -> str:
    lines = [
        "# Ablation Summary",
        "",
        "| variant | status | rows | features | selected model | macro_f1 | accuracy |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for variant in ablation.get("variants", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    md(variant.get("name")),
                    md(variant.get("status")),
                    format_stat(variant.get("rows_used")),
                    format_stat(variant.get("feature_count")),
                    md(variant.get("selected_model", "")),
                    format_stat(variant.get("macro_f1")),
                    format_stat(variant.get("accuracy")),
                ]
            )
            + " |"
        )
    lines.extend([
        "",
        "Reading guide: compare each macro_f1 to `without_constant_features`; drops indicate feature groups carrying signal.",
        "This summary is reporting-only and does not change the saved model.",
        "",
    ])
    return "\n".join(lines)


def format_stat(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def md(value: Any) -> str:
    text = str(value).replace("|", "\\|")
    return f"`{text}`" if text else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_feature_view_reports(input_csv=args.input, model_dir=args.model_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
