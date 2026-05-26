"""Audit bootstrap v4 data, train the offline v4 model, and compare v3/v4.

This is an offline quality gate. It never connects a model to the live bot and
does not promote bootstrap labels to GTO labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_bootstrap_model import train_bootstrap_model


CLASSES = ("CHECK", "FOLD", "RAISE")
SOURCES = ("solver_candidate", "weak_rule_bootstrap")
SOURCE_DOMINANCE_WARNING_THRESHOLD = 0.70
DETERMINISTIC_FEATURE_WARNING_THRESHOLD = 0.75
DEFAULT_V3_CANDIDATES = "outputs/readiness/bootstrap_candidate_dataset_v3/candidates.csv"
DEFAULT_V3_REPORT = "outputs/readiness/bootstrap_model_v3/evaluation_report.json"
FALLBACK_V3_REPORT = "outputs/bootstrap_model_v3/evaluation_report.json"
DEFAULT_V4_CANDIDATES = "outputs/readiness/bootstrap_candidate_dataset_v4/candidates.csv"
DEFAULT_V4_MODEL_DIR = "outputs/readiness/bootstrap_model_v4"
DEFAULT_AUDIT_DIR = "outputs/readiness/bootstrap_candidate_dataset_v4"


def audit_bootstrap_v4(
    *,
    v4_candidates: str | Path = DEFAULT_V4_CANDIDATES,
    v3_candidates: str | Path = DEFAULT_V3_CANDIDATES,
    v3_report: str | Path = DEFAULT_V3_REPORT,
    v4_model_dir: str | Path = DEFAULT_V4_MODEL_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    sample_per_class: int = 10,
    min_rows: int = 100,
) -> dict[str, Any]:
    audit_path = Path(audit_dir)
    audit_path.mkdir(parents=True, exist_ok=True)

    v4_rows = load_csv_rows(v4_candidates)
    v3_rows = load_csv_rows(v3_candidates)
    audit = build_dataset_audit(v4_rows, dataset_path=v4_candidates)
    sample_summary = write_manual_review_sample(
        v4_rows,
        output_csv=audit_path / "manual_review_sample.csv",
        sample_per_class=sample_per_class,
    )
    v4_training = train_bootstrap_model(
        input_path=v4_candidates,
        output_dir=v4_model_dir,
        model_type="auto",
        min_rows=min_rows,
    )
    v3_report_path = resolve_v3_report(v3_report)
    v3_eval = load_json(v3_report_path)
    v4_eval = load_json(Path(v4_model_dir) / "evaluation_report.json")
    comparison = build_comparison(
        v3_rows=v3_rows,
        v4_rows=v4_rows,
        v3_eval=v3_eval,
        v4_eval=v4_eval,
        v3_candidates=v3_candidates,
        v4_candidates=v4_candidates,
        v3_report=v3_report_path,
        v4_report=Path(v4_model_dir) / "evaluation_report.json",
    )
    heatmap_paths = write_heatmaps(
        comparison,
        output_dir=audit_path,
        v4_rows=v4_rows,
    )

    payload = {
        "status": "ok",
        "not_gto": True,
        "not_for_production": True,
        "audit": audit,
        "manual_review_sample": sample_summary,
        "v4_training": summarize_training(v4_training),
        "comparison": comparison,
        "heatmaps": heatmap_paths,
        "outputs": {
            "audit_json": str(audit_path / "audit_report.json"),
            "audit_md": str(audit_path / "audit_report.md"),
            "manual_review_sample_csv": str(audit_path / "manual_review_sample.csv"),
            "comparison_json": str(audit_path / "v3_v4_comparison.json"),
            "comparison_md": str(audit_path / "v3_v4_comparison.md"),
            "v4_model_dir": str(v4_model_dir),
        },
    }
    write_json(payload, audit_path / "audit_report.json")
    (audit_path / "audit_report.md").write_text(render_audit_md(payload), encoding="utf-8")
    write_json(comparison, audit_path / "v3_v4_comparison.json")
    (audit_path / "v3_v4_comparison.md").write_text(render_comparison_md(comparison, heatmap_paths), encoding="utf-8")
    return payload


def build_dataset_audit(rows: list[dict[str, str]], *, dataset_path: str | Path) -> dict[str, Any]:
    usable = [row for row in rows if not is_true(row.get("excluded"))]
    rejected = [row for row in rows if is_true(row.get("excluded"))]
    source_counts = Counter(value(row, "label_source") for row in usable)
    solver_count = source_counts.get("solver_candidate", 0)
    weak_count = source_counts.get("weak_rule_bootstrap", 0)
    total_usable = len(usable)
    class_counts = Counter(value(row, "bootstrap_label") for row in usable)
    class_by_source = nested_counts(usable, "label_source", "bootstrap_label")
    class_by_street = nested_counts(usable, "street", "bootstrap_label")
    class_by_position = nested_counts(usable, "position_model", "bootstrap_label")
    rejected_by_reason = Counter(value(row, "exclusion_reason") for row in rejected)
    quality_warnings = build_quality_warnings(
        usable,
        class_by_source=class_by_source,
        class_by_street=class_by_street,
    )
    critical_findings = []
    if class_by_source.get("solver_candidate", {}).get("RAISE", 0) < 10:
        critical_findings.append("solver_candidate_raise_sample_shortfall")
    if not total_usable:
        critical_findings.append("no_usable_rows")

    return {
        "dataset_path": str(dataset_path),
        "rows_total": len(rows),
        "usable_rows": total_usable,
        "rejected_rows": len(rejected),
        "rows_by_source": dict(sorted(source_counts.items())),
        "classes_total": class_counter_payload(class_counts),
        "classes_by_source": normalize_nested_class_counts(class_by_source),
        "classes_by_street": normalize_nested_class_counts(class_by_street),
        "classes_by_hero_position": normalize_nested_class_counts(class_by_position),
        "rejections_by_reason": dict(sorted(rejected_by_reason.items())),
        "solver_candidate_rate": round(solver_count / total_usable, 6) if total_usable else 0.0,
        "weak_rule_bootstrap_rate": round(weak_count / total_usable, 6) if total_usable else 0.0,
        "quality_warnings": quality_warnings,
        "bootstrap_status": bootstrap_status(quality_warnings),
        "critical_findings": critical_findings,
    }


def build_quality_warnings(
    usable_rows: list[dict[str, str]],
    *,
    class_by_source: dict[str, Counter[str]],
    class_by_street: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    streets = sorted(class_by_street)
    if len(streets) <= 1:
        warnings.append(
            {
                "code": "street_coverage_warning",
                "severity": "non_critical",
                "detail": "dataset_covers_single_street",
                "streets": streets,
            }
        )

    totals_by_class = Counter(value(row, "bootstrap_label") for row in usable_rows)
    for label in CLASSES:
        total = totals_by_class.get(label, 0)
        if total <= 0:
            continue
        source_counts = {source: int(class_by_source.get(source, Counter()).get(label, 0)) for source in SOURCES}
        dominant_source, dominant_count = max(source_counts.items(), key=lambda item: item[1])
        rate = dominant_count / total
        if rate > SOURCE_DOMINANCE_WARNING_THRESHOLD:
            warnings.append(
                {
                    "code": "source_action_bias_warning",
                    "severity": "non_critical",
                    "detail": "class_dominated_by_single_label_source",
                    "class": label,
                    "dominant_source": dominant_source,
                    "dominant_source_rate": round(rate, 6),
                    "source_counts": source_counts,
                }
            )

    for warning in deterministic_feature_warnings(usable_rows):
        warnings.append(warning)
    return warnings


def deterministic_feature_warnings(usable_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for feature in ("to_call", "to_call_ratio", "board_card_count"):
        stats = numeric_feature_stats_by_class(usable_rows, feature)
        values = [payload["mean"] for payload in stats.values() if payload["mean"] is not None]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        max_abs = max(abs(value) for value in values)
        relative_spread = spread / max(max_abs, 1.0)
        unique_means = {round(value, 6) for value in values}
        if relative_spread > DETERMINISTIC_FEATURE_WARNING_THRESHOLD or (feature == "board_card_count" and len(unique_means) > 1):
            warnings.append(
                {
                    "code": "deterministic_feature_warning",
                    "severity": "non_critical",
                    "detail": "feature_distribution_may_identify_output_class",
                    "feature": feature,
                    "relative_spread": round(relative_spread, 6),
                    "class_stats": stats,
                }
            )
    return warnings


def numeric_feature_stats_by_class(rows: list[dict[str, str]], feature: str) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for label in CLASSES:
        values = [float_or_none(row.get(feature)) for row in rows if value(row, "bootstrap_label") == label]
        values = [item for item in values if item is not None]
        stats[label] = {
            "count": len(values),
            "mean": round(sum(values) / len(values), 6) if values else None,
            "min": round(min(values), 6) if values else None,
            "max": round(max(values), 6) if values else None,
        }
    return stats


def bootstrap_status(quality_warnings: list[dict[str, Any]]) -> dict[str, Any]:
    warning_codes = {str(item.get("code")) for item in quality_warnings}
    return {
        "not_gto": True,
        "not_for_production": True,
        "river_only": "street_coverage_warning" in warning_codes,
    }


def write_manual_review_sample(
    rows: list[dict[str, str]],
    *,
    output_csv: str | Path,
    sample_per_class: int,
) -> dict[str, Any]:
    selected: list[dict[str, str]] = []
    available_by_class: dict[str, int] = {}
    selected_by_class: dict[str, int] = {}
    for label in CLASSES:
        candidates = [
            row
            for row in rows
            if not is_true(row.get("excluded"))
            and value(row, "label_source") == "solver_candidate"
            and value(row, "bootstrap_label") == label
        ]
        candidates.sort(key=lambda row: (-float_or_zero(row.get("dominant_action_frequency")), value(row, "source_id")))
        chosen = candidates[:sample_per_class]
        selected.extend(chosen)
        available_by_class[label] = len(candidates)
        selected_by_class[label] = len(chosen)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    review_fields = ["manual_review_decision", "manual_review_notes", *fields]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"manual_review_decision": "", "manual_review_notes": "", **row})

    shortfalls = {
        label: sample_per_class - selected_by_class[label]
        for label in CLASSES
        if selected_by_class[label] < sample_per_class
    }
    return {
        "output_csv": str(output_path),
        "requested_per_class": sample_per_class,
        "selected_total": len(selected),
        "selected_by_class": selected_by_class,
        "available_solver_candidates_by_class": available_by_class,
        "shortfalls": shortfalls,
    }


def build_comparison(
    *,
    v3_rows: list[dict[str, str]],
    v4_rows: list[dict[str, str]],
    v3_eval: dict[str, Any],
    v4_eval: dict[str, Any],
    v3_candidates: str | Path,
    v4_candidates: str | Path,
    v3_report: str | Path,
    v4_report: str | Path,
) -> dict[str, Any]:
    return {
        "not_gto": True,
        "not_for_production": True,
        "v3": dataset_and_model_summary(v3_rows, v3_eval, candidates_path=v3_candidates, report_path=v3_report),
        "v4": dataset_and_model_summary(v4_rows, v4_eval, candidates_path=v4_candidates, report_path=v4_report),
        "overfit_signals": overfit_signals(v3_eval, v4_eval),
    }


def dataset_and_model_summary(
    rows: list[dict[str, str]],
    eval_report: dict[str, Any],
    *,
    candidates_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    usable = [row for row in rows if not is_true(row.get("excluded"))]
    return {
        "candidates_path": str(candidates_path),
        "evaluation_report": str(report_path),
        "rows_total": len(rows),
        "usable_rows": len(usable),
        "sources": dict(sorted(Counter(value(row, "label_source") for row in usable).items())),
        "class_distribution": class_counter_payload(Counter(value(row, "bootstrap_label") for row in usable)),
        "accuracy": eval_report.get("accuracy"),
        "macro_f1": eval_report.get("macro_f1"),
        "weighted_f1": eval_report.get("weighted_f1"),
        "confusion_matrix": eval_report.get("confusion_matrix", {}),
        "classification_report": eval_report.get("classification_report", {}),
        "warnings": eval_report.get("warnings", []),
    }


def overfit_signals(v3_eval: dict[str, Any], v4_eval: dict[str, Any]) -> list[str]:
    signals = []
    for version, report in (("v3", v3_eval), ("v4", v4_eval)):
        if report.get("accuracy") == 1.0 or report.get("macro_f1") == 1.0:
            signals.append(f"{version}:perfect_metrics_likely_bootstrap_rule_learning")
        if report.get("contains_weak_rule_labels") is True:
            signals.append(f"{version}:contains_weak_rule_labels")
        if report.get("model_may_learn_synthetic_rules") is True:
            signals.append(f"{version}:model_may_learn_synthetic_rules")
        if "call_class_absent" in set(report.get("warnings", [])):
            signals.append(f"{version}:call_class_absent")
    return signals


def write_heatmaps(
    comparison: dict[str, Any],
    *,
    output_dir: Path,
    v4_rows: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    paths = {}
    for version in ("v3", "v4"):
        matrix = comparison[version]["confusion_matrix"]
        path = output_dir / f"{version}_confusion_matrix_heatmap.svg"
        write_confusion_heatmap_svg(matrix, path, title=f"{version.upper()} Confusion Matrix")
        paths[f"{version}_confusion_matrix_heatmap"] = str(path)
    source_matrix = {
        source: comparison["v4"]["sources"].get(source, 0)
        for source in SOURCES
    }
    path = output_dir / "v4_source_bar_heatmap.svg"
    write_bar_heatmap_svg(source_matrix, path, title="V4 Label Sources")
    paths["v4_source_bar_heatmap"] = str(path)
    if v4_rows is not None:
        numeric_path = output_dir / "v4_input_numeric_by_output_heatmap.svg"
        categorical_path = output_dir / "v4_input_category_by_output_heatmap.svg"
        write_input_numeric_by_output_heatmap(v4_rows, numeric_path)
        write_input_category_by_output_heatmap(v4_rows, categorical_path)
        paths["v4_input_numeric_by_output_heatmap"] = str(numeric_path)
        paths["v4_input_category_by_output_heatmap"] = str(categorical_path)
    return paths


def write_input_numeric_by_output_heatmap(rows: list[dict[str, str]], output_path: Path) -> None:
    kept = [row for row in rows if not is_true(row.get("excluded")) and value(row, "bootstrap_label") in CLASSES]
    features = [
        "pot",
        "to_call",
        "stack",
        "spr",
        "dominant_action_frequency",
        "to_call_ratio",
        "stack_to_pot_ratio",
        "board_card_count",
    ]
    matrix: list[tuple[str, list[float | None]]] = []
    for feature in features:
        row_values: list[float | None] = []
        for label in CLASSES:
            values = [float_or_none(row.get(feature)) for row in kept if value(row, "bootstrap_label") == label]
            values = [item for item in values if item is not None]
            row_values.append(round(sum(values) / len(values), 6) if values else None)
        matrix.append((feature, row_values))
    write_matrix_heatmap_svg(
        matrix,
        output_path,
        title="V4 Inputs Numeriques -> Sortie",
        columns=list(CLASSES),
        normalize_by_row=True,
    )


def write_input_category_by_output_heatmap(rows: list[dict[str, str]], output_path: Path) -> None:
    kept = [row for row in rows if not is_true(row.get("excluded")) and value(row, "bootstrap_label") in CLASSES]
    category_specs = [
        ("label_source", ("solver_candidate", "weak_rule_bootstrap")),
        ("position_model", ("IP", "OOP")),
        ("decision_context_type", ("hero_check_or_bet", "hero_facing_bet")),
        ("street", ("TURN", "RIVER")),
    ]
    matrix: list[tuple[str, list[float | None]]] = []
    for field, expected_values in category_specs:
        for expected in expected_values:
            row_values = [
                float(sum(1 for row in kept if value(row, field) == expected and value(row, "bootstrap_label") == label))
                for label in CLASSES
            ]
            matrix.append((f"{field}={expected}", row_values))
    write_matrix_heatmap_svg(
        matrix,
        output_path,
        title="V4 Inputs Categoriels -> Sortie",
        columns=list(CLASSES),
        normalize_by_row=False,
    )


def write_matrix_heatmap_svg(
    matrix: list[tuple[str, list[float | None]]],
    output_path: Path,
    *,
    title: str,
    columns: list[str],
    normalize_by_row: bool,
) -> None:
    cell_w = 88
    cell_h = 34
    left = 230
    top = 72
    width = left + cell_w * len(columns) + 36
    height = top + cell_h * len(matrix) + 46
    lines = [svg_header(width, height), f'<text x="20" y="30" font-size="18" font-family="Arial">{escape_xml(title)}</text>']
    lines.append('<text x="20" y="52" font-size="12" font-family="Arial" fill="#555">Colonnes = sortie bootstrap_label; lignes = donnees d entree agregees.</text>')
    for index, column in enumerate(columns):
        x = left + index * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="64" text-anchor="middle" font-size="12" font-family="Arial">{escape_xml(column)}</text>')
    global_max = max([float(value) for _, values in matrix for value in values if value is not None] or [1.0])
    for y_index, (feature, values) in enumerate(matrix):
        y = top + y_index * cell_h
        row_numbers = [float(value) for value in values if value is not None]
        row_min = min(row_numbers) if row_numbers else 0.0
        row_max = max(row_numbers) if row_numbers else 1.0
        lines.append(f'<text x="{left - 12}" y="{y + 22}" text-anchor="end" font-size="12" font-family="Arial">{escape_xml(feature)}</text>')
        for x_index, raw_value in enumerate(values):
            x = left + x_index * cell_w
            if raw_value is None:
                ratio = 0.0
                label = "NA"
            else:
                numeric = float(raw_value)
                if normalize_by_row:
                    ratio = 0.0 if row_max == row_min else (numeric - row_min) / (row_max - row_min)
                else:
                    ratio = numeric / global_max if global_max else 0.0
                label = format_heatmap_value(numeric)
            fill = heat_color(ratio)
            text_color = "#ffffff" if ratio > 0.55 else "#111111"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(f'<text x="{x + cell_w / 2}" y="{y + 22}" text-anchor="middle" font-size="11" font-family="Arial" fill="{text_color}">{escape_xml(label)}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_heatmap_svg(matrix: Any, output_path: Path, *, title: str) -> None:
    labels = list(CLASSES)
    values = [
        [int((matrix.get(actual, {}) if isinstance(matrix, dict) else {}).get(predicted, 0)) for predicted in labels]
        for actual in labels
    ]
    max_value = max([value for row in values for value in row] or [1])
    cell = 72
    left = 110
    top = 70
    width = left + cell * len(labels) + 20
    height = top + cell * len(labels) + 40
    lines = [svg_header(width, height), f'<text x="20" y="30" font-size="18" font-family="Arial">{escape_xml(title)}</text>']
    for index, label in enumerate(labels):
        lines.append(f'<text x="{left + index * cell + cell / 2}" y="55" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
        lines.append(f'<text x="95" y="{top + index * cell + cell / 2 + 5}" text-anchor="end" font-size="12" font-family="Arial">{label}</text>')
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            fill = heat_color(value / max_value if max_value else 0.0)
            text_color = "#ffffff" if value / max_value > 0.55 else "#111111"
            lines.append(f'<rect x="{left + x * cell}" y="{top + y * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(f'<text x="{left + x * cell + cell / 2}" y="{top + y * cell + cell / 2 + 5}" text-anchor="middle" font-size="14" font-family="Arial" fill="{text_color}">{value}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_bar_heatmap_svg(values: dict[str, int], output_path: Path, *, title: str) -> None:
    max_value = max(values.values() or [1])
    width = 520
    height = 160
    lines = [svg_header(width, height), f'<text x="20" y="30" font-size="18" font-family="Arial">{escape_xml(title)}</text>']
    y = 55
    for label, value_count in values.items():
        bar_width = int((value_count / max_value) * 320) if max_value else 0
        lines.append(f'<text x="20" y="{y + 18}" font-size="13" font-family="Arial">{escape_xml(label)}</text>')
        lines.append(f'<rect x="180" y="{y}" width="{bar_width}" height="26" fill="{heat_color(value_count / max_value if max_value else 0)}"/>')
        lines.append(f'<text x="{190 + bar_width}" y="{y + 18}" font-size="13" font-family="Arial">{value_count}</text>')
        y += 42
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_audit_md(payload: dict[str, Any]) -> str:
    audit = payload["audit"]
    sample = payload["manual_review_sample"]
    lines = [
        "# Bootstrap V4 Audit",
        "",
        "Offline audit only. This is not GTO and not production data.",
        "",
        f"- rows_total: `{audit['rows_total']}`",
        f"- usable_rows: `{audit['usable_rows']}`",
        f"- rejected_rows: `{audit['rejected_rows']}`",
        f"- solver_candidate_rate: `{audit['solver_candidate_rate']}`",
        f"- weak_rule_bootstrap_rate: `{audit['weak_rule_bootstrap_rate']}`",
        f"- river_only: `{audit['bootstrap_status']['river_only']}`",
        "",
        "## Classes By Source",
        "",
        "```json",
        json.dumps(audit["classes_by_source"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Manual Sample",
        "",
        f"- output_csv: `{sample['output_csv']}`",
        f"- selected_by_class: `{sample['selected_by_class']}`",
        f"- shortfalls: `{sample['shortfalls']}`",
        "",
        "## Quality Warnings",
        "",
        "```json",
        json.dumps(audit["quality_warnings"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Critical Findings",
        "",
    ]
    lines.extend(f"- `{finding}`" for finding in audit["critical_findings"])
    return "\n".join(lines) + "\n"


def render_comparison_md(comparison: dict[str, Any], heatmaps: dict[str, str]) -> str:
    lines = [
        "# Bootstrap V3 vs V4 Comparison",
        "",
        "Offline comparison only. Perfect metrics are treated as an overfit warning, not proof of strategy quality.",
        "",
        "## Summary",
        "",
        "| version | usable_rows | sources | classes | accuracy | macro_f1 |",
        "|---|---:|---|---|---:|---:|",
    ]
    for version in ("v3", "v4"):
        item = comparison[version]
        lines.append(
            f"| {version} | {item['usable_rows']} | `{json.dumps(item['sources'], sort_keys=True)}` | "
            f"`{json.dumps(item['class_distribution'], sort_keys=True)}` | {item['accuracy']} | {item['macro_f1']} |"
        )
    lines.extend(
        [
            "",
            "## Overfit Signals",
            "",
        ]
    )
    lines.extend(f"- `{signal}`" for signal in comparison["overfit_signals"])
    lines.extend(
        [
            "",
            "## Heatmaps",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{path}`" for name, path in heatmaps.items())
    lines.extend(
        [
            "",
            "## V4 Classification Report",
            "",
            "```json",
            json.dumps(comparison["v4"]["classification_report"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def resolve_v3_report(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    fallback = Path(FALLBACK_V3_REPORT)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"v3_report_not_found:{candidate}")


def summarize_training(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "selected_model": report.get("selected_model"),
        "rows_used": report.get("rows_used"),
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro_f1"),
        "warnings": report.get("warnings", []),
        "output_files": report.get("output_files", {}),
    }


def nested_counts(rows: list[dict[str, str]], outer_field: str, inner_field: str) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[value(row, outer_field)][value(row, inner_field)] += 1
    return dict(result)


def normalize_nested_class_counts(value_map: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {outer: class_counter_payload(counter) for outer, counter in sorted(value_map.items())}


def class_counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in CLASSES}


def value(row: dict[str, str], key: str) -> str:
    raw = row.get(key)
    text = "" if raw is None else str(raw).strip()
    return text or "UNKNOWN"


def is_true(value_: Any) -> bool:
    return str(value_).strip().lower() in {"true", "1", "yes"}


def float_or_zero(value_: Any) -> float:
    try:
        return float(value_ or 0)
    except (TypeError, ValueError):
        return 0.0


def float_or_none(value_: Any) -> float | None:
    try:
        text = str(value_).strip()
        if text.lower() in {"", "none", "null", "nan"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def format_heatmap_value(value_: float) -> str:
    if abs(value_) >= 100:
        return str(int(round(value_)))
    if abs(value_) >= 10:
        return f"{value_:.1f}".rstrip("0").rstrip(".")
    return f"{value_:.3f}".rstrip("0").rstrip(".")


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def heat_color(ratio: float) -> str:
    safe = max(0.0, min(1.0, float(ratio)))
    red = int(245 - safe * 180)
    green = int(247 - safe * 120)
    blue = int(250 - safe * 40)
    return f"#{red:02x}{green:02x}{blue:02x}"


def escape_xml(value_: str) -> str:
    return str(value_).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-candidates", default=DEFAULT_V4_CANDIDATES)
    parser.add_argument("--v3-candidates", default=DEFAULT_V3_CANDIDATES)
    parser.add_argument("--v3-report", default=DEFAULT_V3_REPORT)
    parser.add_argument("--v4-model-dir", default=DEFAULT_V4_MODEL_DIR)
    parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--sample-per-class", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_bootstrap_v4(
        v4_candidates=args.v4_candidates,
        v3_candidates=args.v3_candidates,
        v3_report=args.v3_report,
        v4_model_dir=args.v4_model_dir,
        audit_dir=args.audit_dir,
        sample_per_class=args.sample_per_class,
        min_rows=args.min_rows,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
