"""Train separate preflop and postflop 3-intent models from the merged oracle dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pokerbench_oracle_baseline_v1 import FEATURE_COLUMNS


DEFAULT_DATA_DIR = Path("outputs/readiness/merged_oracle_3intent_v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/readiness")
PREFLOP_DIR_NAME = "merged_oracle_preflop_model_v1"
POSTFLOP_DIR_NAME = "merged_oracle_postflop_model_v1"
COMPARISON_PATH = Path("outputs/readiness/merged_oracle_preflop_postflop_comparison_v1.md")
LABELS = ("NO_INVEST", "CALL", "RAISE")
STAGES = ("preflop", "postflop")
SPLITS = ("train", "validation", "test")
CATEGORICAL_FEATURES = ("metadata.street", "features.hero_position")
NUMERIC_FEATURES = tuple(feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES)


def run_merged_oracle_preflop_postflop_v1(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    preflop_output_dir: str | Path | None = None,
    postflop_output_dir: str | Path | None = None,
    comparison_path: str | Path = COMPARISON_PATH,
    force: bool = False,
    random_seed: int = 17,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    output_root_path = Path(output_root)
    preflop_dir = Path(preflop_output_dir) if preflop_output_dir else output_root_path / PREFLOP_DIR_NAME
    postflop_dir = Path(postflop_output_dir) if postflop_output_dir else output_root_path / POSTFLOP_DIR_NAME
    reset_output_dir(preflop_dir, force=force)
    reset_output_dir(postflop_dir, force=force)

    reports = {
        "preflop": train_stage_model(data_path=data_path, stage="preflop", output_dir=preflop_dir, random_seed=random_seed),
        "postflop": train_stage_model(data_path=data_path, stage="postflop", output_dir=postflop_dir, random_seed=random_seed),
    }
    comparison = render_comparison_report(reports)
    comparison_output = Path(comparison_path)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.write_text(comparison, encoding="utf-8")
    return {
        "status": "ok",
        "preflop_report": str(preflop_dir / "training_report.json"),
        "postflop_report": str(postflop_dir / "training_report.json"),
        "comparison_report": str(comparison_output),
        "reports": reports,
    }


def reset_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            return
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def train_stage_model(*, data_path: Path, stage: str, output_dir: Path, random_seed: int) -> dict[str, Any]:
    splits = load_stage_splits(data_path, stage)
    validate_stage_splits(splits, stage=stage)
    model = make_model(random_seed=random_seed)
    x_train = splits["train"]["x"]
    y_train = splits["train"]["y"]["label_3intent"].astype(str)
    model.fit(x_train, y_train)

    report = evaluate_model(model, splits=splits, stage=stage)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    report["model_path"] = str(model_path)
    report["feature_names"] = list(FEATURE_COLUMNS)
    report["leakage_columns_used_by_model"] = [feature for feature in FEATURE_COLUMNS if feature in {"source_dataset", "source_row_id", "label_3intent"}]

    write_json(report, output_dir / "training_report.json")
    write_bar_chart(report["label_distribution"], output_dir / "label_distribution.svg", title=f"{stage} label distribution")
    write_bar_chart(report["source_distribution"], output_dir / "source_distribution.svg", title=f"{stage} source distribution")
    write_confusion_matrix_svg(report["confusion_matrix"], output_dir / "confusion_matrix.svg")
    write_metric_bar_chart(report["performance_by_source_dataset"], output_dir / "performance_by_source.svg", title=f"{stage} performance by source")
    write_metric_bar_chart(report["performance_by_street"], output_dir / "performance_by_street.svg", title=f"{stage} performance by street")
    write_feature_importance_svg(model, output_dir / "feature_importance.svg")
    return report


def load_stage_splits(data_path: Path, stage: str) -> dict[str, dict[str, pd.DataFrame]]:
    splits = {}
    for split in SPLITS:
        x_path = data_path / f"X_{split}_{stage}.csv"
        y_path = data_path / f"y_{split}_{stage}.csv"
        splits[split] = {
            "x": pd.read_csv(x_path),
            "y": pd.read_csv(y_path),
        }
    return splits


def validate_stage_splits(splits: dict[str, dict[str, pd.DataFrame]], *, stage: str) -> None:
    for split in SPLITS:
        x = splits[split]["x"]
        y = splits[split]["y"]
        missing = [feature for feature in FEATURE_COLUMNS if feature not in x.columns]
        if missing:
            raise ValueError(f"{stage}:{split}:missing_feature:{missing[0]}")
        forbidden = set(x.columns) & {"source_dataset", "source_row_id", "label_3intent", "raw_prompt", "raw_response", "raw_action"}
        if forbidden:
            raise ValueError(f"{stage}:{split}:leakage_column:{sorted(forbidden)[0]}")
        if len(x) != len(y):
            raise ValueError(f"{stage}:{split}:x_y_length_mismatch")
    train_labels = set(splits["train"]["y"]["label_3intent"].astype(str))
    if len(train_labels) < 2:
        raise ValueError(f"{stage}:not_enough_train_classes:{sorted(train_labels)}")


def make_model(*, random_seed: int) -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipe, list(CATEGORICAL_FEATURES)),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(model: Pipeline, *, splits: dict[str, dict[str, pd.DataFrame]], stage: str) -> dict[str, Any]:
    x_test = splits["test"]["x"]
    y_test = splits["test"]["y"]["label_3intent"].astype(str)
    predictions = pd.Series(model.predict(x_test)).astype(str)
    y_all = pd.concat([splits[split]["y"] for split in SPLITS], ignore_index=True)
    x_all = pd.concat([splits[split]["x"] for split in SPLITS], ignore_index=True)
    report = {
        "status": "ok",
        "stage": stage.upper(),
        "rows_train": int(len(splits["train"]["x"])),
        "rows_validation": int(len(splits["validation"]["x"])),
        "rows_test": int(len(splits["test"]["x"])),
        "label_distribution": count_series(y_all["label_3intent"]),
        "label_distribution_train": count_series(splits["train"]["y"]["label_3intent"]),
        "label_distribution_validation": count_series(splits["validation"]["y"]["label_3intent"]),
        "label_distribution_test": count_series(splits["test"]["y"]["label_3intent"]),
        "source_distribution": count_series(y_all["source_dataset"]),
        "source_distribution_train": count_series(splits["train"]["y"]["source_dataset"]),
        "source_distribution_test": count_series(splits["test"]["y"]["source_dataset"]),
        "street_distribution": count_series(x_all["metadata.street"]),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=list(LABELS), average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, labels=list(LABELS), average="weighted", zero_division=0)), 6),
        "classification_report": classification_report(y_test, predictions, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": matrix_as_dict(y_test, predictions, LABELS),
        "performance_by_source_dataset": grouped_performance(splits["test"]["y"], y_test, predictions, group_column="source_dataset"),
        "performance_by_street": grouped_performance(x_test, y_test, predictions, group_column="metadata.street"),
        "performance_by_label_and_source": performance_by_label_and_source(splits["test"]["y"], y_test, predictions),
    }
    for label in LABELS:
        report[f"recall_{label}"] = round(float(report["classification_report"].get(label, {}).get("recall", 0.0)), 6)
    report["precision_per_class"] = {
        label: round(float(report["classification_report"].get(label, {}).get("precision", 0.0)), 6)
        for label in LABELS
    }
    return report


def grouped_performance(frame: pd.DataFrame, truth: pd.Series, predictions: pd.Series, *, group_column: str) -> dict[str, dict[str, Any]]:
    result = {}
    groups = frame[group_column].fillna("UNKNOWN").astype(str) if group_column in frame.columns else pd.Series(["UNKNOWN"] * len(frame))
    for group in sorted(set(groups)):
        indexes = [idx for idx, value in enumerate(groups) if value == group]
        if not indexes:
            continue
        y_true = truth.iloc[indexes]
        y_pred = predictions.iloc[indexes]
        result[group] = {
            "rows": int(len(indexes)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)), 6),
        }
    return result


def performance_by_label_and_source(frame: pd.DataFrame, truth: pd.Series, predictions: pd.Series) -> dict[str, dict[str, dict[str, Any]]]:
    groups = frame["source_dataset"].fillna("UNKNOWN").astype(str) if "source_dataset" in frame.columns else pd.Series(["UNKNOWN"] * len(frame))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for source in sorted(set(groups)):
        result[source] = {}
        source_indexes = [idx for idx, value in enumerate(groups) if value == source]
        for label in LABELS:
            label_indexes = [idx for idx in source_indexes if truth.iloc[idx] == label]
            if not label_indexes:
                result[source][label] = {"rows": 0, "recall": None}
                continue
            correct = sum(1 for idx in label_indexes if predictions.iloc[idx] == label)
            result[source][label] = {"rows": len(label_indexes), "recall": round(float(correct / len(label_indexes)), 6)}
    return result


def count_series(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(series.fillna("UNKNOWN").astype(str)).items())}


def matrix_as_dict(truth: pd.Series, predictions: pd.Series, labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predictions, labels=list(labels))
    return {
        expected: {predicted: int(matrix[row_index][col_index]) for col_index, predicted in enumerate(labels)}
        for row_index, expected in enumerate(labels)
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_bar_chart(counts: dict[str, int], output_path: Path, *, title: str) -> None:
    items = list(counts.items())
    width = 760
    height = max(180, 70 + 32 * len(items))
    max_count = max([value for _, value in items] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, title, size=18)]
    for index, (label, value) in enumerate(items):
        y = 58 + index * 32
        bar_width = 560 * (value / max_count if max_count else 0)
        lines.append(svg_text(20, y + 13, str(label), size=11))
        lines.append(f'<rect x="190" y="{y}" width="{bar_width:.2f}" height="18" fill="#3d6f9f"/>')
        lines.append(svg_text(190 + bar_width + 8, y + 13, str(value), size=11))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_metric_bar_chart(metrics: dict[str, dict[str, Any]], output_path: Path, *, title: str) -> None:
    counts = {key: int(round(float(value.get("macro_f1", 0.0)) * 1000)) for key, value in metrics.items()}
    write_bar_chart(counts, output_path, title=title + " (macro F1 x1000)")


def write_confusion_matrix_svg(matrix: dict[str, dict[str, int]], output_path: Path) -> None:
    cell = 86
    left = 135
    top = 80
    width = left + cell * len(LABELS) + 80
    height = top + cell * len(LABELS) + 70
    max_value = max([int(matrix.get(row, {}).get(col, 0)) for row in LABELS for col in LABELS] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, "Confusion matrix", size=18)]
    for col, label in enumerate(LABELS):
        lines.append(svg_text(left + col * cell + cell / 2, top - 14, f"pred {label}", size=10, anchor="middle"))
    for row_index, label in enumerate(LABELS):
        lines.append(svg_text(left - 12, top + row_index * cell + cell / 2 + 4, f"true {label}", size=10, anchor="end"))
        for col, pred in enumerate(LABELS):
            value = int(matrix.get(label, {}).get(pred, 0))
            x = left + col * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{blue_scale(value / max_value if max_value else 0)}" stroke="#ffffff"/>')
            lines.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), size=11, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_importance_svg(model: Pipeline, output_path: Path) -> None:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names = list(FEATURE_COLUMNS)
    values = list(getattr(classifier, "feature_importances_", []))
    ranking = sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:20]
    width = 920
    height = max(180, 55 + 28 * len(ranking))
    left = 330
    max_value = max([value for _, value in ranking] or [1.0])
    lines = [svg_header(width, height), svg_text(20, 28, "Feature importance", size=18)]
    for index, (name, value) in enumerate(ranking):
        y = 55 + index * 28
        bar_width = 520 * (float(value) / max_value if max_value else 0)
        lines.append(svg_text(left - 10, y + 5, short_label(name, 44), size=10, anchor="end"))
        lines.append(f'<rect x="{left}" y="{y - 10}" width="{bar_width:.2f}" height="16" fill="#5b7f44"/>')
        lines.append(svg_text(left + bar_width + 6, y + 4, f"{float(value):.4f}", size=10))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_comparison_report(reports: dict[str, dict[str, Any]]) -> str:
    baseline = read_json(Path("outputs/readiness/pokerbench_oracle_3intent_v1/training_report.json"))
    rows = [
        "# Merged Oracle Preflop/Postflop Comparison",
        "",
        "| model | rows | accuracy | macro F1 | recall NO_INVEST | recall CALL | recall RAISE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline:
        rows.append(
            f"| pokerbench_oracle_3intent_v1 | {baseline.get('rows_usable')} | {baseline.get('accuracy')} | {baseline.get('macro_f1')} | "
            f"{recall_for(baseline, 'NO_INVEST')} | {recall_for(baseline, 'CALL')} | {recall_for(baseline, 'RAISE')} |"
        )
    for name, report in reports.items():
        total_rows = report["rows_train"] + report["rows_validation"] + report["rows_test"]
        rows.append(
            f"| merged_{name}_model_v1 | {total_rows} | {report['accuracy']} | {report['macro_f1']} | "
            f"{recall_for(report, 'NO_INVEST')} | {recall_for(report, 'CALL')} | {recall_for(report, 'RAISE')} |"
        )
    rows.extend(
        [
            "",
            "## Warnings",
            "",
            "- Source/domain shift is expected: PokerBench structured rows and GTO-style HF rows do not share the same distribution.",
            "- Source dataset is used only for grouped evaluation, not as a model feature.",
            "- PHH/ACPC remains excluded from supervised oracle training.",
            "",
            "## Recommendation",
            "",
            "Use these models for offline comparison only. Treat improvement as coverage evidence only after source-level and street-level performance are stable.",
            "",
        ]
    )
    return "\n".join(rows)


def recall_for(report: dict[str, Any], label: str) -> Any:
    return round(float(report.get("classification_report", {}).get(label, {}).get("recall", 0.0)), 6)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-family="Arial">{escape_xml(text)}</text>'


def escape_xml(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short_label(value: str, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def blue_scale(value: float) -> str:
    ratio = max(0.0, min(1.0, value))
    red = int(238 - 160 * ratio)
    green = int(244 - 110 * ratio)
    blue = int(250 - 40 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--preflop-output-dir", default=None)
    parser.add_argument("--postflop-output-dir", default=None)
    parser.add_argument("--comparison-path", default=str(COMPARISON_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_merged_oracle_preflop_postflop_v1(
        data_dir=args.data_dir,
        output_root=args.output_root,
        preflop_output_dir=args.preflop_output_dir,
        postflop_output_dir=args.postflop_output_dir,
        comparison_path=args.comparison_path,
        force=args.force,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
