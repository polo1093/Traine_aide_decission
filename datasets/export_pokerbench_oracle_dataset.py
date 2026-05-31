"""Export a PokerBench oracle dataset aligned with the model input features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pokerbench_oracle_3intent_v1 import (  # noqa: E402
    INTENT_LABELS,
    INTENT_MAPPING,
    map_candidates_to_intents,
)
from experiments.pokerbench_oracle_baseline_v1 import (  # noqa: E402
    ALLOWED_LABELS,
    AUDIT_ONLY_COLUMNS,
    DEFAULT_DATA_DIR,
    FEATURE_COLUMNS,
    build_candidates,
    is_leakage_column,
    load_pokerbench_rows,
    prepare_pokerbench_sources,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("outputs/readiness/pokerbench_oracle_dataset_v1")
LABEL_MODES = ("3intent", "4class")
SPLITS = ("train", "validation", "test")


def export_pokerbench_oracle_dataset(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    label_mode: str = "3intent",
    download: bool = True,
    max_rows: int | None = None,
    random_seed: int = 17,
    validation_size: float = 0.10,
    test_size: float = 0.20,
) -> dict[str, Any]:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported_label_mode:{label_mode}")
    validate_split_sizes(validation_size=validation_size, test_size=test_size)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_paths = prepare_pokerbench_sources(data_path, download=download)
    loaded_rows = load_pokerbench_rows(source_paths, max_rows=max_rows)
    four_class_rows, parse_report = build_candidates(loaded_rows)
    rows = map_rows_for_label_mode(four_class_rows, label_mode=label_mode)
    if len(rows) < 20:
        raise ValueError(f"not_enough_usable_pokerbench_rows:{len(rows)}")

    rows = assign_splits(
        rows,
        random_seed=random_seed,
        validation_size=validation_size,
        test_size=test_size,
    )

    paths = write_dataset_files(rows, output_path=output_path)
    report = build_report(
        rows,
        source_paths=source_paths,
        loaded_count=len(loaded_rows),
        label_mode=label_mode,
        parse_report=parse_report,
        paths=paths,
        random_seed=random_seed,
        validation_size=validation_size,
        test_size=test_size,
    )
    write_json(report, output_path / "dataset_report.json")
    (output_path / "dataset_card.md").write_text(render_dataset_card(report), encoding="utf-8")
    write_json(build_feature_contract(label_mode=label_mode), output_path / "feature_contract.json")
    write_json(build_preprocessing_schema(rows), output_path / "preprocessing_schema.json")
    return report


def validate_split_sizes(*, validation_size: float, test_size: float) -> None:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("split_sizes_must_be_positive")
    if validation_size + test_size >= 0.8:
        raise ValueError("split_sizes_leave_too_little_training_data")


def map_rows_for_label_mode(rows: Sequence[Mapping[str, Any]], *, label_mode: str) -> list[dict[str, Any]]:
    if label_mode == "4class":
        return [dict(row) for row in rows]
    return map_candidates_to_intents(rows)


def assign_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    validation_size: float,
    test_size: float,
) -> list[dict[str, Any]]:
    indexed = [dict(row) for row in rows]
    labels = [str(row["bootstrap_label"]) for row in indexed]
    train_validation, test = train_test_split(
        indexed,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    train_validation_labels = [str(row["bootstrap_label"]) for row in train_validation]
    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=random_seed,
        stratify=train_validation_labels,
    )

    split_rows = []
    for split_name, split_items in (("train", train), ("validation", validation), ("test", test)):
        for row in split_items:
            with_split = dict(row)
            with_split["split"] = split_name
            split_rows.append(with_split)
    return sorted(split_rows, key=lambda row: str(row["snapshot_id"]))


def write_dataset_files(rows: Sequence[Mapping[str, Any]], *, output_path: Path) -> dict[str, str]:
    paths = {
        "model_input_csv": str(output_path / "model_input.csv"),
        "audit_candidates_csv": str(output_path / "audit_candidates.csv"),
        "dataset_jsonl": str(output_path / "dataset.jsonl"),
    }
    write_model_input_csv(rows, Path(paths["model_input_csv"]))
    write_audit_csv(rows, Path(paths["audit_candidates_csv"]))
    write_jsonl(rows, Path(paths["dataset_jsonl"]))

    for split in SPLITS:
        split_rows = [row for row in rows if row.get("split") == split]
        x_path = output_path / f"X_{split}.csv"
        y_path = output_path / f"y_{split}.csv"
        split_path = output_path / f"{split}.csv"
        write_x_csv(split_rows, x_path)
        write_y_csv(split_rows, y_path)
        write_model_input_csv(split_rows, split_path)
        paths[f"X_{split}_csv"] = str(x_path)
        paths[f"y_{split}_csv"] = str(y_path)
        paths[f"{split}_csv"] = str(split_path)
    return paths


def write_model_input_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = ["snapshot_id", "split", *FEATURE_COLUMNS, "bootstrap_label"]
    write_csv_rows(rows, output_path, fieldnames)


def write_x_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    write_csv_rows(rows, output_path, list(FEATURE_COLUMNS))


def write_y_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    write_csv_rows(rows, output_path, ["snapshot_id", "bootstrap_label"])


def write_audit_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    audit_columns = list(AUDIT_ONLY_COLUMNS)
    if any("pokerbench.original_four_class_label" in row for row in rows):
        audit_columns.append("pokerbench.original_four_class_label")
    fieldnames = ["snapshot_id", "split", *FEATURE_COLUMNS, *audit_columns, "metadata.label_source", "bootstrap_label"]
    write_csv_rows(rows, output_path, fieldnames)


def write_csv_rows(rows: Sequence[Mapping[str, Any]], output_path: Path, fieldnames: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_paths: Sequence[Path],
    loaded_count: int,
    label_mode: str,
    parse_report: Mapping[str, Any],
    paths: Mapping[str, str],
    random_seed: int,
    validation_size: float,
    test_size: float,
) -> dict[str, Any]:
    split_counts = Counter(str(row["split"]) for row in rows)
    split_label_distribution = {
        split: dict(sorted(Counter(row["bootstrap_label"] for row in rows if row.get("split") == split).items()))
        for split in SPLITS
    }
    return {
        "status": "ok",
        "schema": "pokerbench_oracle_dataset_v1",
        "label_mode": label_mode,
        "dataset_source": "RZ412/PokerBench",
        "input_files": [str(path) for path in source_paths],
        "rows_loaded": loaded_count,
        "rows_usable": len(rows),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "bootstrap_label",
        "label_source": "pokerbench_solver_oracle" if label_mode == "4class" else "pokerbench_solver_oracle_3intent",
        "allowed_labels": list(ALLOWED_LABELS if label_mode == "4class" else INTENT_LABELS),
        "intent_mapping": dict(INTENT_MAPPING) if label_mode == "3intent" else None,
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in rows).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_distribution": split_label_distribution,
        "street_distribution": dict(sorted(Counter(row["metadata.street"] for row in rows).items())),
        "preflop_postflop_distribution": preflop_postflop_distribution(rows),
        "unmapped_outputs": parse_report["unmapped_outputs"],
        "unmapped_count": parse_report["unmapped_count"],
        "random_seed": random_seed,
        "validation_size": validation_size,
        "test_size": test_size,
        "output_files": dict(paths),
        "feature_contract": "feature_contract.json",
        "preprocessing_schema": "preprocessing_schema.json",
        "dataset_card": "dataset_card.md",
        "leakage_columns_used_by_x": [feature for feature in FEATURE_COLUMNS if is_leakage_column(feature)],
        "x_files_exclude_label_audit_and_raw_text": True,
        "raw_text_direct_features": False,
        "bot_live_connection": "not_modified",
        "not_for_production": True,
        "recommended_training_input": "X_train.csv + y_train.csv",
    }


def preflop_postflop_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter("preflop" if row.get("metadata.street") == "PREFLOP" else "postflop" for row in rows)
    return dict(sorted(counts.items()))


def build_feature_contract(*, label_mode: str) -> dict[str, Any]:
    return {
        "schema": "pokerbench_oracle_dataset_v1",
        "label_mode": label_mode,
        "features_model_used": list(FEATURE_COLUMNS),
        "features_audit_only": [*AUDIT_ONLY_COLUMNS, "pokerbench.original_four_class_label"],
        "features_leakage_excluded": [
            "bootstrap_label",
            "split",
            "metadata.label_source",
            "pokerbench.correct_decision_raw",
            "pokerbench.original_four_class_label",
            "labels.*",
            "debug.*",
            "audit.*",
            "raw text",
        ],
        "leakage_columns_used_by_x": [feature for feature in FEATURE_COLUMNS if is_leakage_column(feature)],
        "target_column": "bootstrap_label",
        "allowed_labels": list(ALLOWED_LABELS if label_mode == "4class" else INTENT_LABELS),
        "raw_text_direct_features": False,
        "bot_live_connection": "not_modified",
    }


def build_preprocessing_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categorical_features = [feature for feature in FEATURE_COLUMNS if feature.startswith("metadata.") or feature == "features.hero_position"]
    numeric_features = [feature for feature in FEATURE_COLUMNS if feature not in categorical_features]
    return {
        "feature_order": list(FEATURE_COLUMNS),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_values": {
            feature: sorted({str(row.get(feature) or "UNKNOWN") for row in rows})
            for feature in categorical_features
        },
        "target": "bootstrap_label",
        "split_column": "split",
    }


def render_dataset_card(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PokerBench Oracle Dataset V1",
            "",
            "Dataset offline construit depuis PokerBench oracle, aligne sur les features d'entree du modele.",
            "",
            f"- label_mode: `{report['label_mode']}`",
            f"- rows_usable: `{report['rows_usable']}`",
            f"- label_source: `{report['label_source']}`",
            f"- target_column: `{report['target_column']}`",
            f"- feature_count: `{len(report['feature_columns'])}`",
            "",
            "## Fichiers",
            "",
            "- `X_train.csv`, `X_validation.csv`, `X_test.csv` contiennent seulement les features d'entree.",
            "- `y_train.csv`, `y_validation.csv`, `y_test.csv` contiennent les labels.",
            "- `audit_candidates.csv` garde les champs PokerBench bruts pour audit, pas pour entrainement.",
            "",
            "## Labels",
            "",
            "```json",
            json.dumps(report["label_distribution"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Garde-fous",
            "",
            "- Aucun branchement live.",
            "- Aucun label/debug/audit/raw text dans les fichiers `X_*`.",
            "- Les labels viennent de PokerBench oracle, pas des logs live.",
            "- Dataset offline uniquement.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="3intent")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export_pokerbench_oracle_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        label_mode=args.label_mode,
        download=not args.no_download,
        max_rows=args.max_rows,
        random_seed=args.random_seed,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
