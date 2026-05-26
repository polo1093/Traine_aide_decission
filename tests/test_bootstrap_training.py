from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from models import train_bootstrap_model as training_module
from models.train_bootstrap_model import BootstrapTrainingError, FEATURE_COLUMNS, train_bootstrap_model


FIELDNAMES = [
    "source_id",
    "street",
    "hero_cards",
    "villain_hand",
    "board_cards",
    "pot",
    "to_call",
    "stack",
    "spr",
    "position_model",
    "decision_context_type",
    "action_frequencies",
    "dominant_action",
    "dominant_action_frequency",
    "raw_action",
    "normalized_action",
    "bootstrap_label",
    "label_source",
    "label_quality",
    "weak_rule_reason",
    "excluded",
    "exclusion_reason",
]


def write_dataset(path: Path, rows: list[dict], extra_fields: list[str] | None = None) -> None:
    fields = list(FIELDNAMES)
    for field in extra_fields or []:
        if field not in fields:
            fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            full = {field: row.get(field, "") for field in fields}
            writer.writerow(full)


def row(index: int, label: str, *, excluded: bool = False) -> dict:
    is_fold = label == "FOLD"
    return {
        "source_id": f"row_{index}",
        "street": "RIVER",
        "hero_cards": "null",
        "villain_hand": "null",
        "board_cards": "null",
        "pot": 300 + index,
        "to_call": 100 if is_fold else 0,
        "stack": 1000,
        "spr": 3.33,
        "position_model": "IP" if is_fold else "OOP",
        "decision_context_type": "hero_facing_bet" if is_fold else "hero_check_or_bet",
        "action_frequencies": "{}",
        "dominant_action": label,
        "dominant_action_frequency": 0.9,
        "raw_action": label,
        "normalized_action": label,
        "bootstrap_label": "" if excluded else label,
        "label_source": "solver_candidate",
        "label_quality": "bootstrap_solver_untrusted",
        "weak_rule_reason": "",
        "excluded": str(excluded),
        "exclusion_reason": "all_in_excluded" if excluded else "",
    }


def two_class_rows() -> list[dict]:
    return [row(i, "CHECK") for i in range(5)] + [row(i + 5, "FOLD") for i in range(7)]


def test_trains_on_mini_two_class_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model"
    write_dataset(input_path, two_class_rows())

    report = train_bootstrap_model(
        input_path=input_path,
        output_dir=output_dir,
        model_type="random_forest",
        min_rows=10,
    )

    assert report["status"] == "ok"
    assert report["training_quality"] == "pipeline_smoke_only"
    assert report["not_for_production"] is True
    assert report["contains_weak_rule_labels"] is False
    assert report["label_distribution"] == {"CHECK": 5, "FOLD": 7}
    assert "dataset_has_less_than_50_training_rows" in report["warnings"]
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "label_mapping.json").exists()
    assert (output_dir / "evaluation_report.json").exists()
    assert (output_dir / "evaluation_report.md").exists()


def test_refuses_single_class_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    write_dataset(input_path, [row(i, "CHECK") for i in range(10)])

    with pytest.raises(BootstrapTrainingError, match="single_class_dataset"):
        train_bootstrap_model(input_path=input_path, output_dir=tmp_path / "out", model_type="random_forest", min_rows=10)


def test_refuses_gto_label_column(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    rows = two_class_rows()
    for item in rows:
        item["gto_label"] = item["bootstrap_label"]
    write_dataset(input_path, rows, extra_fields=["gto_label"])

    with pytest.raises(BootstrapTrainingError, match="forbidden_column_present:gto_label"):
        train_bootstrap_model(input_path=input_path, output_dir=tmp_path / "out", model_type="random_forest", min_rows=10)


def test_refuses_training_label_column(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    rows = two_class_rows()
    for item in rows:
        item["training_label"] = item["bootstrap_label"]
    write_dataset(input_path, rows, extra_fields=["training_label"])

    with pytest.raises(BootstrapTrainingError, match="forbidden_column_present:training_label"):
        train_bootstrap_model(input_path=input_path, output_dir=tmp_path / "out", model_type="random_forest", min_rows=10)


def test_refuses_all_in_label(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    rows = two_class_rows() + [row(99, "ALL_IN")]
    write_dataset(input_path, rows)

    with pytest.raises(BootstrapTrainingError, match="all_in_label_present"):
        train_bootstrap_model(input_path=input_path, output_dir=tmp_path / "out", model_type="random_forest", min_rows=10)


def test_refuses_missing_bootstrap_label_column(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    fields = [field for field in FIELDNAMES if field != "bootstrap_label"]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in two_class_rows():
            writer.writerow({field: item.get(field, "") for field in fields})

    with pytest.raises(BootstrapTrainingError, match="bootstrap_label_missing"):
        train_bootstrap_model(input_path=input_path, output_dir=tmp_path / "out", model_type="random_forest", min_rows=10)


def test_exports_are_readable(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model"
    write_dataset(input_path, two_class_rows())

    train_bootstrap_model(input_path=input_path, output_dir=output_dir, model_type="logistic_regression", min_rows=10)

    report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
    mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    assert report["model_type"] == "logistic_regression"
    assert schema["target"] == "bootstrap_label"
    assert mapping["labels"] == ["CHECK", "FOLD"]


def test_auto_training_includes_dummy_and_selects_by_macro_f1(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model"
    rows = two_class_rows()
    for index in range(10):
        weak = row(200 + index, "RAISE")
        weak["label_source"] = "weak_rule_bootstrap"
        weak["label_quality"] = "bootstrap_weak_rule_untrusted"
        weak["raw_action"] = "BET_33"
        weak["normalized_action"] = "RAISE"
        rows.append(weak)
    write_dataset(input_path, rows)

    report = train_bootstrap_model(input_path=input_path, output_dir=output_dir, model_type="auto", min_rows=10)

    assert report["selection_metric"] == "macro_f1"
    assert "dummy" in report["model_comparison"]
    assert report["selected_model"] in {"logistic_regression", "random_forest", "extra_trees"}
    selected = report["model_comparison"][report["selected_model"]]
    non_dummy_scores = [
        model_report["macro_f1"]
        for name, model_report in report["model_comparison"].items()
        if name != "dummy"
    ]
    assert selected["macro_f1"] == max(non_dummy_scores)


def test_report_marks_weak_rule_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model"
    rows = two_class_rows()
    for index in range(10):
        weak = row(100 + index, "RAISE")
        weak["label_source"] = "weak_rule_bootstrap"
        weak["label_quality"] = "bootstrap_weak_rule_untrusted"
        weak["weak_rule_reason"] = "two_pair_plus_or_better_value_aggression_no_all_in"
        weak["raw_action"] = "BET_33"
        weak["normalized_action"] = "RAISE"
        rows.append(weak)
    write_dataset(input_path, rows)

    report = train_bootstrap_model(input_path=input_path, output_dir=output_dir, model_type="random_forest", min_rows=10)

    assert report["contains_weak_rule_labels"] is True
    assert report["not_for_production"] is True
    assert "contains_weak_rule_labels" in report["warnings"]
    assert "label_quality_bootstrap_weak_rule_untrusted" in report["warnings"]


def test_training_module_does_not_import_solver_or_aide_decision() -> None:
    source = Path(training_module.__file__).read_text(encoding="utf-8")

    assert "PokerSolver" not in source
    assert "poker_solver" not in source
    assert "aide_decision" not in source


def test_model_features_are_dist_aligned_and_exclude_audit_leakage() -> None:
    assert "features.pot" in FEATURE_COLUMNS
    assert "metadata.street" in FEATURE_COLUMNS

    forbidden = {
        "label_source",
        "label_quality",
        "dominant_action_frequency",
        "iterations",
        "exploitability_last",
        "candidate_confidence",
        "villain_hand",
    }
    assert not (set(FEATURE_COLUMNS) & forbidden)
