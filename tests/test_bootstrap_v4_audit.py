from __future__ import annotations

import csv
from pathlib import Path

from experiments.audit_bootstrap_v4 import build_dataset_audit, build_quality_warnings, write_manual_review_sample
from experiments.audit_bootstrap_v4 import write_input_category_by_output_heatmap, write_input_numeric_by_output_heatmap


def row(
    *,
    label: str = "CHECK",
    source: str = "solver_candidate",
    excluded: bool = False,
    reason: str = "",
    position: str = "OOP",
) -> dict[str, str]:
    return {
        "source_id": f"{source}_{label}",
        "street": "RIVER",
        "position_model": position,
        "decision_context_type": "hero_check_or_bet" if position == "OOP" else "hero_facing_bet",
        "bootstrap_label": "" if excluded else label,
        "label_source": "" if excluded else source,
        "excluded": str(excluded),
        "exclusion_reason": reason,
        "dominant_action_frequency": "0.9",
        "pot": "100",
        "to_call": "20",
        "stack": "500",
        "spr": "5",
        "to_call_ratio": "0.2",
        "stack_to_pot_ratio": "5",
        "board_card_count": "5",
    }


def test_audit_counts_sources_classes_positions_and_rejections() -> None:
    rows = [
        row(label="CHECK", source="solver_candidate", position="OOP"),
        row(label="FOLD", source="solver_candidate", position="IP"),
        row(label="RAISE", source="weak_rule_bootstrap", position="OOP"),
        row(excluded=True, reason="all_in_excluded"),
    ]

    audit = build_dataset_audit(rows, dataset_path="dataset.csv")

    assert audit["usable_rows"] == 3
    assert audit["rejected_rows"] == 1
    assert audit["rows_by_source"] == {"solver_candidate": 2, "weak_rule_bootstrap": 1}
    assert audit["classes_by_source"]["solver_candidate"] == {"CHECK": 1, "FOLD": 1, "RAISE": 0}
    assert audit["classes_by_hero_position"]["OOP"] == {"CHECK": 1, "FOLD": 0, "RAISE": 1}
    assert audit["rejections_by_reason"] == {"all_in_excluded": 1}
    assert audit["solver_candidate_rate"] == 0.666667
    assert audit["bootstrap_status"]["not_gto"] is True
    assert audit["bootstrap_status"]["not_for_production"] is True
    assert audit["bootstrap_status"]["river_only"] is True
    assert any(warning["code"] == "street_coverage_warning" for warning in audit["quality_warnings"])


def test_manual_sample_reports_missing_solver_raise(tmp_path: Path) -> None:
    rows = [
        row(label="CHECK", source="solver_candidate"),
        row(label="FOLD", source="solver_candidate"),
        row(label="RAISE", source="weak_rule_bootstrap"),
    ]
    output = tmp_path / "sample.csv"

    summary = write_manual_review_sample(rows, output_csv=output, sample_per_class=1)

    assert summary["selected_by_class"] == {"CHECK": 1, "FOLD": 1, "RAISE": 0}
    assert summary["shortfalls"] == {"RAISE": 1}
    with output.open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    assert len(sample_rows) == 2
    assert sample_rows[0]["manual_review_decision"] == ""


def test_input_output_heatmaps_are_written(tmp_path: Path) -> None:
    rows = [
        row(label="CHECK", source="solver_candidate", position="OOP"),
        row(label="FOLD", source="solver_candidate", position="IP"),
        row(label="RAISE", source="weak_rule_bootstrap", position="OOP"),
    ]
    numeric_path = tmp_path / "numeric.svg"
    category_path = tmp_path / "category.svg"

    write_input_numeric_by_output_heatmap(rows, numeric_path)
    write_input_category_by_output_heatmap(rows, category_path)

    assert "V4 Inputs Numeriques" in numeric_path.read_text(encoding="utf-8")
    assert "V4 Inputs Categoriels" in category_path.read_text(encoding="utf-8")
    assert "CHECK" in numeric_path.read_text(encoding="utf-8")


def test_quality_warnings_detect_source_action_bias() -> None:
    rows = [
        row(label="RAISE", source="weak_rule_bootstrap"),
        row(label="RAISE", source="weak_rule_bootstrap"),
        row(label="RAISE", source="weak_rule_bootstrap"),
        row(label="RAISE", source="solver_candidate"),
    ]
    audit = build_dataset_audit(rows, dataset_path="dataset.csv")
    warnings = audit["quality_warnings"]

    assert any(
        warning["code"] == "source_action_bias_warning"
        and warning["class"] == "RAISE"
        and warning["dominant_source"] == "weak_rule_bootstrap"
        for warning in warnings
    )


def test_quality_warnings_detect_deterministic_to_call() -> None:
    rows = [
        row(label="CHECK", source="solver_candidate", position="OOP") | {"to_call": "0", "to_call_ratio": "0"},
        row(label="FOLD", source="solver_candidate", position="IP") | {"to_call": "200", "to_call_ratio": "0.8"},
        row(label="RAISE", source="solver_candidate", position="OOP") | {"to_call": "100", "to_call_ratio": "0.4"},
    ]
    audit = build_dataset_audit(rows, dataset_path="dataset.csv")

    assert any(
        warning["code"] == "deterministic_feature_warning"
        and warning["feature"] == "to_call"
        for warning in audit["quality_warnings"]
    )
