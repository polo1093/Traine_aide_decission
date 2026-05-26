from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from datasets.export_candidate_dataset import (
    build_candidate_dataset_rows,
    calculate_spr,
    export_candidate_dataset,
    normalize_bootstrap_action,
)


def sensitivity_record(
    *,
    source_id: str,
    context: str = "hero_ip_facing_bet",
    scenario: str = "stable_call",
    iterations: int = 25,
    action: str = "CALL",
    frequency: float = 0.75,
    solver_status: str = "ok",
    root_matches_hero: bool = True,
    root_player_role: str = "hero",
    error: str | None = None,
    danger_flags: list[str] | None = None,
) -> dict:
    return {
        "record_type": "candidate_sensitivity_result",
        "solver_job_id": source_id,
        "context": context,
        "scenario": scenario,
        "street": "RIVER",
        "stack": 1000,
        "pot": 300,
        "to_call": 100,
        "spr": calculate_spr(1000, 300),
        "iterations": iterations,
        "solver_status": solver_status,
        "root_matches_hero": root_matches_hero,
        "root_player_role": root_player_role,
        "action_frequencies": {action: frequency, "FOLD": round(1.0 - frequency, 6)},
        "dominant_action": action,
        "dominant_frequency": frequency,
        "danger_flags": danger_flags or [],
        "error": error,
        "training_label": "SHOULD_NOT_LEAK",
        "gto_label": "SHOULD_NOT_LEAK",
    }


def stable_pair(**overrides: object) -> list[dict]:
    base = dict(overrides)
    first = sensitivity_record(source_id="job_a_it25", iterations=25, **base)
    second = sensitivity_record(source_id="job_a_it50", iterations=50, **base)
    return [first, second]


def first_row(records: list[dict]) -> dict:
    return build_candidate_dataset_rows(records)[0]


def test_valid_candidate_is_exported() -> None:
    row = first_row(stable_pair(action="CALL", frequency=0.75))

    assert row["excluded"] is False
    assert row["bootstrap_label"] == "CALL"
    assert row["raw_action"] == "CALL"
    assert row["normalized_action"] == "CALL"
    assert row["label_source"] == "solver_candidate"
    assert row["label_quality"] == "bootstrap_solver_untrusted"


def test_all_in_is_excluded() -> None:
    row = first_row(stable_pair(action="ALL_IN", frequency=0.9, danger_flags=["extreme_action_all_in"]))

    assert row["excluded"] is True
    assert row["bootstrap_label"] is None
    assert row["normalized_action"] is None
    assert row["exclusion_reason"] == "all_in_excluded"


def test_root_not_hero_is_excluded() -> None:
    row = first_row(stable_pair(root_matches_hero=False, root_player_role="villain"))

    assert row["excluded"] is True
    assert row["exclusion_reason"] == "root_not_hero"


def test_low_frequency_is_excluded() -> None:
    row = first_row(stable_pair(action="CALL", frequency=0.69))

    assert row["excluded"] is True
    assert row["exclusion_reason"] == "dominant_action_too_weak"


def test_timeout_is_excluded_before_solver_failed() -> None:
    row = first_row(stable_pair(solver_status="failed", error="subprocess timeout after 5s"))

    assert row["excluded"] is True
    assert row["exclusion_reason"] == "timeout"


def test_unstable_action_is_excluded() -> None:
    records = [
        sensitivity_record(source_id="job_a_it25", iterations=25, action="CALL", frequency=0.8),
        sensitivity_record(source_id="job_a_it50", iterations=50, action="FOLD", frequency=0.8),
    ]

    row = first_row(records)

    assert row["excluded"] is True
    assert row["exclusion_reason"] == "action_unstable"


def test_iterations_too_low_is_excluded() -> None:
    records = [
        sensitivity_record(source_id="job_a_it10", iterations=10, action="CALL", frequency=0.8),
        sensitivity_record(source_id="job_a_it20", iterations=20, action="CALL", frequency=0.8),
    ]

    row = first_row(records)

    assert row["excluded"] is True
    assert row["exclusion_reason"] == "iterations_too_low"


def test_jsonl_and_csv_are_readable(tmp_path: Path) -> None:
    input_path = tmp_path / "runs.jsonl"
    jsonl_path = tmp_path / "dataset.jsonl"
    csv_path = tmp_path / "dataset.csv"
    records = stable_pair(action="BET_33", frequency=0.8)
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    summary = export_candidate_dataset([input_path], output_jsonl=jsonl_path, output_csv=csv_path)

    assert summary["candidates_exported"] == 2
    assert Path(summary["dataset_report_json"]).exists()
    assert Path(summary["dataset_report_md"]).exists()
    json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert json_rows[0]["raw_action"] == "BET_33"
    assert json_rows[0]["normalized_action"] == "RAISE"
    assert json_rows[0]["bootstrap_label"] == "RAISE"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["raw_action"] == "BET_33"
    assert csv_rows[0]["normalized_action"] == "RAISE"
    assert csv_rows[0]["bootstrap_label"] == "RAISE"


def test_export_does_not_emit_forbidden_label_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "runs.jsonl"
    jsonl_path = tmp_path / "dataset.jsonl"
    records = stable_pair(action="CALL", frequency=0.75)
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    export_candidate_dataset([input_path], output_jsonl=jsonl_path)

    payload = jsonl_path.read_text(encoding="utf-8")
    row = json.loads(payload.splitlines()[0])
    assert "training_label" not in row
    assert "gto_label" not in row
    assert "is_training_label" not in row
    assert row["label_quality"] == "bootstrap_solver_untrusted"


def test_bet_and_raise_are_normalized_to_raise() -> None:
    assert normalize_bootstrap_action("BET_33") == "RAISE"
    assert normalize_bootstrap_action("BET_50") == "RAISE"
    assert normalize_bootstrap_action("RAISE_33") == "RAISE"
    assert normalize_bootstrap_action("RAISE_66") == "RAISE"


def test_all_in_normalizes_to_none() -> None:
    assert normalize_bootstrap_action("ALL_IN") is None


def test_include_weak_rules_creates_raise_rows_and_source_is_distinct(tmp_path: Path) -> None:
    input_path = tmp_path / "runs.jsonl"
    jsonl_path = tmp_path / "dataset.jsonl"
    csv_path = tmp_path / "dataset.csv"
    records = stable_pair(action="CHECK", frequency=0.9)
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    summary = export_candidate_dataset(
        [input_path],
        output_jsonl=jsonl_path,
        output_csv=csv_path,
        include_weak_rules=True,
        min_usable_rows=12,
        class_floor=3,
    )

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    kept = [row for row in rows if not row["excluded"]]
    labels = {row["bootstrap_label"] for row in kept}
    sources = {row["label_source"] for row in kept}
    raise_rows = [row for row in kept if row["bootstrap_label"] == "RAISE"]
    assert summary["candidates_exported"] >= 12
    assert {"CHECK", "FOLD", "RAISE"}.issubset(labels)
    assert {"solver_candidate", "weak_rule_bootstrap"}.issubset(sources)
    assert raise_rows
    assert raise_rows[0]["label_quality"] == "bootstrap_weak_rule_untrusted"
    assert raise_rows[0]["weak_rule_reason"]


def test_weak_rule_fixture_has_check_fold_raise() -> None:
    rows = build_candidate_dataset_rows(stable_pair(action="CHECK", frequency=0.9))
    from datasets.export_candidate_dataset import generate_weak_rule_rows

    rows.extend(generate_weak_rule_rows(rows, min_usable_rows=15, class_floor=4))
    labels = Counter(row["bootstrap_label"] for row in rows if not row["excluded"])

    assert labels["CHECK"] >= 4
    assert labels["FOLD"] >= 4
    assert labels["RAISE"] >= 4
