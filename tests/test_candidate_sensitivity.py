from __future__ import annotations

import json
from pathlib import Path

from experiments.analyze_candidate_sensitivity import (
    calculate_spr,
    load_jsonl_safely,
    summarize_sensitivity_records,
)


def record(
    *,
    context: str = "hero_ip_facing_bet",
    scenario: str = "mid",
    iterations: int = 25,
    action: str = "CALL",
    frequency: float = 0.7,
    root_hero: bool = True,
) -> dict:
    return {
        "context": context,
        "scenario": scenario,
        "iterations": iterations,
        "solver_job_id": f"job_{context}_{scenario}_{iterations}",
        "solver_status": "ok",
        "root_matches_hero": root_hero,
        "root_player_role": "hero" if root_hero else "villain",
        "dominant_action": action,
        "dominant_frequency": frequency,
        "candidate_status": "ok",
        "candidate_exclusion_reason": None,
        "exploitability_last": 0.1,
        "danger_flags": ["extreme_action_all_in"] if action == "ALL_IN" else [],
        "is_training_label": False,
    }


def test_calculate_spr() -> None:
    assert calculate_spr(1000, 300) == 3.333333
    assert calculate_spr(1000, 0) is None


def test_summary_by_context() -> None:
    summary = summarize_sensitivity_records(
        [
            record(iterations=25, action="CALL"),
            record(iterations=50, action="CALL", frequency=0.8),
            record(iterations=100, action="CALL", frequency=0.9),
        ]
    )

    ctx = summary["by_context"]["hero_ip_facing_bet"]
    assert ctx["solve_count"] == 3
    assert ctx["dominant_action_counts"] == {"CALL": 3}
    assert ctx["all_in_count"] == 0
    assert summary["groups"][0]["stable_action"] is True
    assert summary["groups"][0]["quality_status"] == "ok"
    assert summary["is_training_label"] is False


def test_detects_frequent_all_in() -> None:
    summary = summarize_sensitivity_records(
        [
            record(iterations=25, action="ALL_IN", frequency=0.7),
            record(iterations=50, action="ALL_IN", frequency=0.8),
            record(iterations=100, action="ALL_IN", frequency=0.9),
        ]
    )

    ctx = summary["by_context"]["hero_ip_facing_bet"]
    assert ctx["all_in_count"] == 3
    assert ctx["all_in_rate"] == 1.0
    assert "extreme_action_all_in" in summary["groups"][0]["danger_flags"]


def test_detects_unstable_action() -> None:
    summary = summarize_sensitivity_records(
        [
            record(iterations=25, action="CALL", frequency=0.7),
            record(iterations=50, action="FOLD", frequency=0.8),
            record(iterations=100, action="CALL", frequency=0.9),
        ]
    )

    group = summary["groups"][0]
    assert group["quality_status"] == "failed"
    assert group["exclusion_reason"] == "dominant_action_unstable"
    assert "dominant_action_unstable" in group["danger_flags"]


def test_no_training_label_fields_in_summary() -> None:
    summary = summarize_sensitivity_records([record(iterations=25), record(iterations=50)])

    assert "training_label" not in summary
    assert "gto_label" not in summary
    assert summary["is_training_label"] is False
    assert all(group["is_training_label"] is False for group in summary["groups"])


def test_load_jsonl_safely_handles_incomplete_lines(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_text(
        json.dumps(record(iterations=25)) + "\n{not json}\n[]\n\n",
        encoding="utf-8",
    )

    result = load_jsonl_safely(path)

    assert result["status"] == "ok"
    assert len(result["records"]) == 1
    assert "line_1:invalid_json" in result["warnings"]
    assert "line_2:non_object" in result["warnings"]
    assert "line_3:empty" in result["warnings"]
