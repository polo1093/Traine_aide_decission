from __future__ import annotations

from experiments.generate_bootstrap_solver_candidates import (
    build_solver_candidate_plan,
    count_root_player_not_hero,
    critical_warning_list,
)


def test_solver_plan_uses_exact_staged_solve_count_and_hero_contexts() -> None:
    plan = build_solver_candidate_plan(target_solves=50, seed=9400)

    assert len(plan) == 50
    assert {row["iterations"] for row in plan} == {25, 100}
    assert {row["context"] for row in plan} <= {
        "hero_oop_check_or_bet",
        "hero_ip_facing_bet",
        "hero_oop_facing_bet",
    }
    assert all(row["scenario"].startswith("solver_") for row in plan)


def test_solver_report_flags_root_player_not_hero_as_critical() -> None:
    records = [
        {"root_matches_hero": True, "root_player_role": "hero", "danger_flags": []},
        {"root_matches_hero": False, "root_player_role": "villain", "danger_flags": []},
    ]
    export_summary = {
        "status": "ok",
        "warnings": ["not_gto"],
        "candidates_exported": 3,
    }
    class_distribution = {"CHECK": 1, "FOLD": 1, "RAISE": 1}

    critical = critical_warning_list(
        export_summary=export_summary,
        root_player_not_hero_errors=count_root_player_not_hero(records),
        class_distribution=class_distribution,
    )

    assert "root_player_not_hero" in critical


def test_solver_report_accepts_balanced_expected_bootstrap_warnings() -> None:
    export_summary = {
        "status": "ok",
        "warnings": ["not_gto", "not_for_production", "dataset_contains_weak_rule_labels"],
        "candidates_exported": 300,
    }
    class_distribution = {"CHECK": 100, "FOLD": 100, "RAISE": 100}

    critical = critical_warning_list(
        export_summary=export_summary,
        root_player_not_hero_errors=0,
        class_distribution=class_distribution,
    )

    assert critical == []
