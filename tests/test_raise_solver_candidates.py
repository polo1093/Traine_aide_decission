from __future__ import annotations

from experiments.generate_raise_solver_candidates import (
    build_raise_plan,
    build_report,
    context_fields,
)


def test_raise_plan_has_stability_pairs() -> None:
    plan = build_raise_plan()

    assert plan
    assert len(plan) % 2 == 0
    scenarios = {}
    for row in plan:
        scenarios.setdefault(row["scenario"], set()).add(row["iterations"])
    assert all(iterations == {25, 100} for iterations in scenarios.values())


def test_context_fields_only_allow_facing_bet_contexts() -> None:
    assert context_fields("hero_ip_facing_bet") == ("IP", "hero_facing_bet")
    assert context_fields("hero_oop_facing_bet") == ("OOP", "hero_facing_bet")


def test_raise_report_counts_usable_solver_raise_records() -> None:
    records = [
        {
            "solver_status": "ok",
            "root_matches_hero": True,
            "root_player_role": "hero",
            "dominant_action": "RAISE_10",
            "dominant_frequency": 0.8,
            "danger_flags": [],
        },
        {
            "solver_status": "ok",
            "root_matches_hero": True,
            "root_player_role": "hero",
            "dominant_action": "ALL_IN",
            "dominant_frequency": 0.9,
            "danger_flags": ["extreme_action_all_in"],
        },
    ]
    report = build_report(
        records,
        {"label_source_counts": {"solver_candidate": 1}, "normalized_label_distribution": {"RAISE": 1}},
        output_dir=__import__("pathlib").Path("out"),
        duration_ms=1.0,
    )

    assert report["status"] == "ok"
    assert report["usable_raise_solver_records"] == 1
    assert report["root_player_not_hero_errors"] == 0
