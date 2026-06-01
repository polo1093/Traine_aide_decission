"""Generate a guarded bootstrap solver dataset from hero-oriented solver spots.

This script keeps the workflow offline and bootstrap-only. It never writes
``training_label`` or ``gto_label`` and it does not connect anything to the live
bot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.export_candidate_dataset import export_candidate_dataset
from experiments.analyze_candidate_sensitivity import calculate_spr
from solver_jobs.action_candidate import build_solver_action_candidate
from solver_jobs.hero_oriented_builder import build_hero_oriented_solver_job, validate_hero_root_alignment
from solver_jobs.strategy_extractor import extract_root_strategy
from solver_jobs.subprocess_runner import run_solver_job_subprocess
from synthetic.deck import draw_cards


STAGES = {"small": 50, "medium": 200, "large": 500}
STAGE_ORDER = ("small", "medium", "large")
CONTEXTS = ("hero_oop_check_or_bet", "hero_ip_facing_bet", "hero_oop_facing_bet")
ITERATIONS_PER_GROUP = (25, 100)
STREET = "RIVER"
BOARD_COUNT = 5
EXPECTED_WARNINGS = {
    "all_in_excluded",
    "bootstrap_solver_untrusted",
    "call_class_absent",
    "card_fields_missing_or_partial",
    "dataset_contains_weak_rule_labels",
    "not_for_production",
    "not_gto",
    "small_dataset",
    "synthetic_distribution_bias",
}
LABEL_QUALITY = "solver_candidate_untrusted"

AMOUNT_TEMPLATES = (
    {"name": "high_spr_small_call_b33", "stack": 2000.0, "pot": 100.0, "to_call_ratio": 0.10, "bet_sizes": [0.33]},
    {"name": "mid_spr_medium_call_b66", "stack": 1000.0, "pot": 300.0, "to_call_ratio": 0.33, "bet_sizes": [0.66]},
    {"name": "low_spr_large_call_b100", "stack": 200.0, "pot": 800.0, "to_call_ratio": 0.75, "bet_sizes": [1.0]},
    {"name": "probe_stack500_medium_call_b50", "stack": 500.0, "pot": 300.0, "to_call_ratio": 0.33, "bet_sizes": [0.5]},
    {"name": "deep_stack_dry_raise_b33", "stack": 1500.0, "pot": 220.0, "to_call_ratio": 0.18, "bet_sizes": [0.33]},
    {"name": "mid_stack_pressure_b66", "stack": 800.0, "pot": 260.0, "to_call_ratio": 0.45, "bet_sizes": [0.66]},
    {"name": "short_stack_commit_b100", "stack": 320.0, "pot": 500.0, "to_call_ratio": 0.60, "bet_sizes": [1.0]},
    {"name": "single_raise_probe_b50", "stack": 1200.0, "pot": 180.0, "to_call_ratio": 0.25, "bet_sizes": [0.5]},
)


def build_solver_candidate_plan(*, target_solves: int, seed: int) -> list[dict[str, Any]]:
    """Return an exact-size plan using only hero-oriented contexts."""

    if target_solves <= 0:
        raise ValueError("target_solves_must_be_positive")
    if target_solves % len(ITERATIONS_PER_GROUP) != 0:
        raise ValueError(f"target_solves_must_be_multiple_of:{len(ITERATIONS_PER_GROUP)}")

    plan: list[dict[str, Any]] = []
    group_count = target_solves // len(ITERATIONS_PER_GROUP)
    for group_index in range(group_count):
        context = CONTEXTS[group_index % len(CONTEXTS)]
        template = AMOUNT_TEMPLATES[group_index % len(AMOUNT_TEMPLATES)]
        scenario = f"solver_{template['name']}_{group_index:04d}"
        for iterations in ITERATIONS_PER_GROUP:
            to_call = 0.0 if context == "hero_oop_check_or_bet" else round(float(template["pot"]) * float(template["to_call_ratio"]), 6)
            plan.append(
                {
                    "context": context,
                    "scenario": scenario,
                    "stack": float(template["stack"]),
                    "pot": float(template["pot"]),
                    "to_call": to_call,
                    "to_call_ratio": 0.0 if context == "hero_oop_check_or_bet" else float(template["to_call_ratio"]),
                    "bet_sizes": [float(value) for value in template["bet_sizes"]],
                    "iterations": int(iterations),
                    "seed": int(seed),
                    "group_index": group_index,
                }
            )
    return plan


def run_bootstrap_solver_candidates(
    *,
    output_dir: str | Path,
    max_stage: str = "large",
    seed: int = 9400,
    timeout_s: float = 5.0,
    backend: str = "rust",
    min_dominant_frequency: float = 0.70,
    min_usable_rows: int = 500,
    class_floor: int = 50,
) -> dict[str, Any]:
    """Run staged solves and export the largest successful solver dataset."""

    if max_stage not in STAGES:
        raise ValueError(f"unsupported_stage:{max_stage}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_target = STAGES[max_stage]
    plan = build_solver_candidate_plan(target_solves=final_target, seed=seed)
    sensitivity_path = output_path / "candidate_sensitivity_results.jsonl"
    report_path = output_path / "dataset_report_solver.json"
    candidates_jsonl = output_path / "candidates.jsonl"
    candidates_csv = output_path / "candidates.csv"

    records: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    stage_targets = [STAGES[name] for name in STAGE_ORDER if STAGES[name] <= final_target]

    with sensitivity_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(plan, start=1):
            record = run_one_solver_candidate_row(row, timeout_s=timeout_s, backend=backend)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

            if index in stage_targets:
                stage_name = _stage_name(index)
                stage_input = output_path / f"{stage_name}_candidate_sensitivity_results.jsonl"
                write_records(stage_input, records)
                export_summary = export_candidate_dataset(
                    [stage_input],
                    output_jsonl=candidates_jsonl,
                    output_csv=candidates_csv,
                    include_weak_rules=True,
                    min_usable_rows=min_usable_rows,
                    class_floor=class_floor,
                    min_dominant_frequency=min_dominant_frequency,
                    balance_weak_rules=True,
                )
                stage_summary = build_solver_candidate_report(
                    records,
                    export_summary,
                    stage_name=stage_name,
                    requested_solves=index,
                    output_dir=output_path,
                    duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                )
                stage_summaries.append(stage_summary)
                report_path.write_text(
                    json.dumps({**stage_summary, "stages": stage_summaries}, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if stage_summary["critical_warnings"]:
                    return {**stage_summary, "stages": stage_summaries}

    final_summary = dict(stage_summaries[-1])
    final_summary["stages"] = stage_summaries
    report_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return final_summary


def run_one_solver_candidate_row(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    job = build_solver_candidate_job(row, timeout_s=timeout_s, backend=backend)
    root_validation = validate_hero_root_alignment(job)
    subprocess_result = run_solver_job_subprocess(job, timeout_s=timeout_s)
    solver_result = subprocess_result.get("solver_result") if isinstance(subprocess_result.get("solver_result"), dict) else {}
    extraction_input = {
        "solver_job_id": job["solver_job_id"],
        "solver_status": subprocess_result.get("solver_status"),
        "solver_result": solver_result,
        "quality": subprocess_result.get("quality"),
    }
    strategy = extract_root_strategy(extraction_input)
    candidate = build_solver_action_candidate(extraction_input)
    output = solver_result.get("output") if isinstance(solver_result.get("output"), dict) else {}
    root_strategy = strategy.get("root_strategy") if isinstance(strategy.get("root_strategy"), dict) else {}
    strategy_source = root_strategy.get("source")
    dominant_action = strategy.get("dominant_action")

    danger_flags: list[str] = []
    if dominant_action == "ALL_IN":
        danger_flags.append("extreme_action_all_in")
    if strategy.get("status") != "ok":
        danger_flags.append(str(strategy.get("error") or "strategy_not_available"))
    if subprocess_result.get("solver_status") != "ok":
        danger_flags.append(str(subprocess_result.get("solver_status") or "solver_not_ok"))
    if strategy.get("status") == "ok" and strategy_source != "average_strategy":
        danger_flags.append("strategy_source_not_average_strategy")

    return {
        "record_type": "candidate_sensitivity_result",
        "context": row["context"],
        "scenario": row["scenario"],
        "street": STREET,
        "hero_cards": job.get("hero_hand"),
        "villain_hand": job.get("villain_hand"),
        "board_cards": job.get("board"),
        "stack": row["stack"],
        "pot": row["pot"],
        "to_call": row["to_call"],
        "to_call_ratio": row["to_call_ratio"],
        "spr": calculate_spr(row["stack"], row["pot"]),
        "bet_size_fractions": row["bet_sizes"],
        "iterations": row["iterations"],
        "backend": backend,
        "solver_job_id": job["solver_job_id"],
        "solver_status": subprocess_result.get("solver_status"),
        "root_matches_hero": output.get("root_matches_hero"),
        "root_player_role": output.get("root_player_role"),
        "root_validation_status": root_validation.get("status"),
        "strategy_source": strategy_source,
        "legal_actions": root_validation.get("legal_action_labels", []),
        "action_frequencies": strategy.get("action_frequencies", {}),
        "dominant_action": dominant_action,
        "dominant_frequency": strategy.get("dominant_action_frequency"),
        "candidate_status": candidate.get("status"),
        "candidate_exclusion_reason": candidate.get("exclusion_reason"),
        "exploitability_last": (subprocess_result.get("quality") or {}).get("exploitability_last"),
        "danger_flags": _dedupe(danger_flags),
        "quality_status": candidate.get("status"),
        "is_training_label": False,
        "label_quality": LABEL_QUALITY,
        "recommendation": "usable_for_candidate_analysis",
        "error": subprocess_result.get("error"),
    }


def build_solver_candidate_job(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    cards = _cards_for(row)
    hero_position_model, decision_context_type = _context_fields(row["context"])
    built = build_hero_oriented_solver_job(
        solver_job_id=_job_id(row),
        source_snapshot_id=f"solver_snapshot_{_job_id(row)}",
        created_at="2026-05-26T00:00:00+00:00",
        source_type="synthetic",
        units="bb",
        street=STREET,
        hero_hand=cards[:2],
        villain_hand=cards[2:4],
        board=cards[4:],
        pot=row["pot"],
        to_call=row["to_call"],
        stack=row["stack"],
        bet_sizes=row["bet_sizes"],
        iterations=row["iterations"],
        timeout_s=timeout_s,
        backend=backend,
        hero_position_model=hero_position_model,
        decision_context_type=decision_context_type,
        root_must_be_hero=True,
    )
    if built["status"] != "ok":
        raise ValueError(f"solver_candidate_job_build_failed:{built['error']}")
    job = dict(built["job"])
    job["generation_profile"] = row["context"]
    job["generation_seed"] = row["seed"]
    job["generation_index"] = row["group_index"]
    job["solver_scenario"] = row["scenario"]
    return job


def build_solver_candidate_report(
    records: list[dict[str, Any]],
    export_summary: dict[str, Any],
    *,
    stage_name: str,
    requested_solves: int,
    output_dir: Path,
    duration_ms: float,
) -> dict[str, Any]:
    root_errors = count_root_player_not_hero(records)
    class_distribution = {
        label: int(export_summary.get("normalized_label_distribution", {}).get(label, 0))
        for label in ("CHECK", "FOLD", "RAISE")
    }
    critical_warnings = critical_warning_list(
        export_summary=export_summary,
        root_player_not_hero_errors=root_errors,
        class_distribution=class_distribution,
    )
    return {
        "status": "ok" if not critical_warnings else "failed",
        "stage": stage_name,
        "requested_solves": requested_solves,
        "total_spots_generated": len(records),
        "usable_rows": export_summary.get("candidates_exported", 0),
        "class_distribution": class_distribution,
        "label_source_distribution": export_summary.get("label_source_counts", {}),
        "rejected_spots": export_summary.get("excluded_count", 0),
        "rejection_reasons": export_summary.get("exclusion_reasons", {}),
        "root_player_not_hero_errors": root_errors,
        "solver_status_distribution": dict(sorted(Counter(str(row.get("solver_status")) for row in records).items())),
        "dominant_action_distribution": dict(sorted(Counter(str(row.get("dominant_action")) for row in records if row.get("dominant_action")).items())),
        "warning_count": len(export_summary.get("warnings", [])),
        "warnings": export_summary.get("warnings", []),
        "critical_warnings": critical_warnings,
        "not_gto": True,
        "not_for_production": True,
        "duration_ms": duration_ms,
        "outputs": {
            "candidate_sensitivity_results": str(output_dir / "candidate_sensitivity_results.jsonl"),
            "candidates_csv": str(output_dir / "candidates.csv"),
            "candidates_jsonl": str(output_dir / "candidates.jsonl"),
            "dataset_report_solver": str(output_dir / "dataset_report_solver.json"),
        },
    }


def critical_warning_list(
    *,
    export_summary: dict[str, Any],
    root_player_not_hero_errors: int,
    class_distribution: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    if root_player_not_hero_errors:
        warnings.append("root_player_not_hero")
    if export_summary.get("status") != "ok":
        warnings.append("export_failed")
    unknown_warnings = sorted(set(export_summary.get("warnings", [])) - EXPECTED_WARNINGS)
    warnings.extend(f"unexpected_export_warning:{warning}" for warning in unknown_warnings)
    usable_rows = int(export_summary.get("candidates_exported", 0) or 0)
    if usable_rows <= 0:
        warnings.append("no_usable_rows")
    counts = [count for count in class_distribution.values() if count > 0]
    if len(counts) < 3:
        warnings.append("missing_check_fold_raise_class")
    elif min(counts) / max(counts) < 0.75:
        warnings.append("class_distribution_imbalanced")
    return warnings


def count_root_player_not_hero(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        error = str(record.get("error") or "").lower()
        flags = " ".join(str(flag).lower() for flag in record.get("danger_flags") or [])
        if record.get("root_matches_hero") is False or record.get("root_player_role") == "villain":
            count += 1
        elif "root_player_not_hero" in error or "root_player_not_hero" in flags:
            count += 1
    return count


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _cards_for(row: dict[str, Any]) -> list[str]:
    import hashlib
    import random

    material = f"{row['seed']}:{row['context']}:{row['scenario']}:{row['group_index']}".encode("utf-8")
    rng = random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16))
    return draw_cards(rng, 4 + BOARD_COUNT)


def _context_fields(context: str) -> tuple[str, str]:
    if context == "hero_oop_check_or_bet":
        return "OOP", "hero_check_or_bet"
    if context == "hero_ip_facing_bet":
        return "IP", "hero_facing_bet"
    if context == "hero_oop_facing_bet":
        return "OOP", "hero_facing_bet"
    raise ValueError(f"unsupported_context:{context}")


def _job_id(row: dict[str, Any]) -> str:
    return f"solver_{row['context']}_{row['scenario']}_it{row['iterations']}"


def _stage_name(solve_count: int) -> str:
    for name, target in STAGES.items():
        if target == solve_count:
            return name
    return f"custom_{solve_count}"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/readiness/bootstrap_candidate_dataset_solver")
    parser.add_argument("--max-stage", choices=STAGE_ORDER, default="large")
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--backend", choices=["rust", "python"], default="rust")
    parser.add_argument("--min-dominant-frequency", type=float, default=0.70)
    parser.add_argument("--min-usable-rows", type=int, default=500)
    parser.add_argument("--class-floor", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_bootstrap_solver_candidates(
        output_dir=args.output_dir,
        max_stage=args.max_stage,
        seed=args.seed,
        timeout_s=args.timeout_s,
        backend=args.backend,
        min_dominant_frequency=args.min_dominant_frequency,
        min_usable_rows=args.min_usable_rows,
        class_floor=args.class_floor,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
