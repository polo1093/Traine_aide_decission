"""Bounded sensitivity analysis for hero-oriented solver action candidates.

This script writes solver-run inspection records only. It never writes
``training_label`` or ``gto_label`` and every record keeps
``is_training_label`` false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver_jobs.action_candidate import build_solver_action_candidate
from solver_jobs.candidate_quality import evaluate_candidate_quality
from solver_jobs.hero_oriented_builder import build_hero_oriented_solver_job, validate_hero_root_alignment
from solver_jobs.strategy_extractor import extract_root_strategy
from solver_jobs.subprocess_runner import run_solver_job_subprocess
from synthetic.deck import draw_cards


CONTEXTS = ("hero_oop_check_or_bet", "hero_ip_facing_bet", "hero_oop_facing_bet")
ITERATIONS = (25, 50, 100)
MAX_DEFAULT_SOLVES = 30
MAX_HARD_SOLVES = 50
STREET = "RIVER"
BOARD_COUNT = 5
LABEL_QUALITY = "solver_candidate_untrusted"


def calculate_spr(stack: float, pot: float) -> float | None:
    """Return stack-to-pot ratio, rounded for stable JSON output."""

    try:
        pot_value = float(pot)
        if pot_value <= 0:
            return None
        return round(float(stack) / pot_value, 6)
    except (TypeError, ValueError):
        return None


def build_default_plan(*, iterations: tuple[int, ...] = ITERATIONS, seed: int = 8800) -> list[dict[str, Any]]:
    """Return a curated <=30-solve plan covering the requested dimensions."""

    base_configs = [
        {"scenario": "high_spr_small_call_b33", "stack": 2000.0, "pot": 100.0, "to_call_ratio": 0.10, "bet_sizes": [0.33]},
        {"scenario": "mid_spr_medium_call_b66", "stack": 1000.0, "pot": 300.0, "to_call_ratio": 0.33, "bet_sizes": [0.66]},
        {"scenario": "low_spr_large_call_b100", "stack": 200.0, "pot": 800.0, "to_call_ratio": 0.75, "bet_sizes": [1.0]},
    ]
    extra_probe = {"scenario": "probe_stack500_medium_call_b50", "stack": 500.0, "pot": 300.0, "to_call_ratio": 0.33, "bet_sizes": [0.5]}

    plan: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for config in base_configs:
            for iteration_count in iterations:
                plan.append(_plan_row(context=context, config=config, iterations=iteration_count, seed=seed))
    for iteration_count in iterations:
        plan.append(_plan_row(context="hero_ip_facing_bet", config=extra_probe, iterations=iteration_count, seed=seed))
    return plan


def run_sensitivity_analysis(
    *,
    output_jsonl: str | Path,
    summary_json: str | Path,
    max_total_solves: int = MAX_DEFAULT_SOLVES,
    allow_large_run: bool = False,
    timeout_s: float = 5.0,
    backend: str = "rust",
    seed: int = 8800,
) -> dict[str, Any]:
    """Run the bounded sensitivity plan and persist JSONL + summary JSON."""

    plan = build_default_plan(seed=seed)
    _validate_solve_count(len(plan), max_total_solves=max_total_solves, allow_large_run=allow_large_run)

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan:
            record = run_one_sensitivity_row(row, timeout_s=timeout_s, backend=backend)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    summary = summarize_sensitivity_records(records)
    summary["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_one_sensitivity_row(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    """Run one solve and return a JSON-serializable sensitivity record."""

    job = _build_job(row, timeout_s=timeout_s, backend=backend)
    root_validation = validate_hero_root_alignment(job)
    subprocess_result = run_solver_job_subprocess(job, timeout_s=timeout_s)
    solver_result = subprocess_result.get("solver_result") or {}
    extraction_input = {
        "solver_job_id": job["solver_job_id"],
        "solver_status": subprocess_result.get("solver_status"),
        "solver_result": solver_result,
        "quality": subprocess_result.get("quality"),
    }
    strategy = extract_root_strategy(extraction_input)
    candidate = build_solver_action_candidate(extraction_input)
    output = solver_result.get("output") if isinstance(solver_result, dict) else {}
    if not isinstance(output, dict):
        output = {}
    dominant_action = strategy.get("dominant_action")
    dominant_frequency = strategy.get("dominant_action_frequency")
    danger_flags = []
    if dominant_action == "ALL_IN":
        danger_flags.append("extreme_action_all_in")
    if strategy.get("status") != "ok":
        danger_flags.append(strategy.get("error") or "strategy_not_available")
    if subprocess_result.get("solver_status") != "ok":
        danger_flags.append(subprocess_result.get("solver_status") or "solver_not_ok")

    return {
        "record_type": "candidate_sensitivity_result",
        "context": row["context"],
        "scenario": row["scenario"],
        "street": STREET,
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
        "legal_actions": root_validation.get("legal_action_labels", []),
        "action_frequencies": strategy.get("action_frequencies", {}),
        "dominant_action": dominant_action,
        "dominant_frequency": dominant_frequency,
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


def summarize_sensitivity_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize sensitivity records by context and scenario."""

    context_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            warnings.append("non_mapping_record_skipped")
            continue
        context = str(record.get("context") or "unknown")
        scenario = str(record.get("scenario") or "unknown")
        context_rows[context].append(record)
        group_rows[(context, scenario)].append(record)

    groups = []
    for (context, scenario), rows in sorted(group_rows.items()):
        run_payloads = [_quality_input_from_record(row) for row in rows]
        quality = evaluate_candidate_quality(run_payloads)
        actions = [row.get("dominant_action") for row in rows]
        groups.append(
            {
                "context": context,
                "scenario": scenario,
                "solve_count": len(rows),
                "actions_by_iteration": [
                    {
                        "iterations": row.get("iterations"),
                        "dominant_action": row.get("dominant_action"),
                        "dominant_frequency": row.get("dominant_frequency"),
                    }
                    for row in sorted(rows, key=lambda item: int(item.get("iterations") or 0))
                ],
                "stable_action": quality["stable_action"],
                "quality_status": quality["status"],
                "candidate_action": quality["candidate_action"],
                "dominant_frequency_avg": quality["dominant_frequency_avg"],
                "dominant_frequency_min": quality["dominant_frequency_min"],
                "danger_flags": quality["danger_flags"],
                "exclusion_reason": quality["exclusion_reason"],
                "is_training_label": False,
                "label_quality": LABEL_QUALITY,
                "recommendation": quality["recommendation"],
            }
        )

    by_context = {}
    for context, rows in sorted(context_rows.items()):
        action_counts = Counter(str(row.get("dominant_action")) for row in rows if row.get("dominant_action"))
        all_in_rows = [row for row in rows if row.get("dominant_action") == "ALL_IN"]
        unstable_groups = [
            group for group in groups if group["context"] == context and group["stable_action"] is False
        ]
        by_context[context] = {
            "solve_count": len(rows),
            "dominant_action_counts": dict(sorted(action_counts.items())),
            "all_in_count": len(all_in_rows),
            "all_in_rate": round(len(all_in_rows) / len(rows), 6) if rows else 0.0,
            "unstable_group_count": len(unstable_groups),
            "danger_flags": _dedupe(
                [
                    flag
                    for row in rows
                    for flag in list(row.get("danger_flags") or [])
                ]
            ),
            "is_training_label": False,
            "label_quality": LABEL_QUALITY,
            "recommendation": "usable_for_candidate_analysis",
        }

    return {
        "record_type": "candidate_sensitivity_summary",
        "status": "ok",
        "total_solves": len(records),
        "by_context": by_context,
        "groups": groups,
        "warnings": _dedupe(warnings),
        "is_training_label": False,
        "label_quality": LABEL_QUALITY,
        "recommendation": "usable_for_candidate_analysis_only_not_training",
    }


def load_jsonl_safely(path: str | Path) -> dict[str, Any]:
    """Load partial JSONL without leaking raw exceptions."""

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                text = line.strip()
                if not text:
                    warnings.append(f"line_{line_index}:empty")
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    warnings.append(f"line_{line_index}:invalid_json")
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    warnings.append(f"line_{line_index}:non_object")
        return {"status": "ok", "records": records, "warnings": warnings}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "records": records, "warnings": [f"{type(exc).__name__}:{exc}"]}


def _plan_row(*, context: str, config: dict[str, Any], iterations: int, seed: int) -> dict[str, Any]:
    to_call = 0.0 if context == "hero_oop_check_or_bet" else round(float(config["pot"]) * float(config["to_call_ratio"]), 6)
    return {
        "context": context,
        "scenario": str(config["scenario"]),
        "stack": float(config["stack"]),
        "pot": float(config["pot"]),
        "to_call": to_call,
        "to_call_ratio": 0.0 if context == "hero_oop_check_or_bet" else float(config["to_call_ratio"]),
        "bet_sizes": [float(value) for value in config["bet_sizes"]],
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _build_job(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    hero_hand, villain_hand, board = _cards_for(row)
    hero_position_model, decision_context_type = _context_fields(row["context"])
    built = build_hero_oriented_solver_job(
        solver_job_id=_job_id(row),
        source_snapshot_id=f"sensitivity_snapshot_{_job_id(row)}",
        created_at="2026-05-25T00:00:00+00:00",
        source_type="synthetic",
        units="bb",
        street=STREET,
        hero_hand=hero_hand,
        villain_hand=villain_hand,
        board=board,
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
        raise ValueError(f"sensitivity_job_build_failed:{built['error']}")
    job = dict(built["job"])
    job["generation_profile"] = row["context"]
    job["sensitivity_scenario"] = row["scenario"]
    return job


def _cards_for(row: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    material = f"{row['seed']}:{row['context']}:{row['scenario']}".encode("utf-8")
    rng = random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16))
    cards = draw_cards(rng, 4 + BOARD_COUNT)
    return cards[:2], cards[2:4], cards[4:]


def _context_fields(context: str) -> tuple[str, str]:
    if context == "hero_oop_check_or_bet":
        return "OOP", "hero_check_or_bet"
    if context == "hero_ip_facing_bet":
        return "IP", "hero_facing_bet"
    if context == "hero_oop_facing_bet":
        return "OOP", "hero_facing_bet"
    raise ValueError(f"unsupported_context:{context}")


def _job_id(row: dict[str, Any]) -> str:
    return (
        f"sensitivity_{row['context']}_{row['scenario']}_"
        f"it{row['iterations']}"
    )


def _quality_input_from_record(row: dict[str, Any]) -> dict[str, Any]:
    frequency = row.get("dominant_frequency")
    action = row.get("dominant_action")
    return {
        "solver_job_id": row.get("solver_job_id"),
        "solver_status": row.get("solver_status"),
        "root_matches_hero": row.get("root_matches_hero"),
        "root_player_role": row.get("root_player_role"),
        "action_candidate": {
            "status": row.get("candidate_status") or "failed",
            "solver_job_id": row.get("solver_job_id"),
            "candidate_action": action,
            "candidate_frequency": 0.0 if frequency is None else frequency,
            "candidate_confidence": "high",
            "is_training_label": False,
            "label_quality": LABEL_QUALITY,
            "exclusion_reason": row.get("candidate_exclusion_reason"),
            "warnings": [],
        },
        "quality": {
            "iterations": row.get("iterations"),
            "exploitability_last": _exploitability_from_record(row),
            "is_label_candidate": False,
        },
        "error": row.get("error"),
    }


def _exploitability_from_record(row: dict[str, Any]) -> float | None:
    value = row.get("exploitability_last")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _validate_solve_count(count: int, *, max_total_solves: int, allow_large_run: bool) -> None:
    if count > MAX_HARD_SOLVES and not allow_large_run:
        raise ValueError(f"max_total_solves_exceeds_hard_limit:{MAX_HARD_SOLVES}")
    if count > max_total_solves and not allow_large_run:
        raise ValueError(f"max_total_solves_exceeds_limit:{max_total_solves}")


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(str(value))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", default="outputs/candidate_sensitivity/results.jsonl")
    parser.add_argument("--summary-json", default="outputs/candidate_sensitivity/summary.json")
    parser.add_argument("--max-total-solves", type=int, default=MAX_DEFAULT_SOLVES)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--backend", choices=["rust", "python"], default="rust")
    parser.add_argument("--seed", type=int, default=8800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sensitivity_analysis(
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        max_total_solves=args.max_total_solves,
        allow_large_run=args.allow_large_run,
        timeout_s=args.timeout_s,
        backend=args.backend,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
