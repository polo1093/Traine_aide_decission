"""Generate curated solver-backed RAISE candidates for bootstrap audits.

The output is still an inspection artifact, not GTO data and not production
training data. It exists to reduce the v4 weak-rule dependency for RAISE before
any CALL work is attempted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


LABEL_QUALITY = "solver_candidate_untrusted"
ITERATIONS = (25, 100)
DEFAULT_OUTPUT_DIR = "outputs/readiness/bootstrap_candidate_dataset_v4_raise"
DEFAULT_BASE_V4_RESULTS = "outputs/readiness/bootstrap_candidate_dataset_v4/candidate_sensitivity_results.jsonl"


CURATED_RAISE_GROUPS = (
    {"context": "hero_ip_facing_bet", "pot": 360.0, "to_call": 198.0, "stack": 400.0, "bet_size": 0.10},
    {"context": "hero_oop_facing_bet", "pot": 360.0, "to_call": 198.0, "stack": 400.0, "bet_size": 0.10},
    {"context": "hero_ip_facing_bet", "pot": 330.0, "to_call": 198.0, "stack": 400.0, "bet_size": 0.10},
    {"context": "hero_oop_facing_bet", "pot": 330.0, "to_call": 198.0, "stack": 400.0, "bet_size": 0.10},
    {"context": "hero_ip_facing_bet", "pot": 360.0, "to_call": 180.0, "stack": 400.0, "bet_size": 0.12},
    {"context": "hero_oop_facing_bet", "pot": 360.0, "to_call": 180.0, "stack": 400.0, "bet_size": 0.12},
    {"context": "hero_ip_facing_bet", "pot": 300.0, "to_call": 180.0, "stack": 400.0, "bet_size": 0.08},
    {"context": "hero_oop_facing_bet", "pot": 300.0, "to_call": 180.0, "stack": 400.0, "bet_size": 0.08},
    {"context": "hero_ip_facing_bet", "pot": 360.0, "to_call": 216.0, "stack": 450.0, "bet_size": 0.15},
    {"context": "hero_oop_facing_bet", "pot": 360.0, "to_call": 216.0, "stack": 450.0, "bet_size": 0.15},
    {"context": "hero_ip_facing_bet", "pot": 330.0, "to_call": 198.0, "stack": 450.0, "bet_size": 0.08},
    {"context": "hero_oop_facing_bet", "pot": 330.0, "to_call": 198.0, "stack": 450.0, "bet_size": 0.08},
)

CARDS = {
    "hero_hand": ["Ah", "As"],
    "villain_hand": ["8c", "3d"],
    "board": ["Ac", "Kd", "7s", "2h", "2c"],
}


def generate_raise_solver_candidates(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_v4_results: str | Path = DEFAULT_BASE_V4_RESULTS,
    timeout_s: float = 5.0,
    backend: str = "rust",
    min_dominant_frequency: float = 0.70,
    min_usable_rows: int = 500,
    class_floor: int = 50,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raise_results = output_path / "raise_solver_candidates.jsonl"
    combined_results = output_path / "candidate_sensitivity_results_with_raise.jsonl"
    candidates_jsonl = output_path / "candidates.jsonl"
    candidates_csv = output_path / "candidates.csv"
    report_json = output_path / "raise_candidate_report.json"

    started = time.perf_counter()
    records = []
    with raise_results.open("w", encoding="utf-8", newline="\n") as handle:
        for row in build_raise_plan():
            record = run_raise_row(row, timeout_s=timeout_s, backend=backend)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    combine_jsonl([base_v4_results, raise_results], combined_results)
    export_summary = export_candidate_dataset(
        [combined_results],
        output_jsonl=candidates_jsonl,
        output_csv=candidates_csv,
        include_weak_rules=True,
        min_usable_rows=min_usable_rows,
        class_floor=class_floor,
        min_dominant_frequency=min_dominant_frequency,
        balance_weak_rules=True,
    )
    report = build_report(
        records,
        export_summary,
        output_dir=output_path,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_raise_plan() -> list[dict[str, Any]]:
    plan = []
    for index, config in enumerate(CURATED_RAISE_GROUPS):
        for iterations in ITERATIONS:
            plan.append({**config, "scenario": f"curated_raise_{index:03d}", "iterations": iterations})
    return plan


def run_raise_row(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    job = build_raise_job(row, timeout_s=timeout_s, backend=backend)
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
        "street": "RIVER",
        "hero_cards": job.get("hero_hand"),
        "villain_hand": job.get("villain_hand"),
        "board_cards": job.get("board"),
        "stack": row["stack"],
        "pot": row["pot"],
        "to_call": row["to_call"],
        "to_call_ratio": round(float(row["to_call"]) / float(row["pot"]), 6),
        "spr": calculate_spr(row["stack"], row["pot"]),
        "bet_size_fractions": [row["bet_size"]],
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


def build_raise_job(row: dict[str, Any], *, timeout_s: float, backend: str) -> dict[str, Any]:
    position, decision_type = context_fields(row["context"])
    built = build_hero_oriented_solver_job(
        solver_job_id=f"curated_raise_{row['context']}_{row['scenario']}_it{row['iterations']}",
        source_snapshot_id=f"curated_raise_snapshot_{row['context']}_{row['scenario']}_it{row['iterations']}",
        created_at="2026-05-26T00:00:00+00:00",
        source_type="synthetic",
        units="bb",
        street="RIVER",
        hero_hand=CARDS["hero_hand"],
        villain_hand=CARDS["villain_hand"],
        board=CARDS["board"],
        pot=row["pot"],
        to_call=row["to_call"],
        stack=row["stack"],
        bet_sizes=[row["bet_size"]],
        iterations=row["iterations"],
        timeout_s=timeout_s,
        backend=backend,
        hero_position_model=position,
        decision_context_type=decision_type,
        root_must_be_hero=True,
    )
    if built["status"] != "ok":
        raise ValueError(f"curated_raise_build_failed:{built['error']}")
    job = dict(built["job"])
    job["generation_profile"] = row["context"]
    job["curated_raise_candidate"] = True
    return job


def build_report(records: list[dict[str, Any]], export_summary: dict[str, Any], *, output_dir: Path, duration_ms: float) -> dict[str, Any]:
    usable_raise_records = [
        row
        for row in records
        if row.get("solver_status") == "ok"
        and row.get("root_matches_hero") is True
        and str(row.get("dominant_action") or "").startswith("RAISE_")
        and float(row.get("dominant_frequency") or 0) >= 0.70
        and not row.get("danger_flags")
    ]
    root_not_hero = sum(1 for row in records if row.get("root_matches_hero") is False or row.get("root_player_role") == "villain")
    label_sources = export_summary.get("label_source_counts", {})
    label_distribution = export_summary.get("normalized_label_distribution", {})
    return {
        "status": "ok" if usable_raise_records and root_not_hero == 0 else "failed",
        "not_gto": True,
        "not_for_production": True,
        "raise_solver_records_total": len(records),
        "usable_raise_solver_records": len(usable_raise_records),
        "root_player_not_hero_errors": root_not_hero,
        "dominant_actions": _counts(row.get("dominant_action") for row in records),
        "label_source_distribution": label_sources,
        "label_distribution": label_distribution,
        "rejection_reasons": export_summary.get("exclusion_reasons", {}),
        "duration_ms": duration_ms,
        "outputs": {
            "raise_solver_candidates": str(output_dir / "raise_solver_candidates.jsonl"),
            "combined_candidate_sensitivity_results": str(output_dir / "candidate_sensitivity_results_with_raise.jsonl"),
            "candidates_csv": str(output_dir / "candidates.csv"),
            "candidates_jsonl": str(output_dir / "candidates.jsonl"),
            "raise_candidate_report": str(output_dir / "raise_candidate_report.json"),
        },
    }


def combine_jsonl(inputs: list[str | Path], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as writer:
        for raw in inputs:
            path = Path(raw)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if line.strip():
                        writer.write(line.rstrip("\n"))
                        writer.write("\n")


def context_fields(context: str) -> tuple[str, str]:
    if context == "hero_ip_facing_bet":
        return "IP", "hero_facing_bet"
    if context == "hero_oop_facing_bet":
        return "OOP", "hero_facing_bet"
    raise ValueError(f"unsupported_raise_context:{context}")


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-v4-results", default=DEFAULT_BASE_V4_RESULTS)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--backend", choices=["rust", "python"], default="rust")
    parser.add_argument("--min-dominant-frequency", type=float, default=0.70)
    parser.add_argument("--min-usable-rows", type=int, default=500)
    parser.add_argument("--class-floor", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_raise_solver_candidates(
        output_dir=args.output_dir,
        base_v4_results=args.base_v4_results,
        timeout_s=args.timeout_s,
        backend=args.backend,
        min_dominant_frequency=args.min_dominant_frequency,
        min_usable_rows=args.min_usable_rows,
        class_floor=args.class_floor,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
