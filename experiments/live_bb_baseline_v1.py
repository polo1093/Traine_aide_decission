"""Train a four-class live/BB baseline from dist/ml_dataset_export_v2."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver_jobs.action_candidate import build_solver_action_candidate
from solver_jobs.hero_oriented_builder import build_hero_oriented_solver_job
from solver_jobs.strategy_extractor import extract_root_strategy
from solver_jobs.subprocess_runner import run_solver_job_subprocess
from solvers.poker_solver_adapter import compute_equity_hand_vs_hand
from synthetic.deck import draw_cards


DEFAULT_INPUT = Path("dist/ml_dataset_export_v2/training_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/readiness/live_bb_baseline_v1")
ALLOWED_LABELS = ("CHECK", "FOLD", "CALL", "RAISE")
LABEL_ALIASES = {"COL": "CALL", "CALL": "CALL", "RES": "RAISE", "BET": "RAISE", "MISE": "RAISE", "RELANCE": "RAISE"}
NUMERIC_FEATURES = (
    "features.pot_bb",
    "features.to_call_bb",
    "features.ev_bb",
    "features.call_max_bb",
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_required",
    "features.equity_gap",
    "features.hero_stack_bb",
    "features.effective_stack_bb",
    "features.stack_to_pot_ratio",
    "features.to_call_pot_ratio",
    "features.has_check",
    "features.has_call",
    "features.has_raise",
    "features.active_opponents",
)
CATEGORICAL_FEATURES = ("metadata.street", "features.hero_position")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_AUDIT_COLUMNS = (
    "features.pot",
    "features.to_call",
    "features.ev",
    "features.call_max",
    "features.amount_unit_value",
    "features.hero_cards",
    "features.board_cards",
    "features.players",
    "features.opponent_profiles",
)
GENERATION_METHODS = ("real_imported", "solver_generated", "resampled_jitter", "rule_generated")
POSTFLOP_STREETS = {"FLOP": 3, "TURN": 4, "RIVER": 5}
LIVE_SOLVER_STREETS = {"RIVER": 5}
DEFAULT_MIN_SOLVER_ROWS = 25
MAX_RESAMPLED_SHARE_WITHOUT_WARNING = 0.80
MAX_INFLATION_RATIO_WITHOUT_WARNING = 20.0
OVERFIT_TARGET_ROW_THRESHOLD = 7000
LEAKAGE_PREFIXES = ("labels.", "debug.", "audit.", "quality_flags.")
LEAKAGE_NAMES = {"metadata.label_source", "label_source", "bootstrap_label", "normalized_label"}


def run_live_bb_baseline_v1(
    *,
    input_jsonl: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    random_seed: int = 17,
    target_rows: int | None = None,
    use_solver: bool = False,
    solver_rows: int = 0,
    solver_timeout_s: float = 2.0,
    solver_iterations: int = 25,
    solver_backend: str = "rust",
    strict_solver: bool = False,
    min_solver_rows: int = DEFAULT_MIN_SOLVER_ROWS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_rows = load_jsonl(Path(input_jsonl))
    rebuilder = load_feature_rebuilder(Path(input_jsonl).parent / "feature_rebuilder.py")
    candidates, export_report = build_candidates(source_rows, rebuilder=rebuilder)
    solver_candidates, solver_report = generate_solver_candidates(
        source_rows,
        output_path=output_path,
        rebuilder=rebuilder,
        requested_rows=solver_rows if use_solver else 0,
        random_seed=random_seed,
        timeout_s=solver_timeout_s,
        iterations=solver_iterations,
        backend=solver_backend,
    )
    if use_solver and strict_solver and len(solver_candidates) < min_solver_rows:
        failure = build_strict_solver_failure(
            solver_report=solver_report,
            solver_generated_rows=len(solver_candidates),
            min_solver_rows=min_solver_rows,
        )
        write_json(failure, output_path / "solver_generation_failure_report.json")
        raise RuntimeError(failure["error"])
    candidates = [*candidates, *solver_candidates]
    augmentation = augment_candidates(candidates, target_rows=target_rows, random_seed=random_seed)
    candidates = augmentation["rows"]
    candidates_csv = output_path / "candidates.csv"
    write_candidates_csv(candidates, candidates_csv)

    training = train_model(candidates, output_path=output_path, random_seed=random_seed)
    diagnostics = write_diagnostics(candidates, output_path=output_path, model_type=training["model_type"], random_seed=random_seed)
    prediction = predict_first_imported_line(model_dir=output_path, rows=candidates)

    source_unique = source_snapshot_id_unique_count(candidates)
    inflation_ratio = round(len(candidates) / source_unique, 6) if source_unique else 0.0
    imported_rows = count_generation_method(candidates, "real_imported")
    solver_generated_rows = count_generation_method(candidates, "solver_generated")
    resampled_rows = count_generation_method(candidates, "resampled_jitter")
    rule_generated_rows = count_generation_method(candidates, "rule_generated")
    generation_mode = determine_generation_mode(
        solver_generated_rows=solver_generated_rows,
        resampled_rows=resampled_rows,
    )
    warnings = generation_warnings(
        solver_report=solver_report,
        final_rows=len(candidates),
        target_rows=target_rows,
        use_solver=use_solver,
        solver_generated_rows=solver_generated_rows,
        min_solver_rows=min_solver_rows,
        resampled_rows=resampled_rows,
        inflation_ratio=inflation_ratio,
    )
    report = {
        **training,
        "input_jsonl": str(input_jsonl),
        "candidates_csv": str(candidates_csv),
        "rows_total": len(source_rows),
        "rows_used": len(candidates),
        "generation_mode": generation_mode,
        "solver_connected_real_generation": bool(solver_generated_rows > 0),
        "imported_rows": imported_rows,
        "solver_attempted": solver_report["solver_attempted"],
        "solver_success": solver_report["solver_success"],
        "solver_failed": solver_report["solver_failed"],
        "solver_generated_rows": solver_generated_rows,
        "min_solver_rows": int(min_solver_rows),
        "all_in_excluded": solver_report["all_in_excluded"],
        "resampled_rows": resampled_rows,
        "rule_generated_rows": rule_generated_rows,
        "final_rows": len(candidates),
        "source_snapshot_id_unique_count": source_unique,
        "inflation_ratio": inflation_ratio,
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in candidates).items())),
        "street_distribution": dict(sorted(Counter(row["metadata.street"] for row in candidates).items())),
        "source_distribution": dict(sorted(Counter(row["metadata.label_source"] for row in candidates).items())),
        "generation_method_distribution": dict(sorted(Counter(row.get("generation_method", "unknown") for row in candidates).items())),
        "augmentation": {
            "requested_target_rows": target_rows,
            "applied": augmentation["applied"],
            "source_rows": augmentation["source_rows"],
            "output_rows": len(candidates),
            "mode": augmentation["mode"],
        },
        "solver_enrichment": export_report["solver_enrichment"],
        "solver_generation": solver_report,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "call_diagnostic": call_diagnostic(training),
        "offline_imported_prediction": prediction,
        "bot_live_connection": "not_modified",
        "not_for_production": True,
        "supported_classes": list(ALLOWED_LABELS),
    }
    write_json(report, output_path / "training_report.json")
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                value = json.loads(text)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def load_feature_rebuilder(path: Path) -> Any:
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("ml_dataset_export_v2_feature_rebuilder", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_candidates(rows: Sequence[Mapping[str, Any]], *, rebuilder: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    enrichment_counter = Counter()
    for row in rows:
        if not is_trainable_row(row, rebuilder=rebuilder):
            continue
        label = normalize_label((row.get("labels") or {}).get("final_action") or (row.get("labels") or {}).get("legacy_action"))
        if label not in ALLOWED_LABELS:
            continue
        rebuilt = rebuild_features(row, rebuilder=rebuilder)
        features = dict(row.get("features") or {})
        merged_features = {**features, **{key: value for key, value in rebuilt.items() if value is not None}}
        enrichment_counter[solver_enrichment_status(features, rebuilt)] += 1
        candidate = flatten_candidate(row, merged_features, label)
        candidate["generation_method"] = "real_imported"
        candidates.append(candidate)
    return candidates, {
        "solver_enrichment": {
            "status_counts": dict(sorted(enrichment_counter.items())),
            "mode": "stored_equity_plus_feature_rebuilder",
            "note": "Uses stored equity/EV/call_max from ml_dataset_export_v2 and rebuilds deterministic derived features; recompute hook remains in feature_rebuilder.py when project equity deps are available.",
        }
    }


def generate_solver_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    rebuilder: Any,
    requested_rows: int,
    random_seed: int,
    timeout_s: float,
    iterations: int,
    backend: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = output_path / "solver_generation_results.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_rows <= 0:
        report_path.write_text("", encoding="utf-8")
        return [], solver_generation_report(
            requested_rows=0,
            attempted=0,
            success=0,
            failed=0,
            all_in_excluded=0,
            ineligible_imported_rows=count_solver_ineligible_rows(rows, rebuilder=rebuilder),
            path=report_path,
            warnings=[],
            last_error=None,
        )

    eligible = solver_eligible_rows(rows, rebuilder=rebuilder)
    ineligible_count = len(rows) - len(eligible)
    if not eligible:
        warning = "solver_requested_but_no_postflop_rows_with_amounts"
        report_path.write_text("", encoding="utf-8")
        return [], solver_generation_report(
            requested_rows=requested_rows,
            attempted=0,
            success=0,
            failed=0,
            all_in_excluded=0,
            ineligible_imported_rows=ineligible_count,
            path=report_path,
            warnings=[warning],
            last_error=warning,
        )

    rng = random.Random(random_seed)
    candidates: list[dict[str, Any]] = []
    attempted = 0
    failed = 0
    all_in_excluded = 0
    last_error = None
    max_attempts = max(requested_rows * 4, requested_rows + 8)

    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        for attempt_index in range(max_attempts):
            if len(candidates) >= requested_rows:
                break
            source = eligible[attempt_index % len(eligible)]
            attempted += 1
            record, candidate = run_one_live_solver_generation(
                source,
                attempt_index=attempt_index,
                rng=rng,
                timeout_s=timeout_s,
                iterations=iterations,
                backend=backend,
                random_seed=random_seed,
            )
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            if candidate is not None:
                candidates.append(candidate)
                continue
            if record.get("excluded_action") == "ALL_IN":
                all_in_excluded += 1
            else:
                failed += 1
                last_error = record.get("error") or record.get("candidate_exclusion_reason") or record.get("strategy_error")

    warnings = []
    if not candidates:
        warnings.append("SOLVER_REQUESTED_BUT_ZERO_SOLVER_GENERATED_ROWS")
    elif len(candidates) < requested_rows:
        warnings.append("solver_generated_less_than_requested")
    return candidates, solver_generation_report(
        requested_rows=requested_rows,
        attempted=attempted,
        success=len(candidates),
        failed=failed,
        all_in_excluded=all_in_excluded,
        ineligible_imported_rows=ineligible_count,
        path=report_path,
        warnings=warnings,
        last_error=last_error,
    )


def solver_generation_report(
    *,
    requested_rows: int,
    attempted: int,
    success: int,
    failed: int,
    all_in_excluded: int,
    ineligible_imported_rows: int,
    path: Path,
    warnings: list[str],
    last_error: str | None,
) -> dict[str, Any]:
    return {
        "mode": "hero_oriented_real_solver_subprocess" if requested_rows else "disabled",
        "requested_rows": requested_rows,
        "solver_attempted": attempted,
        "solver_success": success,
        "solver_failed": failed,
        "solver_generated_rows": success,
        "all_in_excluded": all_in_excluded,
        "ineligible_imported_rows": ineligible_imported_rows,
        "results_jsonl": str(path),
        "warnings": warnings,
        "warning": warnings[0] if warnings else None,
        "last_error": last_error,
        "note": "For this fast baseline the real solver is restricted to RIVER hero-oriented jobs; PREFLOP/FLOP/TURN rows are counted as ineligible instead of silently resampled as solver output.",
    }


def solver_eligible_rows(rows: Sequence[Mapping[str, Any]], *, rebuilder: Any) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        if not is_trainable_row(row, rebuilder=rebuilder):
            continue
        rebuilt = rebuild_features(row, rebuilder=rebuilder)
        features = {**dict(row.get("features") or {}), **{key: value for key, value in rebuilt.items() if value is not None}}
        street = str((row.get("metadata") or {}).get("street") or features.get("street") or "").upper()
        pot = number_or_none(features.get("pot_bb"))
        if pot is None or pot <= 0 or street not in LIVE_SOLVER_STREETS:
            continue
        source = flatten_candidate(row, features, normalize_label((row.get("labels") or {}).get("final_action") or (row.get("labels") or {}).get("legacy_action")))
        if source["bootstrap_label"] in ALLOWED_LABELS:
            eligible.append(source)
    return eligible


def count_solver_ineligible_rows(rows: Sequence[Mapping[str, Any]], *, rebuilder: Any) -> int:
    return len(rows) - len(solver_eligible_rows(rows, rebuilder=rebuilder))


def run_one_live_solver_generation(
    source: Mapping[str, Any],
    *,
    attempt_index: int,
    rng: random.Random,
    timeout_s: float,
    iterations: int,
    backend: str,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    street = str(source.get("metadata.street") or "FLOP").upper()
    board_count = POSTFLOP_STREETS.get(street, 3)
    cards = draw_cards(rng, 4 + board_count)
    hero_hand = cards[:2]
    villain_hand = cards[2:4]
    board = cards[4:]
    pot = max(1.0, number_or_zero(source.get("features.pot_bb")))
    raw_to_call = max(0.0, number_or_zero(source.get("features.to_call_bb")))
    to_call = min(raw_to_call, max(0.0, pot * 0.8))
    hero_position_model, decision_context_type = solver_context_for(source, to_call=to_call, attempt_index=attempt_index)
    if decision_context_type == "hero_check_or_bet":
        to_call = 0.0
    if decision_context_type == "hero_facing_bet" and to_call <= 0:
        to_call = max(0.5, pot * 0.25)
    pot = max(pot, to_call * 1.5, 1.0)
    pot = round(pot, 2)
    to_call = round(to_call, 2)
    stack = max(number_or_zero(source.get("features.effective_stack_bb")), number_or_zero(source.get("features.hero_stack_bb")), pot + to_call + 5.0)
    stack = round(min(max(stack, pot + to_call + 5.0), 100.0), 2)
    solver_job_id = f"live_bb_solver_{street.lower()}_{random_seed}_{attempt_index:05d}"
    built = build_hero_oriented_solver_job(
        solver_job_id=solver_job_id,
        source_snapshot_id=f"live_bb_solver_source_{source_snapshot_id(source)}",
        created_at="2026-05-27T00:00:00+00:00",
        source_type="synthetic",
        units="bb",
        street=street,
        hero_hand=hero_hand,
        villain_hand=villain_hand,
        board=board,
        pot=pot,
        to_call=to_call,
        stack=stack,
        bet_sizes=[0.33, 0.66],
        iterations=iterations,
        timeout_s=timeout_s,
        backend=backend,
        hero_position_model=hero_position_model,
        decision_context_type=decision_context_type,
        root_must_be_hero=True,
    )
    base_record = {
        "record_type": "live_bb_solver_generation_result",
        "generation_method": "solver_generated",
        "solver_job_id": solver_job_id,
        "source_snapshot_id": source_snapshot_id(source),
        "street": street,
        "pot_bb": pot,
        "to_call_bb": to_call,
        "stack_bb": stack,
        "hero_position": hero_position_model,
        "decision_context_type": decision_context_type,
        "hero_cards": hero_hand,
        "villain_hand": villain_hand,
        "board_cards": board,
    }
    if built.get("status") != "ok":
        return {**base_record, "solver_status": "not_called", "error": built.get("error")}, None

    job = dict(built["job"])
    subprocess_result = run_solver_job_subprocess(job, timeout_s=timeout_s)
    solver_result = subprocess_result.get("solver_result") if isinstance(subprocess_result.get("solver_result"), dict) else {}
    extraction_input = {
        "solver_job_id": solver_job_id,
        "solver_status": subprocess_result.get("solver_status"),
        "solver_result": solver_result,
        "quality": subprocess_result.get("quality"),
    }
    strategy = extract_root_strategy(extraction_input)
    candidate_payload = build_solver_action_candidate(extraction_input)
    dominant_action = strategy.get("dominant_action")
    label = normalize_solver_action(dominant_action)
    record = {
        **base_record,
        "solver_status": subprocess_result.get("solver_status"),
        "error": subprocess_result.get("error"),
        "strategy_status": strategy.get("status"),
        "strategy_error": strategy.get("error"),
        "action_frequencies": strategy.get("action_frequencies"),
        "dominant_action": dominant_action,
        "dominant_frequency": strategy.get("dominant_action_frequency"),
        "candidate_status": candidate_payload.get("status"),
        "candidate_exclusion_reason": candidate_payload.get("exclusion_reason"),
        "normalized_label": label,
    }
    if label == "ALL_IN":
        return {**record, "excluded_action": "ALL_IN"}, None
    if label not in ALLOWED_LABELS or subprocess_result.get("solver_status") != "ok" or candidate_payload.get("status") != "ok":
        return record, None

    equity = solver_equity(hero_hand, villain_hand, board, seed=random_seed + attempt_index)
    generated = solver_candidate_row(
        source=source,
        record=record,
        label=label,
        hero_hand=hero_hand,
        villain_hand=villain_hand,
        board=board,
        pot=pot,
        to_call=to_call,
        stack=stack,
        equity=equity,
    )
    return record, generated


def solver_context_for(source: Mapping[str, Any], *, to_call: float, attempt_index: int) -> tuple[str, str]:
    raw_position = str(source.get("features.hero_position") or "UNKNOWN").upper()
    if to_call > 0 or number_or_zero(source.get("features.has_call")) > 0:
        if raw_position in {"IP", "BTN", "CO", "HJ", "LJ"}:
            return "IP", "hero_facing_bet"
        if raw_position in {"OOP", "SB", "BB", "UTG", "MP"}:
            return "OOP", "hero_facing_bet"
        return ("IP" if attempt_index % 2 else "OOP"), "hero_facing_bet"
    return "OOP", "hero_check_or_bet"


def normalize_solver_action(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"BET", "BET_33", "BET_75", "BET_100", "BET_150", "BET_200", "RAISE", "RAISE_33", "RAISE_75", "RAISE_100", "RAISE_150", "RAISE_200"}:
        return "RAISE"
    if text in {"CHECK_CALL", "CALL"}:
        return "CALL"
    if text == "ALLIN":
        return "ALL_IN"
    return LABEL_ALIASES.get(text, text)


def solver_equity(hero_hand: Sequence[str], villain_hand: Sequence[str], board: Sequence[str], *, seed: int) -> float:
    result = compute_equity_hand_vs_hand(hero_hand, villain_hand, board=board, iterations=500, seed=seed)
    if result.get("status") == "ok":
        output = result.get("output") or {}
        equity = number_or_none(output.get("hero_equity"))
        if equity is not None:
            return max(0.0, min(1.0, equity))
    return 0.5


def solver_candidate_row(
    *,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    label: str,
    hero_hand: Sequence[str],
    villain_hand: Sequence[str],
    board: Sequence[str],
    pot: float,
    to_call: float,
    stack: float,
    equity: float,
) -> dict[str, Any]:
    required = to_call / (pot + to_call) if pot + to_call > 0 else 0.0
    ev = equity * (pot + to_call) - to_call
    call_max = (equity * pot) / (1.0 - equity) if equity < 1.0 else 500.0
    legal_actions = set(str(action) for action in (record.get("action_frequencies") or {}))
    has_check = "CHECK" in legal_actions
    has_call = "CALL" in legal_actions or "CHECK_CALL" in legal_actions
    has_raise = any(action.startswith("BET") or action.startswith("RAISE") for action in legal_actions)
    return {
        "snapshot_id": f"{record['solver_job_id']}:candidate",
        "generation_method": "solver_generated",
        "bootstrap_label": label,
        "metadata.street": str(record.get("street") or "UNKNOWN").upper(),
        "metadata.label_source": "solver_generated",
        "labels.final_action": label,
        "labels.legacy_action": None,
        "quality_flags.usable_for_training": True,
        "quality_flags.amount_unit_missing": False,
        "features.pot": pot,
        "features.to_call": to_call,
        "features.ev": ev,
        "features.call_max": call_max,
        "features.amount_unit_value": 1.0,
        "features.hero_cards": json_safe(list(hero_hand)),
        "features.board_cards": json_safe(list(board)),
        "features.players": json_safe(
            [
                {"name": "Flael", "active": True, "stack_bb": stack},
                {"name": "Solver Villain", "active": True, "stack_bb": stack},
            ]
        ),
        "features.opponent_profiles": json_safe([]),
        "features.pot_bb": round(pot, 6),
        "features.to_call_bb": round(to_call, 6),
        "features.ev_bb": round(ev, 6),
        "features.call_max_bb": safe_cap(call_max),
        "features.equity_table": round(equity, 6),
        "features.equity_1v1": round(equity, 6),
        "features.equity_required": round(required, 6),
        "features.equity_gap": round(equity - required, 6),
        "features.hero_stack_bb": round(stack, 6),
        "features.effective_stack_bb": round(stack, 6),
        "features.stack_to_pot_ratio": round(stack / pot, 6) if pot > 0 else 0.0,
        "features.to_call_pot_ratio": round(to_call / pot, 6) if pot > 0 else 0.0,
        "features.has_check": 1.0 if has_check else 0.0,
        "features.has_call": 1.0 if has_call else 0.0,
        "features.has_raise": 1.0 if has_raise else 0.0,
        "features.active_opponents": 1.0,
        "features.hero_position": str(record.get("hero_position") or source.get("features.hero_position") or "UNKNOWN").upper(),
        "audit.source_snapshot_id": source_snapshot_id(source),
        "audit.solver_job_id": record.get("solver_job_id"),
        "audit.solver_dominant_action": record.get("dominant_action"),
        "audit.solver_dominant_frequency": record.get("dominant_frequency"),
        "audit.solver_action_frequencies": json_safe(record.get("action_frequencies") or {}),
        "audit.villain_hand": json_safe(list(villain_hand)),
    }


def augment_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_rows: int | None,
    random_seed: int,
) -> dict[str, Any]:
    base = [dict(row) for row in rows]
    if target_rows is None or target_rows <= len(base):
        return {
            "rows": base,
            "applied": False,
            "source_rows": len(base),
            "mode": "none",
        }
    rng = np.random.default_rng(random_seed)
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in ALLOWED_LABELS}
    for row in base:
        by_label.setdefault(str(row["bootstrap_label"]), []).append(dict(row))
    if any(not by_label.get(label) for label in ALLOWED_LABELS):
        raise ValueError("cannot_augment_missing_class")
    target_per_class = {label: target_rows // len(ALLOWED_LABELS) for label in ALLOWED_LABELS}
    for label in ALLOWED_LABELS[: target_rows % len(ALLOWED_LABELS)]:
        target_per_class[label] += 1
    augmented: list[dict[str, Any]] = []
    for label in ALLOWED_LABELS:
        source = by_label[label]
        label_target = target_per_class[label]
        label_rows = [dict(row) for row in source[:label_target]]
        index = len(label_rows)
        while index < label_target:
            chosen = dict(source[int(rng.integers(0, len(source)))])
            chosen["audit.source_snapshot_id"] = source_snapshot_id(chosen)
            chosen["snapshot_id"] = f"{chosen.get('snapshot_id')}:aug:{index}"
            chosen["metadata.label_source"] = "legacy_augmented"
            chosen["audit.augmentation"] = "resampled_jittered_live_bb_v1"
            chosen["generation_method"] = "resampled_jitter"
            jitter_numeric_features(chosen, rng)
            label_rows.append(chosen)
            index += 1
        augmented.extend(label_rows)
    return {
        "rows": augmented[:target_rows],
        "applied": True,
        "source_rows": len(base),
        "mode": "class_balanced_resample_with_small_numeric_jitter",
    }


def jitter_numeric_features(row: dict[str, Any], rng: np.random.Generator) -> None:
    continuous = [
        "features.pot_bb",
        "features.to_call_bb",
        "features.ev_bb",
        "features.call_max_bb",
        "features.hero_stack_bb",
        "features.effective_stack_bb",
        "features.stack_to_pot_ratio",
    ]
    factor = float(rng.normal(1.0, 0.025))
    factor = max(0.92, min(1.08, factor))
    for feature in continuous:
        value = number_or_none(row.get(feature))
        if value is not None:
            row[feature] = round(max(0.0, value * factor), 6)
    pot = number_or_none(row.get("features.pot_bb"))
    to_call = number_or_none(row.get("features.to_call_bb"))
    if pot is not None and pot > 0 and to_call is not None:
        row["features.to_call_pot_ratio"] = round(to_call / pot, 6)
    required_denominator = (pot or 0.0) + (to_call or 0.0)
    if required_denominator > 0:
        row["features.equity_required"] = round((to_call or 0.0) / required_denominator, 6)
    row["features.equity_gap"] = round(number_or_zero(row.get("features.equity_table")) - number_or_zero(row.get("features.equity_required")), 6)


def is_trainable_row(row: Mapping[str, Any], *, rebuilder: Any) -> bool:
    if rebuilder is not None and hasattr(rebuilder, "is_trainable_row"):
        return bool(rebuilder.is_trainable_row(dict(row)))
    labels = dict(row.get("labels") or {})
    flags = dict(row.get("quality_flags") or {})
    return (
        row.get("type") == "ml_decision_snapshot"
        and labels.get("label_valid") is True
        and labels.get("known_bug_risk") is False
        and flags.get("usable_for_training") is True
        and flags.get("amount_unit_missing") is not True
    )


def rebuild_features(row: Mapping[str, Any], *, rebuilder: Any) -> dict[str, Any]:
    if rebuilder is None or not hasattr(rebuilder, "rebuild_derived_features"):
        return {}
    try:
        return dict(rebuilder.rebuild_derived_features(dict(row), recompute_equity=False))
    except Exception:
        return {}


def solver_enrichment_status(features: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> str:
    if features.get("equity_table") is not None and features.get("ev_bb") is not None and features.get("call_max_bb") is not None:
        return "stored_equity_ev_callmax_used"
    if rebuilt.get("equity_table") is not None:
        return "rebuilt_equity_features_used"
    return "deterministic_features_only"


def flatten_candidate(row: Mapping[str, Any], features: Mapping[str, Any], label: str) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "snapshot_id": row.get("snapshot_id"),
        "bootstrap_label": label,
        "metadata.street": str((row.get("metadata") or {}).get("street") or features.get("street") or "UNKNOWN").upper(),
        "metadata.label_source": str((row.get("metadata") or {}).get("label_source") or "legacy"),
        "labels.final_action": (row.get("labels") or {}).get("final_action"),
        "labels.legacy_action": (row.get("labels") or {}).get("legacy_action"),
        "quality_flags.usable_for_training": (row.get("quality_flags") or {}).get("usable_for_training"),
        "quality_flags.amount_unit_missing": (row.get("quality_flags") or {}).get("amount_unit_missing"),
    }
    for key in RAW_AUDIT_COLUMNS:
        feature_key = key.removeprefix("features.")
        candidate[key] = json_safe(features.get(feature_key))
    candidate.update(
        {
            "features.pot_bb": number_or_zero(features.get("pot_bb")),
            "features.to_call_bb": number_or_zero(features.get("to_call_bb")),
            "features.ev_bb": number_or_zero(features.get("ev_bb")),
            "features.call_max_bb": safe_cap(number_or_zero(features.get("call_max_bb"))),
            "features.equity_table": number_or_zero(features.get("equity_table")),
            "features.equity_1v1": number_or_zero(features.get("equity_1v1")),
            "features.equity_required": number_or_zero(features.get("equity_required")),
            "features.equity_gap": number_or_zero(features.get("equity_table")) - number_or_zero(features.get("equity_required")),
            "features.hero_stack_bb": hero_stack_bb(features),
            "features.effective_stack_bb": effective_stack_bb(features),
            "features.stack_to_pot_ratio": stack_to_pot_ratio(features),
            "features.to_call_pot_ratio": number_or_zero(features.get("to_call_pot_ratio")),
            "features.has_check": bool_to_float(features.get("has_check")),
            "features.has_call": bool_to_float(features.get("has_call")),
            "features.has_raise": bool_to_float(features.get("has_raise")),
            "features.active_opponents": max(0.0, number_or_zero(features.get("player_active")) - 1.0),
            "features.hero_position": str(features.get("hero_position") or "UNKNOWN").upper(),
        }
    )
    return candidate


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return LABEL_ALIASES.get(text, text)


def hero_stack_bb(features: Mapping[str, Any]) -> float:
    players = features.get("players")
    if isinstance(players, list):
        for player in players:
            if isinstance(player, Mapping) and str(player.get("name") or "").lower() == "flael":
                return number_or_zero(player.get("stack_bb"))
        for player in players:
            if isinstance(player, Mapping) and player.get("active"):
                return number_or_zero(player.get("stack_bb"))
    return 0.0


def effective_stack_bb(features: Mapping[str, Any]) -> float:
    hero = hero_stack_bb(features)
    values = []
    players = features.get("players")
    if isinstance(players, list):
        for player in players:
            if isinstance(player, Mapping) and player.get("active") and str(player.get("name") or "").lower() != "flael":
                value = number_or_none(player.get("stack_bb"))
                if value is not None and value > 0:
                    values.append(value)
    if hero > 0 and values:
        return min(hero, max(values))
    return hero


def stack_to_pot_ratio(features: Mapping[str, Any]) -> float:
    pot = number_or_zero(features.get("pot_bb"))
    stack = effective_stack_bb(features)
    return stack / pot if pot > 0 else 0.0


def write_candidates_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = candidate_fieldnames(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "snapshot_id",
        "generation_method",
        *FEATURE_COLUMNS,
        *RAW_AUDIT_COLUMNS,
        "metadata.label_source",
        "labels.final_action",
        "labels.legacy_action",
        "quality_flags.usable_for_training",
        "quality_flags.amount_unit_missing",
        "bootstrap_label",
        "audit.source_snapshot_id",
        "audit.solver_job_id",
        "audit.solver_dominant_action",
        "audit.solver_dominant_frequency",
        "audit.solver_action_frequencies",
        "audit.villain_hand",
        "audit.augmentation",
    ]
    rest = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + rest


def train_model(rows: Sequence[Mapping[str, Any]], *, output_path: Path, random_seed: int) -> dict[str, Any]:
    train_rows, test_rows = split_rows(rows, random_seed=random_seed)
    labels = list(ALLOWED_LABELS)
    comparisons = {
        name: fit_and_score(name, train_rows, test_rows, labels)
        for name in ["logistic_regression", "random_forest", "extra_trees"]
    }
    best_name = max(comparisons.items(), key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]))[0]
    best = comparisons[best_name]
    model = best.pop("model")
    feature_schema = build_feature_schema(rows)
    label_mapping = build_label_mapping(labels)
    joblib.dump(model, output_path / "model.joblib")
    joblib.dump({"feature_schema": feature_schema, "label_mapping": label_mapping}, output_path / "preprocessing.joblib")
    write_json(feature_schema, output_path / "feature_schema.json")
    write_json(label_mapping, output_path / "label_mapping.json")
    write_json(build_feature_contract(), output_path / "feature_contract.json")
    write_json(build_preprocessing_schema(rows), output_path / "preprocessing_schema.json")
    return {
        "status": "ok",
        "model_type": best_name,
        "selected_model": best_name,
        "model_feature_columns": list(FEATURE_COLUMNS),
        "allowed_predictions": labels,
        "split_strategy": "grouped_by_source_snapshot_id_stratified_by_label",
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
        "model_comparison": {name: without_model(value) for name, value in comparisons.items()},
    }


def split_rows(rows: Sequence[Mapping[str, Any]], *, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(random_seed)
    by_label_and_group: dict[str, dict[str, list[Mapping[str, Any]]]] = {label: {} for label in ALLOWED_LABELS}
    for row in rows:
        label = str(row["bootstrap_label"])
        by_label_and_group.setdefault(label, {}).setdefault(source_snapshot_id(row), []).append(row)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in ALLOWED_LABELS:
        groups = list(by_label_and_group.get(label, {}).values())
        if not groups:
            continue
        order = rng.permutation(len(groups))
        test_group_count = max(1, int(round(len(groups) * 0.25))) if len(groups) > 1 else 0
        test_indexes = set(int(index) for index in order[:test_group_count])
        for index, group in enumerate(groups):
            target = test if index in test_indexes else train
            target.extend(dict(row) for row in group)
    return train, test


def source_snapshot_id(row: Mapping[str, Any]) -> str:
    if row.get("generation_method") == "solver_generated" and row.get("audit.solver_job_id"):
        return str(row["audit.solver_job_id"])
    explicit = row.get("audit.source_snapshot_id")
    if explicit:
        return str(explicit)
    snapshot_id = str(row.get("snapshot_id") or "")
    return snapshot_id.split(":aug:", 1)[0]


def count_generation_method(rows: Sequence[Mapping[str, Any]], method: str) -> int:
    return sum(1 for row in rows if row.get("generation_method") == method)


def source_snapshot_id_unique_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({source_snapshot_id(row) for row in rows if source_snapshot_id(row)})


def determine_generation_mode(*, solver_generated_rows: int, resampled_rows: int) -> str:
    if solver_generated_rows > 0 and resampled_rows > 0:
        return "imported_plus_solver_plus_resampling"
    if solver_generated_rows > 0:
        return "imported_plus_solver"
    if resampled_rows > 0:
        return "imported_plus_resampling"
    return "imported_only"


def generation_warnings(
    *,
    solver_report: Mapping[str, Any],
    final_rows: int,
    target_rows: int | None,
    use_solver: bool,
    solver_generated_rows: int,
    min_solver_rows: int,
    resampled_rows: int,
    inflation_ratio: float,
) -> list[str]:
    warnings = list(solver_report.get("warnings") or [])
    if use_solver and solver_generated_rows < min_solver_rows:
        warnings.append(f"solver_generated_rows_below_min_solver_rows:{solver_generated_rows}<{min_solver_rows}")
    if final_rows > 0 and resampled_rows / final_rows > MAX_RESAMPLED_SHARE_WITHOUT_WARNING:
        warnings.append(f"resampled_rows_share_above_80_percent:{resampled_rows}/{final_rows}")
    if inflation_ratio > MAX_INFLATION_RATIO_WITHOUT_WARNING:
        warnings.append(f"inflation_ratio_above_20:{inflation_ratio}")
    requested = target_rows if target_rows is not None else final_rows
    if requested > OVERFIT_TARGET_ROW_THRESHOLD:
        warnings.append(f"target_rows_above_learning_curve_overfit_threshold:{requested}>{OVERFIT_TARGET_ROW_THRESHOLD}")
    return sorted(dict.fromkeys(warnings))


def build_strict_solver_failure(
    *,
    solver_report: Mapping[str, Any],
    solver_generated_rows: int,
    min_solver_rows: int,
) -> dict[str, Any]:
    reason = solver_report.get("warning") or solver_report.get("last_error") or "solver_generated_rows_below_min_solver_rows"
    return {
        "status": "failed",
        "error": f"solver_strict_failed:solver_generated_rows {solver_generated_rows} < min_solver_rows {min_solver_rows}: {reason}",
        "solver_generated_rows": solver_generated_rows,
        "min_solver_rows": min_solver_rows,
        "solver_generation": dict(solver_report),
        "resampling_skipped": True,
        "not_for_production": True,
        "bot_live_connection": "not_modified",
    }


def fit_and_score(name: str, train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    model = make_model(name)
    x_train = [live_feature_payload(row) for row in train_rows]
    y_train = [str(row["bootstrap_label"]) for row in train_rows]
    x_test = [live_feature_payload(row) for row in test_rows]
    y_test = [str(row["bootstrap_label"]) for row in test_rows]
    model.fit(x_train, y_train)
    predictions = [str(value) for value in model.predict(x_test)]
    return {
        "model": model,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="weighted", zero_division=0)), 6),
        "confusion_matrix": matrix_as_dict(y_test, predictions, labels),
        "classification_report": classification_report(y_test, predictions, labels=list(labels), output_dict=True, zero_division=0),
    }


def make_model(name: str) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=17)
    elif name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=160, max_depth=8, class_weight="balanced", random_state=17)
    elif name == "extra_trees":
        classifier = ExtraTreesClassifier(n_estimators=160, max_depth=8, class_weight="balanced", random_state=17)
    else:
        raise ValueError(f"unknown_model:{name}")
    return Pipeline([("vectorizer", DictVectorizer(sparse=False)), ("classifier", classifier)])


def live_feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        payload[feature] = number_or_zero(row.get(feature))
    for feature in CATEGORICAL_FEATURES:
        payload[feature] = str(row.get(feature) or "UNKNOWN")
    return payload


def predict_live_bb_model(*, model_dir: str | Path, input_payload: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(model_dir)
    model = joblib.load(directory / "model.joblib")
    labels = read_json(directory / "label_mapping.json")["labels"]
    if "features.pot_bb" in input_payload:
        row = dict(input_payload)
    else:
        rebuilder = load_feature_rebuilder(Path("dist/ml_dataset_export_v2/feature_rebuilder.py"))
        rebuilt = rebuild_features(input_payload, rebuilder=rebuilder)
        features = {**dict(input_payload.get("features") or {}), **rebuilt}
        label = normalize_label((input_payload.get("labels") or {}).get("final_action") or "CHECK")
        row = flatten_candidate(input_payload, features, label)
    features = live_feature_payload(row)
    prediction = str(model.predict([features])[0])
    return {
        "status": "ok",
        "prediction": prediction,
        "probabilities": predict_probabilities(model, features),
        "labels": labels,
    }


def predict_first_imported_line(*, model_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "failed", "error": "no_rows"}
    return predict_live_bb_model(model_dir=model_dir, input_payload=rows[0])


def predict_probabilities(model: Any, features: Mapping[str, Any]) -> dict[str, float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba([features])[0]
    return {str(label): round(float(probability), 6) for label, probability in zip(model.classes_, probabilities)}


def write_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    model_type: str,
    random_seed: int,
) -> dict[str, str]:
    correlation = build_input_correlation(rows)
    correlation_json = output_path / "input_feature_correlation_matrix.json"
    correlation_svg = output_path / "input_feature_correlation_matrix.svg"
    high_pairs_md = output_path / "high_correlation_pairs.md"
    write_json(correlation, correlation_json)
    write_correlation_svg(correlation, correlation_svg)
    high_pairs_md.write_text(render_high_correlation_pairs(correlation), encoding="utf-8")

    learning = build_learning_curve(rows, model_type=model_type, random_seed=random_seed)
    learning_json = output_path / "learning_curve_report.json"
    learning_svg = output_path / "learning_curve.svg"
    learning_md = output_path / "learning_curve_report.md"
    write_json(learning, learning_json)
    write_learning_curve_svg(learning, learning_svg)
    learning_md.write_text(render_learning_curve_report(learning), encoding="utf-8")

    importance = build_feature_importance(output_path / "model.joblib")
    importance_json = output_path / "feature_importance.json"
    importance_md = output_path / "feature_importance.md"
    write_json(importance, importance_json)
    importance_md.write_text(render_feature_importance(importance), encoding="utf-8")

    return {
        "input_feature_correlation_matrix_json": str(correlation_json),
        "input_feature_correlation_matrix_svg": str(correlation_svg),
        "high_correlation_pairs": str(high_pairs_md),
        "learning_curve_report_json": str(learning_json),
        "learning_curve_svg": str(learning_svg),
        "learning_curve_report_md": str(learning_md),
        "feature_importance_json": str(importance_json),
        "feature_importance_md": str(importance_md),
    }


def build_input_correlation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names, matrix = encoded_feature_matrix(rows)
    corr = safe_correlation_matrix(matrix)
    pairs = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            value = float(corr[i, j])
            if abs(value) > 0.80:
                pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": names[j],
                        "correlation": round(value, 6),
                        "abs_correlation": round(abs(value), 6),
                        "comment": correlation_comment(left, names[j], abs(value)),
                    }
                )
    return {
        "schema": "live_bb_baseline_v1_input_correlation",
        "row_count": len(rows),
        "feature_order": list(FEATURE_COLUMNS),
        "encoded_feature_order": names,
        "matrix": [[round(float(value), 6) for value in row] for row in corr.tolist()],
        "high_correlation_pairs_abs_gt_0_80": sorted(pairs, key=lambda item: (-item["abs_correlation"], item["feature_a"], item["feature_b"])),
        "high_correlation_pairs_abs_gt_0_90": [pair for pair in sorted(pairs, key=lambda item: -item["abs_correlation"]) if pair["abs_correlation"] > 0.90],
    }


def safe_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    feature_count = matrix.shape[1]
    if feature_count == 0:
        return np.zeros((0, 0))
    corr = np.eye(feature_count)
    if matrix.shape[0] < 2:
        return corr
    std = matrix.std(axis=0)
    variable_indexes = np.where(std > 0)[0]
    if len(variable_indexes) < 2:
        return corr
    variable_matrix = matrix[:, variable_indexes]
    variable_corr = np.corrcoef(variable_matrix, rowvar=False)
    variable_corr = np.atleast_2d(np.nan_to_num(variable_corr, nan=0.0, posinf=0.0, neginf=0.0))
    for row_pos, row_index in enumerate(variable_indexes):
        for col_pos, col_index in enumerate(variable_indexes):
            corr[int(row_index), int(col_index)] = float(variable_corr[row_pos, col_pos])
    return corr


def encoded_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    payloads = [live_feature_payload(row) for row in rows]
    encoded: dict[str, list[float]] = {}
    for feature in FEATURE_COLUMNS:
        values = [payload.get(feature) for payload in payloads]
        if feature in NUMERIC_FEATURES:
            encoded[feature] = [number_or_zero(value) for value in values]
        else:
            categories = sorted({str(value or "UNKNOWN") for value in values})
            for category in categories:
                encoded[f"{feature}={category}"] = [1.0 if str(value or "UNKNOWN") == category else 0.0 for value in values]
    names = list(encoded)
    matrix = np.array([encoded[name] for name in names], dtype=float).T if names else np.zeros((len(rows), 0))
    return names, matrix


def correlation_comment(left: str, right: str, absolute: float) -> str:
    base = {left.split("=")[0], right.split("=")[0]}
    if absolute > 0.95:
        return "suspect_redundant"
    if base in [
        {"features.has_check", "features.has_call"},
        {"features.equity_table", "features.equity_gap"},
        {"features.to_call_bb", "features.equity_required"},
        {"features.hero_stack_bb", "features.effective_stack_bb"},
    ]:
        return "redondant"
    return "acceptable_monitor"


def write_correlation_svg(report: Mapping[str, Any], output_path: Path) -> None:
    labels = list(report["encoded_feature_order"])
    matrix = report["matrix"]
    cell = 22
    left = 285
    top = 170
    width = left + cell * len(labels) + 30
    height = top + cell * len(labels) + 30
    lines = [
        svg_header(width, height),
        '<text x="20" y="30" font-size="18" font-family="Arial">live_bb_baseline_v1 input correlation</text>',
        '<text x="20" y="54" font-size="12" font-family="Arial" fill="#555">Pearson correlation on encoded model input features.</text>',
    ]
    for idx, label in enumerate(labels):
        x = left + idx * cell + cell / 2
        y = top + idx * cell + cell / 2
        short = short_label(label)
        lines.append(f'<text x="{x}" y="{top - 8}" text-anchor="start" font-size="9" font-family="Arial" transform="rotate(-55 {x} {top - 8})">{escape_xml(short)}</text>')
        lines.append(f'<text x="{left - 8}" y="{y + 3}" text-anchor="end" font-size="9" font-family="Arial">{escape_xml(short)}</text>')
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{diverging_color(float(value))}" stroke="#ffffff" stroke-width="0.5"/>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_high_correlation_pairs(report: Mapping[str, Any]) -> str:
    lines = [
        "# High Correlation Pairs",
        "",
        "| feature A | feature B | correlation | comment |",
        "|---|---|---:|---|",
    ]
    for pair in report["high_correlation_pairs_abs_gt_0_80"]:
        lines.append(f"| `{pair['feature_a']}` | `{pair['feature_b']}` | {pair['correlation']} | `{pair['comment']}` |")
    if len(lines) == 4:
        lines.append("| NA | NA | NA | `no_pair_above_threshold` |")
    lines.append("")
    return "\n".join(lines)


def build_learning_curve(rows: Sequence[Mapping[str, Any]], *, model_type: str, random_seed: int) -> dict[str, Any]:
    train_rows, validation_rows = split_rows(rows, random_seed=random_seed)
    points = []
    for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = stratified_subset(train_rows, fraction=fraction, random_seed=random_seed)
        if len({row["bootstrap_label"] for row in subset}) < len(ALLOWED_LABELS):
            continue
        model = make_model(model_type)
        x_train = [live_feature_payload(row) for row in subset]
        y_train = [str(row["bootstrap_label"]) for row in subset]
        x_val = [live_feature_payload(row) for row in validation_rows]
        y_val = [str(row["bootstrap_label"]) for row in validation_rows]
        model.fit(x_train, y_train)
        train_pred = [str(value) for value in model.predict(x_train)]
        val_pred = [str(value) for value in model.predict(x_val)]
        train_macro = f1_score(y_train, train_pred, labels=list(ALLOWED_LABELS), average="macro", zero_division=0)
        val_macro = f1_score(y_val, val_pred, labels=list(ALLOWED_LABELS), average="macro", zero_division=0)
        points.append(
            {
                "train_size": len(subset),
                "train_accuracy": round(float(accuracy_score(y_train, train_pred)), 6),
                "validation_accuracy": round(float(accuracy_score(y_val, val_pred)), 6),
                "train_macro_f1": round(float(train_macro), 6),
                "validation_macro_f1": round(float(val_macro), 6),
                "macro_f1_gap": round(float(train_macro - val_macro), 6),
            }
        )
    return {
        "schema": "live_bb_baseline_v1_learning_curve",
        "model_type": model_type,
        "row_count": len(rows),
        "validation_size": len(validation_rows),
        "points": points,
        "diagnostic": learning_curve_diagnostic(points),
    }


def stratified_subset(rows: Sequence[Mapping[str, Any]], *, fraction: float, random_seed: int) -> list[dict[str, Any]]:
    if fraction >= 1.0:
        return [dict(row) for row in rows]
    rng = np.random.default_rng(random_seed + int(fraction * 1000))
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["bootstrap_label"]), []).append(row)
    selected = []
    for label in ALLOWED_LABELS:
        label_rows = by_label.get(label, [])
        count = max(1, int(round(len(label_rows) * fraction)))
        indexes = rng.choice(len(label_rows), size=min(count, len(label_rows)), replace=False)
        selected.extend(dict(label_rows[int(index)]) for index in indexes)
    return selected


def learning_curve_diagnostic(points: Sequence[Mapping[str, Any]]) -> str:
    if not points:
        return "insufficient_data"
    last = points[-1]
    train = float(last.get("train_macro_f1") or 0.0)
    validation = float(last.get("validation_macro_f1") or 0.0)
    gap = train - validation
    if train > 0.95 and gap > 0.08:
        return "overfit_probable"
    if train < 0.65 and validation < 0.65:
        return "underfit_probable"
    if train > 0.98 and validation > 0.98:
        return "dataset_too_simple_or_leakage_possible"
    return "acceptable"


def write_learning_curve_svg(report: Mapping[str, Any], output_path: Path) -> None:
    points = list(report.get("points", []))
    width = 760
    height = 360
    left = 70
    top = 45
    plot_w = 610
    plot_h = 245
    lines = [
        svg_header(width, height),
        '<text x="20" y="28" font-size="18" font-family="Arial">live_bb_baseline_v1 learning curve</text>',
        f'<text x="20" y="348" font-size="12" font-family="Arial" fill="#555">Diagnostic: {escape_xml(str(report.get("diagnostic")))}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cccccc"/>',
    ]
    for tick in range(6):
        y = top + plot_h - tick * plot_h / 5
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-size="10" font-family="Arial">{tick / 5:.1f}</text>')
    if points:
        min_x = min(point["train_size"] for point in points)
        max_x = max(point["train_size"] for point in points)
        draw_polyline(lines, points, "train_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#2957a4")
        draw_polyline(lines, points, "validation_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#c44733")
    lines.append('<circle cx="705" cy="62" r="5" fill="#2957a4"/><text x="716" y="66" font-size="11" font-family="Arial">train macro F1</text>')
    lines.append('<circle cx="705" cy="82" r="5" fill="#c44733"/><text x="716" y="86" font-size="11" font-family="Arial">validation macro F1</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def draw_polyline(lines: list[str], points: Sequence[Mapping[str, Any]], key: str, min_x: int, max_x: int, left: int, top: int, plot_w: int, plot_h: int, color: str) -> None:
    coords = []
    for point in points:
        x = left + (0.5 * plot_w if max_x == min_x else (point["train_size"] - min_x) / (max_x - min_x) * plot_w)
        y = top + plot_h - max(0.0, min(1.0, float(point[key]))) * plot_h
        coords.append(f"{x},{y}")
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')


def render_learning_curve_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Learning Curve Report",
        "",
        f"Diagnostic: `{report.get('diagnostic')}`",
        "",
        "| train size | train accuracy | validation accuracy | train macro F1 | validation macro F1 | gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in report.get("points", []):
        lines.append(
            f"| {point['train_size']} | {point['train_accuracy']} | {point['validation_accuracy']} | "
            f"{point['train_macro_f1']} | {point['validation_macro_f1']} | {point['macro_f1_gap']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_feature_importance(model_path: Path) -> dict[str, Any]:
    model = joblib.load(model_path)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    if hasattr(classifier, "feature_importances_"):
        values = [float(value) for value in classifier.feature_importances_]
        source = "tree_feature_importance"
    elif hasattr(classifier, "coef_"):
        values = [float(value) for value in np.mean(np.abs(classifier.coef_), axis=0)]
        source = "mean_abs_logistic_coefficient"
    else:
        values = [0.0 for _ in names]
        source = "unavailable"
    ranking = sorted(
        [{"feature": name, "importance": round(value, 8)} for name, value in zip(names, values)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {"schema": "live_bb_baseline_v1_feature_importance", "source": source, "features": ranking}


def render_feature_importance(report: Mapping[str, Any]) -> str:
    lines = [
        "# Feature Importance",
        "",
        f"Source: `{report.get('source')}`",
        "",
        "| rank | feature | importance |",
        "|---:|---|---:|",
    ]
    for index, row in enumerate(report.get("features", [])[:40], start=1):
        lines.append(f"| {index} | `{row['feature']}` | {row['importance']} |")
    lines.append("")
    return "\n".join(lines)


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def escape_xml(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short_label(value: str, *, limit: int = 36) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def diverging_color(value: float) -> str:
    safe = max(-1.0, min(1.0, value))
    if safe >= 0:
        ratio = safe
        red = int(245 - 35 * ratio)
        green = int(247 - 130 * ratio)
        blue = int(250 - 175 * ratio)
    else:
        ratio = abs(safe)
        red = int(245 - 170 * ratio)
        green = int(247 - 95 * ratio)
        blue = int(250 - 20 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def build_feature_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "live_bb_baseline_v1",
        "feature_order": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": "bootstrap_label",
        "labels": list(ALLOWED_LABELS),
        "categorical_values": {
            feature: sorted({str(row.get(feature) or "UNKNOWN") for row in rows})
            for feature in CATEGORICAL_FEATURES
        },
        "not_for_production": True,
        "bot_live_connection": "not_modified",
    }


def build_label_mapping(labels: Sequence[str]) -> dict[str, Any]:
    return {
        "labels": list(labels),
        "label_to_id": {label: index for index, label in enumerate(labels)},
        "id_to_label": {str(index): label for index, label in enumerate(labels)},
    }


def build_feature_contract() -> dict[str, Any]:
    return {
        "schema": "live_bb_baseline_v1",
        "features_model_used": list(FEATURE_COLUMNS),
        "features_audit_only": list(RAW_AUDIT_COLUMNS),
        "features_leakage_excluded": ["labels.*", "debug.*", "audit.*", "quality_flags.*", "metadata.label_source"],
        "leakage_columns_used_by_model": [feature for feature in FEATURE_COLUMNS if is_leakage_column(feature)],
        "label_normalization": {"COL": "CALL", "RES": "RAISE"},
        "allowed_predictions": list(ALLOWED_LABELS),
        "call_is_native_class": True,
        "raw_cards_direct_features": False,
        "not_for_production": True,
        "bot_live_connection": "not_modified",
    }


def build_preprocessing_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "feature_order": list(FEATURE_COLUMNS),
        "dict_vectorizer_input": "live_bb_feature_payload",
        "null_policy": {"numeric": "missing_or_invalid_to_0.0", "categorical": "missing_to_UNKNOWN"},
        "categorical_values": {
            feature: sorted({str(row.get(feature) or "UNKNOWN") for row in rows})
            for feature in CATEGORICAL_FEATURES
        },
    }


def call_diagnostic(training: Mapping[str, Any]) -> dict[str, Any]:
    report = training["classification_report"].get("CALL", {})
    recall = float(report.get("recall") or 0.0)
    precision = float(report.get("precision") or 0.0)
    if recall >= 0.7:
        status = "call_learned"
    elif recall >= 0.35:
        status = "call_weak_needs_more_rows"
    else:
        status = "call_too_weak"
    return {"status": status, "precision": round(precision, 6), "recall": round(recall, 6), "support": report.get("support")}


def matrix_as_dict(truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predictions, labels=list(labels))
    return {
        expected: {predicted: int(matrix[row_index][col_index]) for col_index, predicted in enumerate(labels)}
        for row_index, expected in enumerate(labels)
    }


def is_leakage_column(column: str) -> bool:
    return column.startswith(LEAKAGE_PREFIXES) or column in LEAKAGE_NAMES


def without_model(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "model"}


def number_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text.lower() in {"", "none", "null", "nan", "inf", "infinity"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def number_or_zero(value: Any) -> float:
    return number_or_none(value) or 0.0


def safe_cap(value: float, *, cap: float = 500.0) -> float:
    return max(-cap, min(cap, value))


def bool_to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "oui"}:
        return 1.0
    return 0.0


def json_safe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Optionally resample/jitter the imported dataset to this many rows for a quick experimental baseline.",
    )
    parser.add_argument("--use-solver", action="store_true", help="Run real hero-oriented postflop solver jobs before fallback resampling.")
    parser.add_argument("--solver-rows", type=int, default=0, help="Maximum solver_generated rows to add when --use-solver is set.")
    parser.add_argument("--min-solver-rows", type=int, default=DEFAULT_MIN_SOLVER_ROWS)
    parser.add_argument("--solver-timeout-s", type=float, default=2.0)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=["rust", "python"], default="rust")
    parser.add_argument("--strict-solver", action="store_true", help="Fail if --use-solver produces zero solver_generated rows.")
    parser.add_argument("--solver-strict", action="store_true", help="Alias for --strict-solver; fail below --min-solver-rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_live_bb_baseline_v1(
        input_jsonl=args.input,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        target_rows=args.target_rows,
        use_solver=args.use_solver,
        solver_rows=args.solver_rows,
        solver_timeout_s=args.solver_timeout_s,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        strict_solver=args.strict_solver or args.solver_strict,
        min_solver_rows=args.min_solver_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
