"""Export a flat bootstrap dataset from guarded solver action candidates.

This exporter is intentionally conservative. It creates ``bootstrap_label`` for
pipeline testing, never ``gto_label`` or ``training_label``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver_jobs.action_candidate import build_solver_action_candidate
from solver_jobs.strategy_extractor import extract_root_strategy


ALLOWED_ACTION_PREFIXES = ("BET_", "RAISE_")
ALLOWED_ACTIONS = {"CHECK", "FOLD", "CALL"}
MODEL_ACTIONS = {"CHECK", "FOLD", "CALL", "RAISE"}
BLOCKING_DANGER_FLAGS = {"extreme_action_all_in", "dominant_action_unstable", "frequency_too_low", "iterations_too_low", "exploitability_missing", "timeout", "root_not_hero", "strategy_source_not_average_strategy"}
MIN_DOMINANT_FREQUENCY = 0.70
MIN_ITERATIONS = 25
LABEL_SOURCE = "solver_candidate"
LABEL_QUALITY = "bootstrap_solver_untrusted"
WEAK_LABEL_SOURCE = "weak_rule_bootstrap"
WEAK_LABEL_QUALITY = "bootstrap_weak_rule_untrusted"
DEFAULT_V2_MIN_USABLE_ROWS = 100
DEFAULT_V2_CLASS_FLOOR = 10
CSV_FIELDS = (
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
    "iterations",
    "exploitability_last",
    "candidate_confidence",
    "raw_action",
    "normalized_action",
    "bootstrap_label",
    "label_source",
    "label_quality",
    "weak_rule_reason",
    "board_card_count",
    "is_river",
    "is_turn",
    "is_check_or_bet_context",
    "is_facing_bet_context",
    "to_call_ratio",
    "stack_to_pot_ratio",
    "excluded",
    "exclusion_reason",
)
DIST_BOOTSTRAP_FIX_ID = "bootstrap_dist_schema_alignment_v5"
DIST_BOOTSTRAP_FIX_DATE = "2026-05-26"
DIST_AUDIT_FIELDS = (
    "bootstrap_label",
    "excluded",
    "exclusion_reason",
    "audit.source_id",
    "audit.label_source",
    "audit.label_quality",
    "audit.weak_rule_reason",
    "audit.dominant_action",
    "audit.dominant_action_frequency",
    "audit.raw_action",
    "audit.normalized_action",
    "audit.action_frequencies",
    "audit.iterations",
    "audit.exploitability_last",
    "audit.candidate_confidence",
)
DIST_BASE_FIELDS = (
    "schema_version",
    "type",
    "snapshot_id",
    "recorded_at",
    "metadata.game",
    "metadata.hand_id",
    "metadata.scan_count",
    "metadata.street",
    "metadata.status",
    "metadata.decision_mode",
    "metadata.label_source",
    "metadata.new_party_state",
    "metadata.decision_engine_version",
    "metadata.legacy_rules_version",
    "metadata.decision_engine_fix_id",
    "metadata.decision_engine_fix_date",
    "metadata.git_commit",
    "features.hero_cards",
    "features.board_cards",
    "features.board_card_count",
    "features.hero_position",
    "features.player_start",
    "features.player_active",
    "features.active_opponents",
    "features.hero_stack",
    "features.effective_stack",
    "features.stack_to_pot_ratio",
    "features.pot",
    "features.to_call",
    "features.to_call_pot_ratio",
    "features.buttons",
    "features.buttons_active",
    "features.has_check",
    "features.has_call",
    "features.has_raise",
    "features.players",
    "features.opponent_profiles",
    "features.equity_table",
    "features.equity_1v1",
    "features.equity_known",
    "features.equity_required",
    "features.ev",
    "features.call_max",
    "labels.legacy_action",
    "labels.legacy_reason",
    "labels.legacy_raise_amount",
    "labels.ml_action",
    "labels.ml_confidence",
    "labels.final_action",
    "labels.fallback_reason",
    "labels.label_valid",
    "labels.label_exclusion_reason",
    "labels.known_bug_risk",
    "confidence.hero_cards_min",
    "confidence.board_cards_min",
    "confidence.pot_ocr",
    "confidence.to_call_ocr",
    "confidence.buttons_min",
    "confidence.hero_position",
    "confidence.player_count",
    "quality_flags.hero_cards_uncertain",
    "quality_flags.board_uncertain",
    "quality_flags.opponent_count_uncertain",
    "quality_flags.pot_to_call_incoherent",
    "quality_flags.buttons_incoherent",
    "quality_flags.hero_position_low_confidence",
    "quality_flags.street_transient",
    "quality_flags.usable_for_training",
    "debug.decision_reason",
    "debug.scan_status",
)
DIST_ALIGNED_FIELDS = DIST_BASE_FIELDS + DIST_AUDIT_FIELDS
DIST_JSON_FIELDS = {
    "features.hero_cards",
    "features.board_cards",
    "features.buttons",
    "features.buttons_active",
    "features.players",
    "features.opponent_profiles",
    "audit.action_frequencies",
}


def export_candidate_dataset(
    input_paths: list[str | Path],
    *,
    output_jsonl: str | Path,
    output_csv: str | Path | None = None,
    include_weak_rules: bool = False,
    min_usable_rows: int = DEFAULT_V2_MIN_USABLE_ROWS,
    class_floor: int = DEFAULT_V2_CLASS_FLOOR,
    min_dominant_frequency: float = MIN_DOMINANT_FREQUENCY,
    balance_weak_rules: bool = False,
) -> dict[str, Any]:
    records, load_warnings = load_records(input_paths)
    rows = build_candidate_dataset_rows(records, min_dominant_frequency=min_dominant_frequency)
    if include_weak_rules:
        rows.extend(
            generate_weak_rule_rows(
                rows,
                min_usable_rows=min_usable_rows,
                class_floor=class_floor,
                balance_classes=balance_weak_rules,
            )
        )
    write_jsonl(rows, output_jsonl)
    if output_csv is not None:
        write_csv(rows, output_csv)
    summary = summarize_export(
        rows,
        warnings=load_warnings,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        input_records_count=len(records),
    )
    report_paths = write_dataset_reports(rows, summary, output_jsonl)
    return {**summary, **report_paths}


def load_records(input_paths: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        warnings.append(f"{path}:line_{line_index}:invalid_json")
                        continue
                    if isinstance(value, Mapping):
                        records.append(dict(value))
                    else:
                        warnings.append(f"{path}:line_{line_index}:non_object")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path}:{type(exc).__name__}:{exc}")
    return records, warnings


def build_candidate_dataset_rows(
    records: list[dict[str, Any]],
    *,
    min_dominant_frequency: float = MIN_DOMINANT_FREQUENCY,
) -> list[dict[str, Any]]:
    normalized = [normalize_record(record) for record in records]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[group_key(row)].append(row)

    stable_by_key: dict[str, tuple[bool, str | None]] = {}
    for key, rows in grouped.items():
        actions = [row["dominant_action"] for row in rows if row["dominant_action"]]
        if len(rows) < 2:
            stable_by_key[key] = (False, "single_run_only")
        elif not actions or len(set(actions)) != 1:
            stable_by_key[key] = (False, "action_unstable")
        else:
            stable_by_key[key] = (True, None)

    exported: list[dict[str, Any]] = []
    for row in normalized:
        stable, stability_error = stable_by_key[group_key(row)]
        exclusion = exclusion_reason(
            row,
            stable=stable,
            stability_error=stability_error,
            min_dominant_frequency=min_dominant_frequency,
        )
        exported.append(flat_export_row(row, exclusion_reason_value=exclusion))
    return exported


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("record_type") == "candidate_sensitivity_result":
        return normalize_sensitivity_record(record)
    return normalize_solver_run_record(record)


def normalize_sensitivity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": _text(record.get("solver_job_id") or record.get("source_id")),
        "group_id": f"{record.get('context')}::{record.get('scenario')}",
        "street": _text(record.get("street")),
        "hero_cards": record.get("hero_cards") or record.get("hero_hand"),
        "villain_hand": record.get("villain_hand"),
        "board_cards": record.get("board_cards") or record.get("board"),
        "pot": _float_or_none(record.get("pot")),
        "to_call": _float_or_none(record.get("to_call")),
        "stack": _float_or_none(record.get("stack")),
        "spr": _float_or_none(record.get("spr")),
        "position_model": _position_from_context(record.get("context")),
        "decision_context_type": _decision_type_from_context(record.get("context")),
        "action_frequencies": dict(record.get("action_frequencies") or {}),
        "dominant_action": _text(record.get("dominant_action")),
        "dominant_action_frequency": _float_or_none(record.get("dominant_frequency")),
        "iterations": _int_or_none(record.get("iterations")),
        "exploitability_last": _float_or_none(record.get("exploitability_last")),
        "candidate_confidence": _confidence_from_frequency(record.get("dominant_frequency")),
        "root_matches_hero": record.get("root_matches_hero"),
        "root_player_role": record.get("root_player_role"),
        "solver_status": _text(record.get("solver_status")),
        "danger_flags": list(record.get("danger_flags") or []),
        "raw_error": record.get("error"),
    }


def normalize_solver_run_record(record: Mapping[str, Any]) -> dict[str, Any]:
    solver_job = record.get("solver_job") if isinstance(record.get("solver_job"), Mapping) else {}
    solver_result = record.get("solver_result") if isinstance(record.get("solver_result"), Mapping) else {}
    output = solver_result.get("output") if isinstance(solver_result.get("output"), Mapping) else {}
    strategy = extract_root_strategy(record)
    candidate = build_solver_action_candidate(record)
    pot = _float_or_none(solver_job.get("pot"))
    stack = _float_or_none(solver_job.get("stack"))
    return {
        "source_id": _text(record.get("solver_job_id") or solver_job.get("solver_job_id")),
        "group_id": _group_from_job_id(record.get("solver_job_id") or solver_job.get("solver_job_id")),
        "street": _text(solver_job.get("street")),
        "hero_cards": solver_job.get("hero_hand"),
        "villain_hand": solver_job.get("villain_hand"),
        "board_cards": solver_job.get("board"),
        "pot": pot,
        "to_call": _float_or_none(solver_job.get("to_call")),
        "stack": stack,
        "spr": calculate_spr(stack, pot),
        "position_model": solver_job.get("hero_position_model"),
        "decision_context_type": solver_job.get("decision_context_type"),
        "action_frequencies": strategy.get("action_frequencies") or {},
        "dominant_action": candidate.get("candidate_action") or strategy.get("dominant_action"),
        "dominant_action_frequency": candidate.get("candidate_frequency") or strategy.get("dominant_action_frequency"),
        "iterations": _int_or_none((record.get("quality") or {}).get("iterations") if isinstance(record.get("quality"), Mapping) else output.get("iterations")),
        "exploitability_last": _float_or_none(output.get("exploitability_last")),
        "candidate_confidence": candidate.get("candidate_confidence") or strategy.get("confidence"),
        "root_matches_hero": output.get("root_matches_hero", strategy.get("root_player_role") == "hero"),
        "root_player_role": output.get("root_player_role", strategy.get("root_player_role")),
        "solver_status": _text(record.get("solver_status") or solver_result.get("status")),
        "danger_flags": [],
        "raw_error": record.get("error"),
    }


def exclusion_reason(
    row: Mapping[str, Any],
    *,
    stable: bool,
    stability_error: str | None,
    min_dominant_frequency: float = MIN_DOMINANT_FREQUENCY,
) -> str | None:
    if row.get("raw_error"):
        return "timeout" if "timeout" in str(row.get("raw_error")).lower() else "solver_failed"
    if row.get("solver_status") not in {"ok", "OK"}:
        return "solver_failed"
    if row.get("root_matches_hero") is not True or row.get("root_player_role") != "hero":
        return "root_not_hero"
    action = row.get("dominant_action")
    if not action:
        return "strategy_not_available"
    if action == "ALL_IN":
        return "all_in_excluded"
    if normalize_bootstrap_action(action) is None:
        return "unsupported_action"
    frequency = _float_or_none(row.get("dominant_action_frequency"))
    if frequency is None:
        return "strategy_not_available"
    if frequency < float(min_dominant_frequency):
        return "dominant_action_too_weak"
    if _int_or_none(row.get("iterations")) is None or int(row.get("iterations")) < MIN_ITERATIONS:
        return "iterations_too_low"
    danger_flags = set(str(flag) for flag in row.get("danger_flags") or [])
    blocking = sorted(danger_flags & BLOCKING_DANGER_FLAGS)
    if blocking:
        return f"danger_flag:{blocking[0]}"
    if not stable:
        return stability_error or "action_unstable"
    return None


def flat_export_row(row: Mapping[str, Any], *, exclusion_reason_value: str | None) -> dict[str, Any]:
    excluded = exclusion_reason_value is not None
    raw_action = _text(row.get("dominant_action"))
    normalized_action = normalize_bootstrap_action(raw_action)
    action = None if excluded else normalized_action
    export_row = {
        "source_id": row.get("source_id"),
        "street": row.get("street"),
        "hero_cards": _jsonable_field(row.get("hero_cards")),
        "villain_hand": _jsonable_field(row.get("villain_hand")),
        "board_cards": _jsonable_field(row.get("board_cards")),
        "pot": row.get("pot"),
        "to_call": row.get("to_call"),
        "stack": row.get("stack"),
        "spr": row.get("spr"),
        "position_model": row.get("position_model"),
        "decision_context_type": row.get("decision_context_type"),
        "action_frequencies": dict(row.get("action_frequencies") or {}),
        "dominant_action": row.get("dominant_action"),
        "dominant_action_frequency": row.get("dominant_action_frequency"),
        "iterations": row.get("iterations"),
        "exploitability_last": row.get("exploitability_last"),
        "candidate_confidence": row.get("candidate_confidence") or _confidence_from_frequency(row.get("dominant_action_frequency")),
        "raw_action": raw_action,
        "normalized_action": normalized_action,
        "bootstrap_label": action,
        "label_source": LABEL_SOURCE if not excluded else None,
        "label_quality": LABEL_QUALITY,
        "weak_rule_reason": None,
        "excluded": excluded,
        "exclusion_reason": exclusion_reason_value,
    }
    return add_derived_features(export_row)


def generate_weak_rule_rows(
    existing_rows: list[dict[str, Any]],
    *,
    min_usable_rows: int = DEFAULT_V2_MIN_USABLE_ROWS,
    class_floor: int = DEFAULT_V2_CLASS_FLOOR,
    balance_classes: bool = False,
) -> list[dict[str, Any]]:
    usable = [row for row in existing_rows if not row.get("excluded")]
    counts = Counter(str(row.get("bootstrap_label")) for row in usable if row.get("bootstrap_label") in MODEL_ACTIONS)
    target_classes = ("CHECK", "FOLD", "RAISE")
    base_target = math_ceil_div(max(min_usable_rows, class_floor * len(target_classes)), len(target_classes))
    balance_target = max(counts.get(label, 0) for label in target_classes) if balance_classes else 0
    per_class_target = max(class_floor, base_target, balance_target)

    generated: list[dict[str, Any]] = []
    for label in target_classes:
        needed = max(0, per_class_target - counts.get(label, 0))
        for index in range(needed):
            generated.append(weak_rule_row(label, index))

    while len(usable) + len(generated) < min_usable_rows:
        label = target_classes[len(generated) % len(target_classes)]
        generated.append(weak_rule_row(label, len(generated)))
    return generated


def weak_rule_row(label: str, index: int) -> dict[str, Any]:
    templates = {
        "RAISE": {
            "raw_actions": ("BET_33", "BET_50", "BET_66", "RAISE_33", "RAISE_66"),
            "hero_cards": ["Ah", "As"],
            "villain_hand": ["8c", "3d"],
            "board_cards": ["Ac", "Kd", "7s", "2h", "2c"],
            "pot": 220 + (index % 5) * 40,
            "to_call": 0 if index % 2 == 0 else 40,
            "stack": 1200 + (index % 4) * 200,
            "position_model": "OOP" if index % 2 == 0 else "IP",
            "decision_context_type": "hero_check_or_bet" if index % 2 == 0 else "hero_facing_bet",
            "weak_rule_reason": "two_pair_plus_or_better_value_aggression_no_all_in",
        },
        "CHECK": {
            "raw_actions": ("CHECK",),
            "hero_cards": ["Kh", "Qh"],
            "villain_hand": ["8c", "3d"],
            "board_cards": ["Ad", "7c", "4s", "2d", "9h"],
            "pot": 180 + (index % 5) * 30,
            "to_call": 0,
            "stack": 1000 + (index % 3) * 200,
            "position_model": "OOP",
            "decision_context_type": "hero_check_or_bet",
            "weak_rule_reason": "medium_showdown_value_free_check",
        },
        "FOLD": {
            "raw_actions": ("FOLD",),
            "hero_cards": ["8c", "3d"],
            "villain_hand": ["As", "Ad"],
            "board_cards": ["Ah", "Kd", "Qs", "9c", "2h"],
            "pot": 300 + (index % 5) * 50,
            "to_call": 180 + (index % 4) * 60,
            "stack": 900 + (index % 3) * 150,
            "position_model": "IP" if index % 2 == 0 else "OOP",
            "decision_context_type": "hero_facing_bet",
            "weak_rule_reason": "weak_hand_facing_large_bet_fold",
        },
    }
    template = templates[label]
    raw_action = template["raw_actions"][index % len(template["raw_actions"])]
    normalized_action = normalize_bootstrap_action(raw_action)
    pot = float(template["pot"])
    stack = float(template["stack"])
    to_call = float(template["to_call"])
    row = {
        "source_id": f"weak_rule_{label.lower()}_{index:04d}",
        "street": "RIVER",
        "hero_cards": list(template["hero_cards"]),
        "villain_hand": list(template["villain_hand"]),
        "board_cards": list(template["board_cards"]),
        "pot": pot,
        "to_call": to_call,
        "stack": stack,
        "spr": calculate_spr(stack, pot),
        "position_model": template["position_model"],
        "decision_context_type": template["decision_context_type"],
        "action_frequencies": {raw_action: 1.0},
        "dominant_action": raw_action,
        "dominant_action_frequency": 1.0,
        "iterations": None,
        "exploitability_last": None,
        "candidate_confidence": "weak_rule",
        "raw_action": raw_action,
        "normalized_action": normalized_action,
        "bootstrap_label": normalized_action,
        "label_source": WEAK_LABEL_SOURCE,
        "label_quality": WEAK_LABEL_QUALITY,
        "weak_rule_reason": template["weak_rule_reason"],
        "excluded": False,
        "exclusion_reason": None,
    }
    return add_derived_features(row)


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            safe = _strip_forbidden_label_fields(row)
            handle.write(json.dumps(safe, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            safe = _strip_forbidden_label_fields(row)
            csv_row = {field: safe.get(field) for field in CSV_FIELDS}
            for field in ("hero_cards", "villain_hand", "board_cards", "action_frequencies"):
                csv_row[field] = json.dumps(csv_row[field], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)


def export_dist_aligned_candidate_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    fieldnames: Iterable[str] | None = None,
) -> dict[str, Any]:
    rows = read_candidate_csv(input_csv)
    aligned_rows = [dist_aligned_candidate_row(row, index=index) for index, row in enumerate(rows, start=1)]
    fields = merge_dist_fieldnames(fieldnames)
    write_dist_aligned_csv(aligned_rows, output_csv, fieldnames=fields)
    usable_rows = [
        row for row in aligned_rows
        if str(row.get("quality_flags.usable_for_training")).lower() == "true"
        and row.get("bootstrap_label")
    ]
    return {
        "status": "ok",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_total": len(aligned_rows),
        "rows_usable": len(usable_rows),
        "field_count": len(fields),
        "fieldnames": fields,
    }


def read_candidate_csv(input_csv: str | Path) -> list[dict[str, Any]]:
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def merge_dist_fieldnames(fieldnames: Iterable[str] | None = None) -> list[str]:
    merged: list[str] = []
    for field in list(fieldnames or []) + list(DIST_ALIGNED_FIELDS):
        if field not in merged:
            merged.append(field)
    return merged


def write_dist_aligned_csv(rows: list[dict[str, Any]], output_csv: str | Path, *, fieldnames: list[str]) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field), field=field) for field in fieldnames})


def dist_aligned_candidate_row(row: Mapping[str, Any], *, index: int = 1) -> dict[str, Any]:
    excluded = _boolish(row.get("excluded"))
    label = None if excluded else _text(row.get("bootstrap_label"))
    pot = _float_or_none(row.get("pot"))
    to_call = _float_or_none(row.get("to_call")) or 0.0
    stack = _float_or_none(row.get("stack"))
    cards = _json_value(row.get("hero_cards"))
    villain = _json_value(row.get("villain_hand"))
    board = _json_value(row.get("board_cards"))
    hero_cards = cards if isinstance(cards, list) else []
    villain_hand = villain if isinstance(villain, list) else []
    board_cards = board if isinstance(board, list) else []
    player_start = _int_or_none(row.get("player_start")) or 2
    player_active = _int_or_none(row.get("player_active")) or 2
    active_opponents = max(0, player_active - 1)
    opponent_profiles = _opponent_profiles(row, active_opponents=active_opponents)
    players = _players(row, player_start=player_start, player_active=player_active, hero_stack=stack)
    has_check = to_call <= 0
    has_call = to_call > 0
    buttons = _dist_buttons(to_call=to_call, raw_action=row.get("raw_action"))
    source_id = _text(row.get("source_id")) or f"bootstrap_v5_{index:06d}"
    street = (_text(row.get("street")) or "UNKNOWN").upper()
    position = _text(row.get("hero_position") or row.get("position_model")) or "UNKNOWN"
    label_source = _text(row.get("label_source")) or ("bootstrap_excluded" if excluded else "bootstrap_unknown")
    label_valid = bool(label)
    reason = _text(row.get("weak_rule_reason")) or "bootstrap_solver_candidate"
    equity_1v1 = _feature_float(row, "features.equity_1v1", "equity_1v1")
    if equity_1v1 is None:
        equity_1v1 = _compute_equity_1v1(hero_cards=hero_cards, villain_hand=villain_hand, board_cards=board_cards, seed=index)
    equity_known = equity_1v1 is not None
    if equity_1v1 is None:
        equity_1v1 = 0.5
    equity_table = _feature_float(row, "features.equity_table", "equity_table")
    if equity_table is None:
        equity_table = _multiway_equity(equity_1v1, active_opponents=active_opponents)
    equity_required = _equity_required(pot, to_call)
    ev = _feature_float(row, "features.ev", "ev")
    if ev is None:
        ev = _ev(equity_table=equity_table, pot=pot, to_call=to_call)
    call_max = _feature_float(row, "features.call_max", "call_max")
    if call_max is None:
        call_max = _call_max(equity_table=equity_table, pot=pot)
    aligned = {
        "schema_version": "ml_dataset_v1",
        "type": "ml_decision_snapshot",
        "snapshot_id": source_id,
        "recorded_at": "",
        "metadata.game": "bootstrap_solver",
        "metadata.hand_id": index,
        "metadata.scan_count": 1,
        "metadata.street": street,
        "metadata.status": "ok" if not excluded else "excluded",
        "metadata.decision_mode": "bootstrap",
        "metadata.label_source": label_source,
        "metadata.new_party_state": False,
        "metadata.decision_engine_version": "decision_engine_v2",
        "metadata.legacy_rules_version": "legacy_rules_v2",
        "metadata.decision_engine_fix_id": DIST_BOOTSTRAP_FIX_ID,
        "metadata.decision_engine_fix_date": DIST_BOOTSTRAP_FIX_DATE,
        "metadata.git_commit": "",
        "features.hero_cards": hero_cards,
        "features.board_cards": board_cards,
        "features.board_card_count": len([card for card in board_cards if card]),
        "features.hero_position": position,
        "features.player_start": player_start,
        "features.player_active": player_active,
        "features.active_opponents": active_opponents,
        "features.hero_stack": stack,
        "features.effective_stack": stack,
        "features.stack_to_pot_ratio": _ratio(stack, pot),
        "features.pot": pot,
        "features.to_call": to_call,
        "features.to_call_pot_ratio": _ratio(to_call, pot),
        "features.buttons": buttons,
        "features.buttons_active": [button["state"] for button in buttons if button.get("enabled")],
        "features.has_check": has_check,
        "features.has_call": has_call,
        "features.has_raise": True,
        "features.players": players,
        "features.opponent_profiles": opponent_profiles,
        "features.equity_table": equity_table,
        "features.equity_1v1": equity_1v1,
        "features.equity_known": equity_known,
        "features.equity_required": equity_required,
        "features.ev": ev,
        "features.call_max": call_max,
        "labels.legacy_action": label,
        "labels.legacy_reason": reason if label else "",
        "labels.legacy_raise_amount": "",
        "labels.ml_action": "",
        "labels.ml_confidence": "",
        "labels.final_action": label,
        "labels.fallback_reason": "",
        "labels.label_valid": label_valid,
        "labels.label_exclusion_reason": _text(row.get("exclusion_reason")) or "",
        "labels.known_bug_risk": False,
        "confidence.hero_cards_min": "",
        "confidence.board_cards_min": "",
        "confidence.pot_ocr": "",
        "confidence.to_call_ocr": "",
        "confidence.buttons_min": "",
        "confidence.hero_position": "",
        "confidence.player_count": "",
        "quality_flags.hero_cards_uncertain": len(hero_cards) == 0,
        "quality_flags.board_uncertain": len(board_cards) == 0,
        "quality_flags.opponent_count_uncertain": True,
        "quality_flags.pot_to_call_incoherent": pot is None or pot < 0 or to_call < 0,
        "quality_flags.buttons_incoherent": False,
        "quality_flags.hero_position_low_confidence": position == "UNKNOWN",
        "quality_flags.street_transient": False,
        "quality_flags.usable_for_training": label_valid and not excluded,
        "debug.decision_reason": reason if label else _text(row.get("exclusion_reason")) or "",
        "debug.scan_status": "ok" if not excluded else "excluded",
        "bootstrap_label": label,
        "excluded": excluded,
        "exclusion_reason": _text(row.get("exclusion_reason")) or "",
        "audit.source_id": source_id,
        "audit.label_source": label_source,
        "audit.label_quality": _text(row.get("label_quality")) or "",
        "audit.weak_rule_reason": _text(row.get("weak_rule_reason")) or "",
        "audit.dominant_action": _text(row.get("dominant_action")) or "",
        "audit.dominant_action_frequency": _float_or_none(row.get("dominant_action_frequency")),
        "audit.raw_action": _text(row.get("raw_action")) or "",
        "audit.normalized_action": _text(row.get("normalized_action")) or "",
        "audit.action_frequencies": _json_value(row.get("action_frequencies")) or {},
        "audit.iterations": _int_or_none(row.get("iterations")),
        "audit.exploitability_last": _float_or_none(row.get("exploitability_last")),
        "audit.candidate_confidence": _text(row.get("candidate_confidence")) or "",
    }
    return aligned


def csv_cell(value: Any, *, field: str) -> Any:
    if value is None:
        return ""
    if field in DIST_JSON_FIELDS or isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _dist_buttons(*, to_call: float, raw_action: Any) -> list[dict[str, Any]]:
    raise_value = _raise_value(raw_action) or 20.0
    if to_call > 0:
        return [
            {"index": 0, "enabled": True, "state": "relance", "value": raise_value, "text": f"raise {raise_value:g}", "confidence": None},
            {"index": 1, "enabled": True, "state": "paie", "value": to_call, "text": f"call {to_call:g}", "confidence": None},
            {"index": 2, "enabled": True, "state": "fold", "value": 0.0, "text": "fold", "confidence": None},
        ]
    return [
        {"index": 0, "enabled": True, "state": "mise", "value": raise_value, "text": f"bet {raise_value:g}", "confidence": None},
        {"index": 1, "enabled": True, "state": "check", "value": 0.0, "text": "check", "confidence": None},
        {"index": 2, "enabled": True, "state": "fold", "value": 0.0, "text": "fold", "confidence": None},
    ]


def _raise_value(raw_action: Any) -> float | None:
    text = _text(raw_action) or ""
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return _float_or_none(match.group(1)) if match else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 10)


def _equity_required(pot: float | None, to_call: float | None) -> float | None:
    if pot is None or to_call is None:
        return None
    total = pot + to_call
    if total <= 0:
        return None
    return round(to_call / total, 10)


def _compute_equity_1v1(
    *,
    hero_cards: list[Any],
    villain_hand: list[Any],
    board_cards: list[Any],
    seed: int,
) -> float | None:
    if len(hero_cards) != 2 or len(villain_hand) != 2:
        return None
    board = [card for card in board_cards if card]
    try:
        from solvers.poker_solver_adapter import compute_equity_hand_vs_hand

        result = compute_equity_hand_vs_hand(
            hero_cards,
            villain_hand,
            board=board,
            iterations=2_000,
            seed=seed,
        )
    except Exception:
        return None
    if result.get("status") != "ok":
        return None
    output = result.get("output") if isinstance(result.get("output"), Mapping) else {}
    equity = _float_or_none(output.get("hero_equity"))
    if equity is None:
        return None
    return round(max(0.0, min(1.0, equity)), 10)


def _multiway_equity(equity_1v1: float | None, *, active_opponents: int) -> float | None:
    if equity_1v1 is None:
        return None
    opponents = max(1, int(active_opponents or 1))
    return round(max(0.0, min(1.0, float(equity_1v1) ** opponents)), 10)


def _ev(*, equity_table: float | None, pot: float | None, to_call: float | None) -> float | None:
    if equity_table is None or pot is None or to_call is None:
        return None
    return round(float(equity_table) * (float(pot) + float(to_call)) - float(to_call), 10)


def _call_max(*, equity_table: float | None, pot: float | None) -> float | None:
    if equity_table is None or pot is None:
        return None
    equity = max(0.0, min(0.999999, float(equity_table)))
    return round((equity * float(pot)) / (1.0 - equity), 10)


def _feature_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _opponent_profiles(row: Mapping[str, Any], *, active_opponents: int) -> list[dict[str, Any]]:
    existing = _json_value(row.get("features.opponent_profiles") or row.get("opponent_profiles"))
    if isinstance(existing, list) and existing:
        return [dict(item) for item in existing if isinstance(item, Mapping)]
    count = max(0, int(active_opponents or 0))
    return [
        {
            "action": "paid",
            "looseness": 0.5,
            "aggression": 0.5,
            "confidence": 0.0,
        }
        for _ in range(count)
    ]


def _players(row: Mapping[str, Any], *, player_start: int, player_active: int, hero_stack: float | None) -> list[dict[str, Any]]:
    existing = _json_value(row.get("features.players") or row.get("players"))
    if isinstance(existing, list) and existing:
        return [dict(item) for item in existing if isinstance(item, Mapping)]
    total = max(1, int(player_start or player_active or 1))
    active = max(1, min(total, int(player_active or total)))
    stack = hero_stack if hero_stack is not None else 0.0
    players = []
    for seat in range(1, total + 1):
        is_active = seat <= active
        players.append(
            {
                "seat": seat,
                "state": "paid" if is_active else "fold",
                "active": is_active,
                "active_at_start": True,
                "stack": stack,
                "stack_start": stack,
            }
        )
    return players


def _json_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_export(
    rows: list[dict[str, Any]],
    *,
    warnings: list[str],
    output_jsonl: str | Path,
    output_csv: str | Path | None,
    input_records_count: int,
) -> dict[str, Any]:
    excluded = [row for row in rows if row["excluded"]]
    exported = [row for row in rows if not row["excluded"]]
    reasons = Counter(str(row["exclusion_reason"]) for row in excluded)
    label_counts = Counter(str(row["bootstrap_label"]) for row in exported if row.get("bootstrap_label"))
    source_counts = Counter(str(row["label_source"]) for row in exported if row.get("label_source"))
    quality_counts = Counter(str(row["label_quality"]) for row in exported if row.get("label_quality"))
    context_counts = Counter(str(row["decision_context_type"]) for row in exported if row.get("decision_context_type"))
    street_counts = Counter(str(row["street"]) for row in exported if row.get("street"))
    weak_count = source_counts.get(WEAK_LABEL_SOURCE, 0)
    solver_count = source_counts.get(LABEL_SOURCE, 0)
    missing_fields = missing_field_counts(exported)
    warnings_out = sorted(set(warnings + dataset_warnings(rows, exported, label_counts, source_counts, missing_fields)))
    return {
        "status": "ok",
        "input_records_count": input_records_count,
        "rows_total": len(rows),
        "candidates_exported": len(exported),
        "excluded_count": len(excluded),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "normalized_label_distribution": dict(sorted(label_counts.items())),
        "label_source_counts": dict(sorted(source_counts.items())),
        "label_quality_counts": dict(sorted(quality_counts.items())),
        "context_distribution": dict(sorted(context_counts.items())),
        "street_distribution": dict(sorted(street_counts.items())),
        "class_count": len(label_counts),
        "weak_rule_row_rate": round(weak_count / len(exported), 6) if exported else 0.0,
        "solver_candidate_row_rate": round(solver_count / len(exported), 6) if exported else 0.0,
        "missing_fields": dict(sorted(missing_fields.items())),
        "output_jsonl": str(output_jsonl),
        "output_csv": None if output_csv is None else str(output_csv),
        "warnings": warnings_out,
        "not_gto": True,
        "not_for_production": True,
    }


def write_dataset_reports(rows: list[dict[str, Any]], summary: Mapping[str, Any], output_jsonl: str | Path) -> dict[str, str]:
    directory = Path(output_jsonl).parent
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "dataset_report.json"
    md_path = directory / "dataset_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_dataset_report_md(summary), encoding="utf-8")
    return {
        "dataset_report_json": str(json_path),
        "dataset_report_md": str(md_path),
    }


def render_dataset_report_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Bootstrap Candidate Dataset Report",
        "",
        "This dataset is bootstrap-only. It is not GTO, not production data, and not a reliable poker strategy.",
        "",
        f"- input_records_count: `{summary['input_records_count']}`",
        f"- rows_total: `{summary['rows_total']}`",
        f"- candidates_exported: `{summary['candidates_exported']}`",
        f"- excluded_count: `{summary['excluded_count']}`",
        f"- class_count: `{summary['class_count']}`",
        f"- weak_rule_row_rate: `{summary['weak_rule_row_rate']}`",
        f"- solver_candidate_row_rate: `{summary['solver_candidate_row_rate']}`",
        "",
        "## Label Distribution",
        "",
        "```json",
        json.dumps(summary["normalized_label_distribution"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Label Sources",
        "",
        "```json",
        json.dumps(summary["label_source_counts"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Exclusions",
        "",
        "```json",
        json.dumps(summary["exclusion_reasons"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- `{warning}`" for warning in summary["warnings"])
    lines.extend(
        [
            "",
            "## Missing Fields",
            "",
            "```json",
            json.dumps(summary["missing_fields"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def dataset_warnings(
    rows: list[dict[str, Any]],
    exported: list[dict[str, Any]],
    label_counts: Counter[str],
    source_counts: Counter[str],
    missing_fields: Counter[str],
) -> list[str]:
    warnings = ["not_gto", "not_for_production"]
    if source_counts.get(WEAK_LABEL_SOURCE, 0) > 0:
        warnings.extend(["dataset_contains_weak_rule_labels", "synthetic_distribution_bias"])
    if source_counts.get(LABEL_SOURCE, 0) > 0:
        warnings.append("bootstrap_solver_untrusted")
    if any(row.get("exclusion_reason") == "all_in_excluded" for row in rows):
        warnings.append("all_in_excluded")
    if len(exported) < 500:
        warnings.append("small_dataset")
    if "CALL" not in label_counts:
        warnings.append("call_class_absent")
    if any(count > 0 for field, count in missing_fields.items() if field in {"hero_cards", "board_cards", "villain_hand"}):
        warnings.append("card_fields_missing_or_partial")
    return warnings


def missing_field_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    fields = (
        "hero_cards",
        "villain_hand",
        "board_cards",
        "exploitability_last",
        "iterations",
        "candidate_confidence",
    )
    counts: Counter[str] = Counter()
    for row in rows:
        for field in fields:
            if _is_missing_value(row.get(field)):
                counts[field] += 1
    return counts


def add_derived_features(row: dict[str, Any]) -> dict[str, Any]:
    pot = _float_or_none(row.get("pot"))
    to_call = _float_or_none(row.get("to_call"))
    stack = _float_or_none(row.get("stack"))
    street = (_text(row.get("street")) or "").upper()
    context = _text(row.get("decision_context_type")) or ""
    board_cards = row.get("board_cards")
    row["board_card_count"] = len(board_cards) if isinstance(board_cards, list) else 0
    row["is_river"] = street == "RIVER"
    row["is_turn"] = street == "TURN"
    row["is_check_or_bet_context"] = context == "hero_check_or_bet"
    row["is_facing_bet_context"] = context == "hero_facing_bet"
    row["to_call_ratio"] = round(float(to_call) / float(pot), 6) if pot and pot > 0 and to_call is not None else None
    row["stack_to_pot_ratio"] = round(float(stack) / float(pot), 6) if pot and pot > 0 and stack is not None else None
    return row


def calculate_spr(stack: Any, pot: Any) -> float | None:
    stack_value = _float_or_none(stack)
    pot_value = _float_or_none(pot)
    if stack_value is None or pot_value is None or pot_value <= 0:
        return None
    return round(stack_value / pot_value, 6)


def group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("group_id") or row.get("source_id") or "unknown")


def _allowed_non_all_in_action(action: str) -> bool:
    return action in ALLOWED_ACTIONS or action.startswith(ALLOWED_ACTION_PREFIXES)


def normalize_bootstrap_action(value: Any) -> str | None:
    action = _text(value)
    if action is None:
        return None
    action = action.upper().replace("-", "_").replace(" ", "_")
    if action == "ALL_IN":
        return None
    if action in ALLOWED_ACTIONS:
        return action
    if action.startswith(ALLOWED_ACTION_PREFIXES):
        return "RAISE"
    return None


def _confidence_from_frequency(value: Any) -> str:
    frequency = _float_or_none(value)
    if frequency is None:
        return "unknown"
    if frequency >= 0.75:
        return "high"
    if frequency >= 0.55:
        return "medium"
    return "low"


def math_ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _position_from_context(context: Any) -> str | None:
    text = str(context or "")
    if "_ip_" in text:
        return "IP"
    if "_oop_" in text:
        return "OOP"
    return None


def _decision_type_from_context(context: Any) -> str | None:
    text = str(context or "")
    if "check_or_bet" in text:
        return "hero_check_or_bet"
    if "facing_bet" in text:
        return "hero_facing_bet"
    return None


def _group_from_job_id(value: Any) -> str:
    text = str(value or "unknown")
    return re.sub(r"_it\d+$", "", text)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _jsonable_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _strip_forbidden_label_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(row)
    safe.pop("training_label", None)
    safe.pop("gto_label", None)
    return safe


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    text = str(value).strip().lower()
    return text in {"", "null", "none", "[]"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input JSONL solver run files")
    parser.add_argument("--output-jsonl", default="outputs/bootstrap_candidate_dataset/candidates.jsonl")
    parser.add_argument("--output-csv", default="outputs/bootstrap_candidate_dataset/candidates.csv")
    parser.add_argument("--include-weak-rules", action="store_true")
    parser.add_argument("--min-usable-rows", type=int, default=DEFAULT_V2_MIN_USABLE_ROWS)
    parser.add_argument("--class-floor", type=int, default=DEFAULT_V2_CLASS_FLOOR)
    parser.add_argument("--min-dominant-frequency", type=float, default=MIN_DOMINANT_FREQUENCY)
    parser.add_argument("--balance-weak-rules", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_candidate_dataset(
        args.inputs,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        include_weak_rules=args.include_weak_rules,
        min_usable_rows=args.min_usable_rows,
        class_floor=args.class_floor,
        min_dominant_frequency=args.min_dominant_frequency,
        balance_weak_rules=args.balance_weak_rules,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
