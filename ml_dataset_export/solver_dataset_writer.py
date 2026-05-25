"""Strict JSONL writer for solver-labeled ml_decision_snapshot rows."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from . import feature_rebuilder
    from .solver_adapter import (
        SolverAdapter,
        SolverDecision,
        SolverSpot,
        TRAINING_ACTIONS,
        action_allowed_by_buttons,
        normalize_solver_action,
    )
except ImportError:  # pragma: no cover - direct script/module execution
    import feature_rebuilder
    from solver_adapter import (
        SolverAdapter,
        SolverDecision,
        SolverSpot,
        TRAINING_ACTIONS,
        action_allowed_by_buttons,
        normalize_solver_action,
    )


PIPELINE_VERSION = "solver_dataset_pipeline_v1"
SCHEMA_VERSION = "ml_dataset_v1"
STREET_BOARD_COUNT = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}


class DatasetRowRejected(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def build_dataset_row(
    spot: SolverSpot,
    decision: SolverDecision,
    *,
    index: int,
    solver_name: str = "solver",
    solver_version: str = "unknown",
) -> OrderedDict[str, Any]:
    features = _observable_features(spot)
    reasons = _spot_rejection_reasons(spot, features)

    action = normalize_solver_action(decision.action)
    if action is None:
        reasons.append(f"invalid_solver_action:{decision.action}")
    elif not action_allowed_by_buttons(action, features["buttons_active"]):
        reasons.append(f"action_impossible:{action}")

    if action == "WAIT" or str(decision.action).strip().upper() == "WAIT":
        reasons.append("wait_action_rejected")

    if reasons:
        raise DatasetRowRejected(reasons)

    raise_amount = float(decision.raise_amount) if action == "RAISE" and decision.raise_amount is not None else None
    recorded_at = datetime.now(timezone.utc).isoformat()
    snapshot_id = f"solver:{spot.source_snapshot_id or index}"
    decision_reason = f"{solver_name}:{solver_version}:{(decision.raw or {}).get('rule', 'solver_label')}"

    return OrderedDict(
        [
            ("schema_version", SCHEMA_VERSION),
            ("type", "ml_decision_snapshot"),
            ("snapshot_id", snapshot_id),
            ("recorded_at", recorded_at),
            (
                "metadata",
                OrderedDict(
                    [
                        ("game", "PokerTH"),
                        ("hand_id", index),
                        ("scan_count", 1),
                        ("street", spot.street.upper()),
                        ("status", "ok"),
                        ("decision_mode", "solver"),
                        ("label_source", "solver"),
                        ("new_party_state", False),
                        ("decision_engine_version", PIPELINE_VERSION),
                        ("legacy_rules_version", "legacy_rules_v2"),
                        ("decision_engine_fix_id", PIPELINE_VERSION),
                        ("decision_engine_fix_date", "2026-05-24"),
                        ("git_commit", None),
                    ]
                ),
            ),
            ("features", features),
            (
                "labels",
                OrderedDict(
                    [
                        ("legacy_action", action),
                        ("legacy_reason", "solver_label"),
                        ("legacy_raise_amount", raise_amount),
                        ("ml_action", None),
                        ("ml_confidence", None),
                        ("final_action", action),
                        ("fallback_reason", None),
                        ("label_valid", True),
                        ("label_exclusion_reason", None),
                        ("known_bug_risk", False),
                    ]
                ),
            ),
            (
                "confidence",
                OrderedDict(
                    [
                        ("hero_cards_min", 0.95),
                        ("board_cards_min", None if spot.street.upper() == "PREFLOP" else 0.95),
                        ("pot_ocr", 0.9),
                        ("to_call_ocr", 0.9),
                        ("buttons_min", 0.9),
                        ("hero_position", 0.8 if spot.hero_position else 0.0),
                        ("player_count", 0.8),
                    ]
                ),
            ),
            (
                "quality_flags",
                OrderedDict(
                    [
                        ("hero_cards_uncertain", False),
                        ("board_uncertain", False),
                        ("opponent_count_uncertain", False),
                        ("pot_to_call_incoherent", False),
                        ("buttons_incoherent", False),
                        ("hero_position_low_confidence", not bool(spot.hero_position)),
                        ("street_transient", False),
                        ("usable_for_training", True),
                    ]
                ),
            ),
            ("debug", OrderedDict([("decision_reason", decision_reason), ("scan_status", "ok")])),
        ]
    )


def write_solver_dataset(
    spots: Iterable[SolverSpot],
    solver: SolverAdapter,
    output: str | Path,
    *,
    n_rows: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for spot in spots:
            if n_rows is not None and written >= n_rows:
                break
            decision = solver.solve_spot(spot)
            try:
                row = build_dataset_row(spot, decision, index=written + 1, solver_name=solver.name, solver_version=solver.version)
            except DatasetRowRejected as exc:
                rejected.append({"spot": spot.source_snapshot_id, "reasons": exc.reasons})
                continue
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return {"output": str(output_path), "written": written, "rejected": rejected}


def _observable_features(spot: SolverSpot) -> OrderedDict[str, Any]:
    equity_table = _equity_table(spot)
    equity_1v1 = _equity_1v1(spot, equity_table)
    players = spot.players or _default_players(spot.player_active or 1, spot.player_start or spot.player_active or 1)

    features: OrderedDict[str, Any] = OrderedDict(
        [
            ("hero_cards", list(spot.hero_cards)),
            ("board_cards", list(spot.board_cards)),
            ("hero_position", spot.hero_position),
            ("player_start", spot.player_start),
            ("player_active", spot.player_active),
            ("pot", _round_float(spot.pot)),
            ("to_call", _round_float(spot.to_call)),
            ("to_call_pot_ratio", None),
            ("buttons", _buttons(spot)),
            ("buttons_active", list(spot.buttons_active or [])),
            ("has_check", None),
            ("has_call", None),
            ("has_raise", None),
            ("players", players),
            ("opponent_profiles", _opponent_profiles(spot, players)),
            ("equity_table", equity_table),
            ("equity_1v1", equity_1v1),
            ("equity_required", None),
            ("ev", None),
            ("call_max", None),
        ]
    )
    rebuilt = feature_rebuilder.rebuild_derived_features(features, recompute_equity=False)
    for key in ("to_call_pot_ratio", "buttons_active", "has_check", "has_call", "has_raise", "opponent_profiles", "equity_required", "ev", "call_max"):
        if key in rebuilt:
            features[key] = _round_float(rebuilt[key]) if key in {"to_call_pot_ratio", "equity_required", "ev", "call_max"} else rebuilt[key]
    features["opponent_profiles"] = [_strict_profile(profile) for profile in features["opponent_profiles"]]
    features["equity_table"] = _round_float(features["equity_table"])
    features["equity_1v1"] = _round_float(features["equity_1v1"])
    return features


def _spot_rejection_reasons(spot: SolverSpot, features: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    parsed_cards = [feature_rebuilder.parse_card(card) for card in [*spot.hero_cards, *spot.board_cards]]
    if len(spot.hero_cards) != 2 or any(card is None for card in parsed_cards[:2]):
        reasons.append("invalid_hero_cards")
    street = spot.street.upper()
    expected_board_count = STREET_BOARD_COUNT.get(street)
    if expected_board_count is None:
        reasons.append("invalid_street")
    elif len(spot.board_cards) != expected_board_count:
        reasons.append("board_street_mismatch")
    if any(card is None for card in parsed_cards[2:]):
        reasons.append("invalid_board_cards")
    card_keys = [(card.rank, card.suit) for card in parsed_cards if card is not None]
    if len(card_keys) != len(set(card_keys)):
        reasons.append("duplicate_cards")
    if spot.pot is None or spot.pot <= 0 or spot.to_call is None or spot.to_call < 0:
        reasons.append("invalid_pot_or_to_call")
    if spot.player_active is None or spot.player_active < 1 or spot.player_start is None or spot.player_start < spot.player_active:
        reasons.append("invalid_player_counts")
    if not spot.hero_position:
        reasons.append("missing_hero_position")
    if not features.get("buttons_active"):
        reasons.append("missing_buttons")
    return reasons


def _buttons(spot: SolverSpot) -> list[dict[str, Any]]:
    buttons = spot.buttons or []
    normalized: list[dict[str, Any]] = []
    for index, button in enumerate(buttons):
        data = dict(button)
        normalized.append(
            OrderedDict(
                [
                    ("index", int(data.get("index", index))),
                    ("enabled", bool(data.get("enabled", True))),
                    ("state", str(data.get("state") or "").lower()),
                    ("value", _round_float(data.get("value"))),
                    ("text", str(data.get("text") or data.get("state") or "")),
                    ("confidence", data.get("confidence")),
                ]
            )
        )
    return normalized


def _opponent_profiles(spot: SolverSpot, players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = spot.opponent_profiles
    if isinstance(profiles, list):
        return [dict(profile) for profile in profiles if isinstance(profile, dict)]
    rebuilt = feature_rebuilder.opponent_profiles_from_players(players)
    return [
        OrderedDict(
            [
                ("action", profile.action),
                ("looseness", profile.looseness),
                ("aggression", profile.aggression),
                ("confidence", profile.confidence),
            ]
        )
        for profile in rebuilt
    ]


def _strict_profile(profile: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("action", str(profile.get("action") or "play")),
            ("looseness", _round_float(profile.get("looseness"))),
            ("aggression", _round_float(profile.get("aggression"))),
            ("confidence", _round_float(profile.get("confidence"))),
        ]
    )


def _default_players(active_count: int, start_count: int) -> list[dict[str, Any]]:
    return [
        OrderedDict(
            [
                ("seat", seat),
                ("state", "paid" if seat <= active_count else "fold"),
                ("active", seat <= active_count),
                ("active_at_start", True),
                ("stack", 5000.0),
                ("stack_start", 5000.0),
            ]
        )
        for seat in range(1, start_count + 1)
    ]


def _equity_table(spot: SolverSpot) -> float:
    if spot.equity_table is not None:
        return _round_float(spot.equity_table)
    project_equity = feature_rebuilder.estimate_project_equity(
        hero_cards=spot.hero_cards,
        board_cards=spot.board_cards,
        opponent_profiles=[],
        opponent_count=spot.player_active,
        simulations=600,
    )
    if project_equity is not None:
        return _round_float(project_equity)
    base = feature_rebuilder.starting_hand_strength_from_text(spot.hero_cards)
    base = 0.5 if base is None else base
    board_adjustment = 0.02 * len(spot.board_cards)
    opponent_penalty = 0.055 * max(0, int(spot.player_active or 1) - 1)
    return _round_float(max(0.03, min(0.97, base + board_adjustment - opponent_penalty)))


def _equity_1v1(spot: SolverSpot, equity_table: float) -> float:
    if spot.equity_1v1 is not None:
        return _round_float(spot.equity_1v1)
    base = feature_rebuilder.starting_hand_strength_from_text(spot.hero_cards)
    base = equity_table if base is None else base
    return _round_float(max(0.03, min(0.97, base + 0.08)))


def _round_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 10)
    except (TypeError, ValueError):
        return None
