"""Spot generation and import helpers for solver dataset creation."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterable

try:
    from .solver_adapter import SolverSpot
except ImportError:  # pragma: no cover - direct script/module execution
    from solver_adapter import SolverSpot


RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["\u2660", "\u2665", "\u2666", "\u2663"]
POSITIONS = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
STREET_BOARD_COUNT = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}


def generate_synthetic_spots(n_spots: int, *, seed: int | None = None) -> list[SolverSpot]:
    rng = random.Random(seed)
    spots: list[SolverSpot] = []
    for index in range(n_spots):
        deck = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]
        rng.shuffle(deck)
        hero_cards = [deck.pop(), deck.pop()]
        street = rng.choice(list(STREET_BOARD_COUNT))
        board_cards = [deck.pop() for _ in range(STREET_BOARD_COUNT[street])]
        pot = float(rng.randrange(4, 51) * 10)
        to_call = 0.0 if rng.random() < 0.45 else float(rng.randrange(1, max(2, int(pot // 20) + 1)) * 10)
        buttons = _buttons_for_state(to_call, pot)
        player_active = rng.randint(1, 4)
        player_start = rng.randint(player_active, 5)
        players = _players(player_active, player_start)

        spots.append(
            SolverSpot(
                hero_cards=hero_cards,
                board_cards=board_cards,
                street=street,
                hero_position=rng.choice(POSITIONS),
                player_active=player_active,
                player_start=player_start,
                pot=pot,
                to_call=to_call,
                buttons=buttons,
                buttons_active=[str(button["state"]) for button in buttons if button.get("enabled")],
                players=players,
                source_snapshot_id=f"synthetic:{index + 1}",
            )
        )
    return spots


def iter_existing_spots(path: str | Path) -> Iterable[SolverSpot]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            spot = spot_from_row(row)
            if spot is not None:
                yield spot


def spot_from_row(row: dict[str, Any]) -> SolverSpot | None:
    if row.get("type") != "ml_decision_snapshot":
        return None
    features = row.get("features")
    if not isinstance(features, dict):
        return None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    hero_cards = [str(card) for card in features.get("hero_cards") or [] if card]
    board_cards = [str(card) for card in features.get("board_cards") or [] if card]
    return SolverSpot(
        hero_cards=hero_cards,
        board_cards=board_cards,
        street=str(metadata.get("street") or features.get("street") or "").upper(),
        hero_position=features.get("hero_position"),
        player_active=_int_or_none(features.get("player_active")),
        player_start=_int_or_none(features.get("player_start")),
        pot=_float_or_none(features.get("pot")),
        to_call=_float_or_none(features.get("to_call")),
        buttons=_list_of_dicts(features.get("buttons")),
        buttons_active=[str(item) for item in features.get("buttons_active") or []],
        opponent_profiles=features.get("opponent_profiles"),
        players=_list_of_dicts(features.get("players")),
        equity_table=_float_or_none(features.get("equity_table")),
        equity_1v1=_float_or_none(features.get("equity_1v1")),
        source_snapshot_id=str(row.get("snapshot_id") or ""),
    )


def _buttons_for_state(to_call: float, pot: float) -> list[dict[str, Any]]:
    if to_call > 0:
        raise_value = max(to_call * 2.0, round(pot * 0.5, 2))
        return [
            {"index": 0, "enabled": True, "state": "relance", "value": float(raise_value), "text": f"raise {raise_value:g}", "confidence": None},
            {"index": 1, "enabled": True, "state": "paie", "value": float(to_call), "text": f"call {to_call:g}", "confidence": None},
            {"index": 2, "enabled": True, "state": "fold", "value": 0.0, "text": "fold", "confidence": None},
        ]
    bet_value = max(10.0, round(pot * 0.5, 2))
    return [
        {"index": 0, "enabled": True, "state": "mise", "value": float(bet_value), "text": f"bet {bet_value:g}", "confidence": None},
        {"index": 1, "enabled": True, "state": "check", "value": 0.0, "text": "check", "confidence": None},
        {"index": 2, "enabled": True, "state": "fold", "value": 0.0, "text": "fold", "confidence": None},
    ]


def _players(active_count: int, start_count: int) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for seat in range(1, start_count + 1):
        active = seat <= active_count
        players.append(
            {
                "seat": seat,
                "state": "paid" if active else "fold",
                "active": active,
                "active_at_start": True,
                "stack": 5000.0 - (seat * 10.0),
                "stack_start": 5000.0,
            }
        )
    return players


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value] if isinstance(value, list) and all(isinstance(item, dict) for item in value) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return None if number is None else int(number)

