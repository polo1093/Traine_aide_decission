"""Rebuild derived ML dataset features outside the poker bot.

This file is intentionally standalone: it can be copied with a JSONL dataset
into another project.  The light features are rebuilt with pure Python.  Exact
table equity can optionally reuse the original project modules when this file
is executed from the poker project checkout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Optional, Sequence


TRAINING_ACTIONS = {"CHECK", "CALL", "FOLD", "RAISE"}
RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
VALUE_TO_NOTATION = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
SUIT_TO_VALUE = {
    "s": 1,
    "spade": 1,
    "spades": 1,
    "\u2660": 1,
    "h": 2,
    "heart": 2,
    "hearts": 2,
    "\u2665": 2,
    "d": 3,
    "diamond": 3,
    "diamonds": 3,
    "\u2666": 3,
    "c": 4,
    "club": 4,
    "clubs": 4,
    "\u2663": 4,
}
MOJIBAKE_SUIT_TO_VALUE = {
    "\u00e2\u2122\u00a0": 1,
    "\u00e2\u2122\u00a5": 2,
    "\u00e2\u2122\u00a6": 3,
    "\u00e2\u2122\u00a3": 4,
}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int


@dataclass(frozen=True)
class OpponentProfile:
    name: str = "opponent"
    looseness: float = 0.55
    aggression: float = 0.35
    action: str = "play"
    confidence: float = 0.0


def rebuild_derived_features(
    row: dict[str, Any],
    *,
    recompute_equity: bool = False,
    simulations: int = 600,
) -> dict[str, Any]:
    """Return features that can be rebuilt from a dataset row.

    ``row`` may be a complete ``ml_decision_snapshot`` or directly its
    ``features`` object.  By default the function recomputes only deterministic,
    cheap features.  Set ``recompute_equity=True`` to call the original project
    equity engine when it is available in the Python environment.
    """

    features = _features_object(row)
    pot = number_or_none(features.get("pot"))
    to_call = number_or_none(features.get("to_call"))
    amount_context = infer_amount_context(features)
    amount_unit_value = amount_context["value"]
    players = list_or_empty(features.get("players"))
    buttons = list_or_empty(features.get("buttons"))
    hero_cards = list_or_empty(features.get("hero_cards"))
    board_cards = list_or_empty(features.get("board_cards"))

    rebuilt = {
        "amount_unit": amount_context["unit"],
        "amount_unit_value": amount_unit_value,
        "amount_unit_source": amount_context["source"],
        "pot_bb": amount_in_unit(pot, amount_unit_value),
        "to_call_bb": amount_in_unit(to_call, amount_unit_value),
        "to_call_pot_ratio": to_call_pot_ratio(to_call, pot),
        "equity_required": equity_required(pot, to_call),
        "buttons_active": active_button_states(buttons),
        "has_check": has_button_state(buttons, {"check"}),
        "has_call": has_button_state(buttons, {"paie", "call"}),
        "has_raise": has_button_state(buttons, {"mise", "relance", "raise", "all-in"}),
        "opponent_profiles": [asdict(profile) for profile in opponent_profiles_from_players(players)],
        "starting_hand_strength": starting_hand_strength_from_text(hero_cards),
    }

    equity = number_or_none(features.get("equity_table"))
    if recompute_equity:
        equity = estimate_project_equity(
            hero_cards=hero_cards,
            board_cards=board_cards,
            opponent_profiles=rebuilt["opponent_profiles"],
            opponent_count=number_or_none(features.get("player_active")),
            simulations=simulations,
        )

    rebuilt["equity_table"] = equity
    rebuilt["ev"] = expected_value(equity, pot, to_call)
    rebuilt["call_max"] = call_max(equity, pot)
    rebuilt["ev_bb"] = amount_in_unit(rebuilt["ev"], amount_unit_value)
    rebuilt["call_max_bb"] = amount_in_unit(rebuilt["call_max"], amount_unit_value)
    return rebuilt


def is_trainable_row(row: dict[str, Any]) -> bool:
    """Match the filter used by the live ``training_dataset.jsonl`` export."""

    labels = dict_or_empty(row.get("labels"))
    flags = dict_or_empty(row.get("quality_flags"))
    action = str(labels.get("legacy_action") or labels.get("final_action") or "").upper()
    return (
        row.get("type") == "ml_decision_snapshot"
        and action in TRAINING_ACTIONS
        and labels.get("label_valid") is True
        and labels.get("known_bug_risk") is False
        and flags.get("usable_for_training") is True
        and flags.get("amount_unit_missing") is not True
    )


def infer_amount_context(features: dict[str, Any]) -> dict[str, Any]:
    """Infer the money unit used to normalize raw amounts into big blinds."""

    existing = number_or_none(features.get("amount_unit_value"))
    if existing is not None and existing > 0:
        return {
            "unit": features.get("amount_unit") or "big_blind",
            "value": existing,
            "source": features.get("amount_unit_source") or "existing",
        }

    street = str(features.get("street") or "").upper()
    to_call = number_or_none(features.get("to_call"))
    pot = number_or_none(features.get("pot"))
    if street == "PREFLOP" and to_call is not None and to_call > 0:
        if str(features.get("hero_position") or "").upper() == "SB" and pot is not None and pot > 0 and pot / to_call <= 4.0:
            return {"unit": "big_blind", "value": to_call * 2.0, "source": "preflop_small_blind_to_call"}
        return {"unit": "big_blind", "value": to_call, "source": "preflop_to_call"}

    values = [
        value
        for button in list_or_empty(features.get("buttons"))
        for value in [number_or_none(dict_or_empty(button).get("value"))]
        if value is not None and value > 0
    ]
    if values:
        return {"unit": "big_blind", "value": min(values), "source": "button_value"}

    return {"unit": "big_blind", "value": None, "source": None}


def amount_in_unit(value: Any, unit_value: Any) -> Optional[float]:
    number = number_or_none(value)
    unit = number_or_none(unit_value)
    if number is None or unit is None or unit <= 0:
        return None
    return round(number / unit, 6)


def to_call_pot_ratio(to_call: Any, pot: Any) -> Optional[float]:
    to_call_number = number_or_none(to_call)
    pot_number = number_or_none(pot)
    if to_call_number is None or pot_number is None or pot_number <= 0:
        return None
    return to_call_number / pot_number


def equity_required(pot: Any, to_call: Any) -> Optional[float]:
    pot_number = number_or_none(pot)
    to_call_number = number_or_none(to_call)
    if pot_number is None or to_call_number is None:
        return None
    pot_final = pot_number + to_call_number
    if pot_final <= 0:
        return None
    return to_call_number / pot_final


def expected_value(equity: Any, pot: Any, to_call: Any) -> Optional[float]:
    equity_number = number_or_none(equity)
    pot_number = number_or_none(pot)
    to_call_number = number_or_none(to_call)
    if equity_number is None or pot_number is None or to_call_number is None:
        return None
    return equity_number * (pot_number + to_call_number) - to_call_number


def call_max(equity: Any, pot: Any) -> Optional[float]:
    equity_number = number_or_none(equity)
    pot_number = number_or_none(pot)
    if equity_number is None or pot_number is None:
        return None
    if equity_number >= 1:
        return math.inf
    return (equity_number * pot_number) / (1.0 - equity_number)


def active_button_states(buttons: Sequence[Any]) -> list[str]:
    states: list[str] = []
    for button in buttons:
        data = dict_or_empty(button)
        if not data.get("enabled", bool(data.get("text"))):
            continue
        state = button_state(data)
        if state:
            states.append(state)
    return states


def has_button_state(buttons: Sequence[Any], expected_states: set[str]) -> Optional[bool]:
    if not buttons:
        return None
    expected = {state.lower() for state in expected_states}
    return any(state in expected for state in active_button_states(buttons))


def button_state(button: Any) -> Optional[str]:
    data = dict_or_empty(button)
    state = data.get("state")
    if state:
        return str(state).lower()
    return button_state_from_text(data.get("text"))


def button_state_from_text(text: Any) -> Optional[str]:
    if not text:
        return None
    lowered = str(text).lower()
    if "all-in" in lowered or "all in" in lowered:
        return "all-in"
    if "relance" in lowered or "raise" in lowered:
        return "relance"
    if "mise" in lowered or "bet" in lowered:
        return "mise"
    if "paie" in lowered or "call" in lowered:
        return "paie"
    if "check" in lowered:
        return "check"
    if "fold" in lowered or "couche" in lowered:
        return "fold"
    return None


def opponent_profiles_from_players(players: Iterable[Any]) -> list[OpponentProfile]:
    """Rebuild the same compact opponent profiles as the poker project."""

    profiles: list[OpponentProfile] = []
    for index, player in enumerate(players, start=1):
        data = dict_or_empty(player)
        if data.get("active") is False:
            continue

        action = str(data.get("state") or data.get("etat") or "play").lower()
        name = str(data.get("name") or f"J{index}")
        looseness = 0.55
        aggression = 0.35

        if action in {"paid", "call", "paie"}:
            looseness = 0.36
            aggression = 0.68
        elif action in {"raise", "relance", "mise", "all-in"}:
            looseness = 0.24
            aggression = 0.92
        elif action in {"check", "play"}:
            looseness = 0.58
            aggression = 0.32

        if bool(data.get("etat_modified_this_round")):
            aggression += 0.08

        start_amount = number_or_none(data.get("stack_start", data.get("fond_start_Party")))
        current_amount = number_or_none(data.get("stack", data.get("fond")))
        if start_amount and current_amount is not None and current_amount < start_amount:
            invested_ratio = min(1.0, max(0.0, (start_amount - current_amount) / start_amount))
            aggression += 0.18 * invested_ratio
            looseness -= 0.10 * invested_ratio

        profiles.append(
            OpponentProfile(
                name=name,
                looseness=clamp01(looseness),
                aggression=clamp01(aggression),
                action=action,
                confidence=number_or_none(data.get("confidence")) or 0.0,
            )
        )

    return profiles


def profiles_for_opponent_count(
    profiles: Sequence[OpponentProfile | dict[str, Any]],
    opponent_count: Any,
) -> list[OpponentProfile]:
    count_number = number_or_none(opponent_count)
    count = max(0, int(count_number or 0))
    normalized = [_profile_from_any(profile) for profile in profiles[:count]]
    while len(normalized) < count:
        normalized.append(OpponentProfile(name=f"opponent_{len(normalized) + 1}"))
    return normalized


def starting_hand_strength_from_text(cards: Sequence[Any]) -> Optional[float]:
    parsed = [card for card in (parse_card(card_text) for card_text in cards) if card is not None]
    if len(parsed) != 2:
        return None
    return starting_hand_strength(parsed[0], parsed[1])


def starting_hand_strength(left: Card, right: Card) -> float:
    """Fallback preflop strength formula used when no ranking file is loaded."""

    high, low = sorted((int(left.rank), int(right.rank)), reverse=True)
    suited = int(left.suit) == int(right.suit)

    if high == low:
        return clamp01(0.44 + ((high - 2) / 12) * 0.52)

    high_score = ((high - 2) / 12) * 0.34
    low_score = ((low - 2) / 12) * 0.18
    gap = max(0, high - low - 1)
    connected = max(0.0, 0.13 - 0.026 * gap)
    suited_bonus = 0.08 if suited else 0.0
    broadway_bonus = 0.12 if low >= 10 else 0.0
    ace_bonus = 0.06 if high == 14 else 0.0
    return clamp01(0.05 + high_score + low_score + connected + suited_bonus + broadway_bonus + ace_bonus)


def preflop_notation(cards: Sequence[Any]) -> Optional[str]:
    parsed = [card for card in (parse_card(card_text) for card_text in cards) if card is not None]
    if len(parsed) != 2:
        return None
    left, right = parsed
    high, low = sorted((int(left.rank), int(right.rank)), reverse=True)
    if high == low:
        return f"{VALUE_TO_NOTATION[high]}{VALUE_TO_NOTATION[low]}"
    suffix = "s" if left.suit == right.suit else "o"
    return f"{VALUE_TO_NOTATION[high]}{VALUE_TO_NOTATION[low]}{suffix}"


def estimate_project_equity(
    *,
    hero_cards: Sequence[Any],
    board_cards: Sequence[Any],
    opponent_profiles: Sequence[dict[str, Any] | OpponentProfile],
    opponent_count: Any,
    simulations: int = 600,
) -> Optional[float]:
    """Recompute exact project equity if project dependencies are importable.

    Returns ``None`` when the function is copied into a standalone training
    project without ``pokereval`` or ``objet.services.equity`` available.
    """

    try:
        from pokereval.card import Card as PokerEvalCard
        from objet.services.equity import (
            OpponentProfile as ProjectOpponentProfile,
            profiles_for_opponent_count as project_profiles_for_opponent_count,
            weighted_monte_carlo_equity,
        )
    except ImportError:
        return None

    parsed_hero = [card for card in (parse_card(card_text) for card_text in hero_cards) if card is not None]
    parsed_board = [card for card in (parse_card(card_text) for card_text in board_cards) if card is not None]
    if len(parsed_hero) != 2:
        return None

    project_hero = [PokerEvalCard(card.rank, card.suit) for card in parsed_hero]
    project_board = [PokerEvalCard(card.rank, card.suit) for card in parsed_board]
    project_profiles = [
        ProjectOpponentProfile(
            name=profile.name,
            looseness=profile.looseness,
            aggression=profile.aggression,
            action=profile.action,
            confidence=profile.confidence,
        )
        for profile in (_profile_from_any(item) for item in opponent_profiles)
    ]
    count = int(number_or_none(opponent_count) or len(project_profiles))
    project_profiles = project_profiles_for_opponent_count(project_profiles, count)
    return weighted_monte_carlo_equity(
        hero_cards=project_hero,
        board_cards=project_board,
        opponent_profiles=project_profiles,
        simulations=simulations,
    )


def parse_card(value: Any) -> Optional[Card]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "??"}:
        return None

    normalized = text.upper().replace("10", "T")
    rank_match = re.search(r"(A|K|Q|J|T|[2-9])", normalized)
    if not rank_match:
        return None
    rank_token = rank_match.group(1)
    rank = RANK_TO_VALUE.get("10" if rank_token == "T" else rank_token)
    if rank is None:
        return None

    suit = None
    lowered = text.lower()
    for token, value_number in MOJIBAKE_SUIT_TO_VALUE.items():
        if token in lowered:
            suit = value_number
            break
    if suit is None:
        for token, value_number in SUIT_TO_VALUE.items():
            if token in lowered:
                suit = value_number
                break
    if suit is None:
        # Last chance for compact ASCII forms like As, Td, 7c.
        suffix = lowered[-1:]
        suit = SUIT_TO_VALUE.get(suffix)
    if suit is None:
        return None
    return Card(rank=rank, suit=suit)


def number_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clamp01(value: Any) -> float:
    number = number_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, number))


def _features_object(row: dict[str, Any]) -> dict[str, Any]:
    if "features" in row and isinstance(row["features"], dict):
        return row["features"]
    return row


def _profile_from_any(profile: OpponentProfile | dict[str, Any]) -> OpponentProfile:
    if isinstance(profile, OpponentProfile):
        return profile
    data = dict_or_empty(profile)
    return OpponentProfile(
        name=str(data.get("name") or "opponent"),
        looseness=clamp01(data.get("looseness", 0.55)),
        aggression=clamp01(data.get("aggression", 0.35)),
        action=str(data.get("action") or "play").lower(),
        confidence=number_or_none(data.get("confidence")) or 0.0,
    )


__all__ = [
    "Card",
    "OpponentProfile",
    "active_button_states",
    "button_state_from_text",
    "call_max",
    "equity_required",
    "estimate_project_equity",
    "expected_value",
    "has_button_state",
    "is_trainable_row",
    "opponent_profiles_from_players",
    "parse_card",
    "preflop_notation",
    "profiles_for_opponent_count",
    "rebuild_derived_features",
    "starting_hand_strength",
    "starting_hand_strength_from_text",
    "to_call_pot_ratio",
]
