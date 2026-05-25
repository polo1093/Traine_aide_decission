"""Schema and validation helpers for bounded PokerSolver jobs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "solver_job_v1"
MAX_ITERATIONS = 100
MAX_TIMEOUT_S = 10.0
ALLOWED_STREETS = {"FLOP": 3, "TURN": 4, "RIVER": 5}
ALLOWED_BACKENDS = {"rust", "python"}
ALLOWED_LABEL_INTENTS = {"solver_smoke", "solver_candidate"}
ALLOWED_SOURCE_TYPES = {"manual_fixture", "ml_snapshot", "pokerth_history", "synthetic"}
ALLOWED_UNITS = {"chips", "bb"}
CARD_RE = re.compile(r"^(?:[2-9TJQKA][hdcs])$", re.IGNORECASE)


def validate_solver_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable validation result for a solver job dict."""

    try:
        normalized = _normalize_solver_job(job)
        return {"status": "ok", "job": normalized, "error": None}
    except Exception as exc:  # noqa: BLE001 - public boundary is stable
        return {"status": "failed", "job": None, "error": _format_error(exc)}


def _normalize_solver_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(job, Mapping):
        raise TypeError("solver job must be a mapping")

    data = deepcopy(dict(job))
    required = (
        "solver_job_id",
        "source_snapshot_id",
        "created_at",
        "schema_version",
        "source_type",
        "units",
        "street",
        "hero_hand",
        "villain_hand",
        "villain_range",
        "board",
        "pot",
        "to_call",
        "stack",
        "bet_sizes",
        "iterations",
        "timeout_s",
        "backend",
        "label_intent",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing_job_fields:{','.join(missing)}")

    _require_nonempty_text(data, "solver_job_id")
    _require_nonempty_text(data, "source_snapshot_id")
    _require_nonempty_text(data, "created_at")

    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported_schema_version:{data['schema_version']}")
    if data["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"unsupported_source_type:{data['source_type']}")
    if data["units"] not in ALLOWED_UNITS:
        raise ValueError(f"unsupported_units:{data['units']}")

    street = str(data["street"]).upper()
    if street not in ALLOWED_STREETS:
        raise ValueError(f"unsupported_street:{data['street']}")
    data["street"] = street

    hero_hand = _normalize_card_list(data["hero_hand"], "hero_hand", expected_count=2)
    board = _normalize_card_list(data["board"], "board", expected_count=ALLOWED_STREETS[street])
    villain_range = data["villain_range"]
    if villain_range not in (None, ""):
        raise ValueError("villain_range_not_supported")
    data["villain_range"] = None

    if data["villain_hand"] is None:
        raise ValueError("villain_hand_required")
    villain_hand = _normalize_card_list(data["villain_hand"], "villain_hand", expected_count=2)

    all_cards = hero_hand + villain_hand + board
    duplicates = sorted({card for card in all_cards if all_cards.count(card) > 1})
    if duplicates:
        raise ValueError(f"duplicate_cards:{','.join(duplicates)}")

    pot = _positive_float(data["pot"], "pot")
    to_call = _nonnegative_float(data["to_call"], "to_call")
    stack = _positive_float(data["stack"], "stack")
    iterations = _bounded_positive_int(data["iterations"], "iterations", MAX_ITERATIONS)
    timeout_s = _bounded_positive_float(data["timeout_s"], "timeout_s", MAX_TIMEOUT_S)
    bet_sizes = _normalize_bet_sizes(data["bet_sizes"])

    backend = str(data["backend"]).lower()
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"unsupported_backend:{data['backend']}")
    label_intent = str(data["label_intent"])
    if label_intent not in ALLOWED_LABEL_INTENTS:
        raise ValueError(f"unsupported_label_intent:{data['label_intent']}")

    data.update(
        {
            "hero_hand": hero_hand,
            "villain_hand": villain_hand,
            "board": board,
            "pot": pot,
            "to_call": to_call,
            "stack": stack,
            "bet_sizes": bet_sizes,
            "iterations": iterations,
            "timeout_s": timeout_s,
            "backend": backend,
            "label_intent": label_intent,
        }
    )
    return data


def _require_nonempty_text(data: Mapping[str, Any], key: str) -> None:
    if not isinstance(data[key], str) or not data[key].strip():
        raise ValueError(f"{key}_required")


def _normalize_card_list(value: Any, field_name: str, *, expected_count: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name}_must_be_list")
    cards = [_normalize_card(card, field_name) for card in value]
    if len(cards) != expected_count:
        raise ValueError(f"{field_name}_card_count:{len(cards)}_expected:{expected_count}")
    return cards


def _normalize_card(value: Any, field_name: str) -> str:
    card = str(value).strip()
    card = card.replace("10", "T")
    if not CARD_RE.match(card):
        raise ValueError(f"invalid_card:{field_name}:{value}")
    return card[0].upper() + card[1].lower()


def _positive_float(value: Any, field_name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name}_must_be_positive")
    return number


def _nonnegative_float(value: Any, field_name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{field_name}_must_be_nonnegative")
    return number


def _bounded_positive_int(value: Any, field_name: str, max_value: int) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name}_must_be_positive")
    if number > max_value:
        raise ValueError(f"{field_name}_exceeds_limit:{max_value}")
    return number


def _bounded_positive_float(value: Any, field_name: str, max_value: float) -> float:
    if value is None:
        raise ValueError(f"{field_name}_required")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name}_must_be_positive")
    if number > max_value:
        raise ValueError(f"{field_name}_exceeds_limit:{max_value:g}")
    return number


def _normalize_bet_sizes(value: Any) -> list[float]:
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        raw_parts = [value]
    bet_sizes = [float(part) for part in raw_parts if str(part).strip()]
    if not bet_sizes:
        raise ValueError("bet_sizes_required")
    if len(bet_sizes) > 5:
        raise ValueError("bet_sizes_exceeds_limit:5")
    if any(size <= 0 for size in bet_sizes):
        raise ValueError("bet_sizes_must_be_positive")
    return bet_sizes


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
