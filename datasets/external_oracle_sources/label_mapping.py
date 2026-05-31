"""Shared poker action normalization for external oracle datasets."""

from __future__ import annotations

import html
import re
from typing import Any


FOUR_CLASS_LABELS = ("CHECK", "FOLD", "CALL", "RAISE")
INTENT_LABELS = ("NO_INVEST", "CALL", "RAISE")
INTENT_MAPPING = {
    "CHECK": "NO_INVEST",
    "FOLD": "NO_INVEST",
    "CALL": "CALL",
    "RAISE": "RAISE",
}


def clean_action_text(value: Any) -> str:
    text = html.unescape(str(value or "")).strip()
    if not text:
        return ""
    action_match = re.search(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        text = action_match.group(1)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_action_4class(value: Any) -> str | None:
    text = clean_action_text(value)
    lowered = text.lower()
    if not lowered:
        return None
    if re.search(r"\bcheck(?:s|ed|ing)?\b", lowered) or lowered in {"k", "x"}:
        return "CHECK"
    if re.search(r"\bfold(?:s|ed|ing)?\b", lowered) or lowered == "f":
        return "FOLD"
    if re.search(r"\bcall(?:s|ed|ing)?\b", lowered) or lowered == "c":
        return "CALL"
    if re.search(r"\b(?:bet|bets|bett?ing|raise|raises|raised|all\s*in|allin|jam|jams|shove|shoves)\b", lowered):
        return "RAISE"
    if re.fullmatch(r"[br]\s*\d+(?:\.\d+)?", lowered):
        return "RAISE"
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*(?:bb|chips?))?", lowered):
        return "RAISE"
    return None


def normalize_action_3intent(value: Any) -> str | None:
    four_class = normalize_action_4class(value)
    if four_class is None:
        return None
    return INTENT_MAPPING[four_class]


def extract_bet_size(value: Any) -> float | None:
    text = clean_action_text(value).lower()
    if not text:
        return None
    match = re.search(r"(?:bet|raise|raises\s+to|raised\s+to|all\s*in|allin|jam|shove|[br])\s*(?:to\s*)?(\d+(?:\.\d+)?)", text)
    if not match:
        match = re.fullmatch(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = match.group(1) if match.lastindex else match.group(0)
    try:
        return float(number)
    except ValueError:
        return None

