"""Strict validator for solver-generated ml_decision_snapshot JSONL files."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

try:
    from . import feature_rebuilder
    from .solver_adapter import TRAINING_ACTIONS, action_allowed_by_buttons
except ImportError:  # pragma: no cover
    import feature_rebuilder
    from solver_adapter import TRAINING_ACTIONS, action_allowed_by_buttons


PROHIBITED_FEATURE_KEYS = {
    "action_history",
    "debug",
    "final_result",
    "hand_result",
    "hidden_cards",
    "labels",
    "opponent_cards",
    "raw_solver",
    "reads",
    "solver_action",
    "solver_ev",
    "solver_raw",
    "villain_cards",
}
STREET_BOARD_COUNT = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    lines_checked: int = 0


class ExampleSchema:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.keys_by_path: dict[str, list[str]] = {}
        self.types_by_path: dict[str, set[type]] = {}
        for row in rows:
            self._visit("", row)

    @classmethod
    def from_path(cls, path: str | Path) -> "ExampleSchema":
        rows = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError("example schema file is empty")
        return cls(rows)

    def _visit(self, path: str, value: Any) -> None:
        self.types_by_path.setdefault(path, set()).add(type(value))
        if isinstance(value, dict):
            keys = self.keys_by_path.setdefault(path, [])
            for key in value:
                if key not in keys:
                    keys.append(key)
                self._visit(_join(path, key), value[key])
        elif isinstance(value, list):
            for item in value:
                self._visit(path + "[]", item)


def validate_jsonl(input_path: str | Path, *, example_path: str | Path = "example_training_dataset.jsonl") -> ValidationResult:
    schema = ExampleSchema.from_path(example_path)
    errors: list[str] = []
    lines_checked = 0
    with Path(input_path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            lines_checked += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid_json:{exc}")
                continue
            _validate_schema(row, schema, errors, line_number)
            _validate_semantics(row, errors, line_number)
    if lines_checked == 0:
        errors.append("file is empty")
    return ValidationResult(ok=not errors, errors=errors, lines_checked=lines_checked)


def _validate_schema(value: Any, schema: ExampleSchema, errors: list[str], line_number: int, path: str = "") -> None:
    allowed_types = schema.types_by_path.get(path)
    if allowed_types and not _type_allowed(value, allowed_types):
        errors.append(f"line {line_number}: {path or '<root>'} bad_type:{type(value).__name__}")

    if isinstance(value, dict):
        expected_keys = schema.keys_by_path.get(path)
        if expected_keys is not None:
            actual_keys = list(value)
            missing = [key for key in expected_keys if key not in value]
            unknown = [key for key in value if key not in expected_keys]
            if missing:
                errors.append(f"line {line_number}: {path or '<root>'} missing:{missing}")
            if unknown:
                errors.append(f"line {line_number}: {path or '<root>'} unknown:{unknown}")
            if not missing and not unknown and actual_keys != expected_keys:
                errors.append(f"line {line_number}: {path or '<root>'} key_order_mismatch")
        for key, item in value.items():
            _validate_schema(item, schema, errors, line_number, _join(path, key))
    elif isinstance(value, list):
        for item in value:
            _validate_schema(item, schema, errors, line_number, path + "[]")


def _validate_semantics(row: dict[str, Any], errors: list[str], line_number: int) -> None:
    if row.get("type") != "ml_decision_snapshot":
        errors.append(f"line {line_number}: invalid_type")
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    flags = row.get("quality_flags") if isinstance(row.get("quality_flags"), dict) else {}

    prohibited = sorted(PROHIBITED_FEATURE_KEYS & set(features))
    if prohibited:
        errors.append(f"line {line_number}: prohibited_features:{prohibited}")

    actions = [labels.get("legacy_action"), labels.get("final_action")]
    for action in actions:
        if action == "WAIT":
            errors.append(f"line {line_number}: wait_action")
        if action not in TRAINING_ACTIONS:
            errors.append(f"line {line_number}: invalid_action:{action}")
    if labels.get("legacy_action") != labels.get("final_action"):
        errors.append(f"line {line_number}: action_label_mismatch")

    if labels.get("label_valid") is not True or labels.get("known_bug_risk") is not False or flags.get("usable_for_training") is not True:
        errors.append(f"line {line_number}: not_trainable")

    hero_cards = features.get("hero_cards") if isinstance(features.get("hero_cards"), list) else []
    board_cards = features.get("board_cards") if isinstance(features.get("board_cards"), list) else []
    parsed = [feature_rebuilder.parse_card(card) for card in [*hero_cards, *board_cards]]
    if len(hero_cards) != 2 or any(card is None for card in parsed[:2]):
        errors.append(f"line {line_number}: invalid_hero_cards")
    if any(card is None for card in parsed[2:]):
        errors.append(f"line {line_number}: invalid_board_cards")
    card_keys = [(card.rank, card.suit) for card in parsed if card is not None]
    if len(card_keys) != len(set(card_keys)):
        errors.append(f"line {line_number}: duplicate_cards")

    street = str(metadata.get("street") or "").upper()
    expected_board_count = STREET_BOARD_COUNT.get(street)
    if expected_board_count is None:
        errors.append(f"line {line_number}: invalid_street:{street}")
    elif len(board_cards) != expected_board_count:
        errors.append(f"line {line_number}: board_street_mismatch:{street}")

    action = labels.get("legacy_action")
    buttons_active = features.get("buttons_active") if isinstance(features.get("buttons_active"), list) else []
    if isinstance(action, str) and action in TRAINING_ACTIONS and not action_allowed_by_buttons(action, buttons_active):
        errors.append(f"line {line_number}: action_impossible:{action}")


def _type_allowed(value: Any, allowed_types: set[type]) -> bool:
    if type(value) in allowed_types:
        return True
    if type(value) is int and float in allowed_types:
        return True
    return False


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key

