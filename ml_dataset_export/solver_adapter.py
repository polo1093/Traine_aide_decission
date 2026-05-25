"""Solver adapter abstractions for solver-labeled poker ML datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Protocol


TRAINING_ACTIONS = {"FOLD", "CHECK", "CALL", "RAISE"}
RAISE_BUTTON_STATES = {"relance", "raise", "mise", "bet", "all-in", "all in"}
CALL_BUTTON_STATES = {"paie", "call"}
CHECK_BUTTON_STATES = {"check"}
FOLD_BUTTON_STATES = {"fold", "couche"}


@dataclass(frozen=True)
class SolverSpot:
    hero_cards: list[str]
    board_cards: list[str]
    street: str
    hero_position: str | None = None
    player_active: int | None = None
    player_start: int | None = None
    pot: float | None = None
    to_call: float | None = None
    buttons: list[dict[str, Any]] | None = None
    buttons_active: list[str] | None = None
    opponent_profiles: list[dict[str, Any]] | dict[str, Any] | None = None
    players: list[dict[str, Any]] | None = None
    equity_table: float | None = None
    equity_1v1: float | None = None
    source_snapshot_id: str | None = None


@dataclass(frozen=True)
class SolverDecision:
    action: str
    raise_amount: float | None = None
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class SolverAdapter(Protocol):
    name: str
    version: str

    def solve_spot(self, spot: SolverSpot) -> SolverDecision:
        """Return the solver decision for one observable spot."""


class MockSolverAdapter:
    """Deterministic local solver used until a real external solver is bound."""

    name = "mock"
    version = "mock_solver_v1"

    def solve_spot(self, spot: SolverSpot) -> SolverDecision:
        equity = spot.equity_table if spot.equity_table is not None else spot.equity_1v1
        equity = 0.5 if equity is None else float(equity)
        pot = float(spot.pot or 0.0)
        to_call = float(spot.to_call or 0.0)
        required = to_call / (pot + to_call) if pot + to_call > 0 else 0.0

        if to_call <= 0:
            if equity >= 0.62 and action_allowed_by_buttons("RAISE", spot.buttons_active):
                return SolverDecision("BET", max(1.0, round(pot * 0.5, 2)), 0.72, {"rule": "value_bet"})
            return SolverDecision("CHECK", None, 0.68, {"rule": "free_check"})

        if equity >= required + 0.24 and action_allowed_by_buttons("RAISE", spot.buttons_active):
            return SolverDecision("RAISE", max(to_call * 2.0, round(pot * 0.6, 2)), 0.74, {"rule": "raise_edge"})
        if equity >= required and action_allowed_by_buttons("CALL", spot.buttons_active):
            return SolverDecision("CALL", None, 0.67, {"rule": "call_edge"})
        return SolverDecision("FOLD", None, 0.7, {"rule": "below_required_equity"})


class ImportSolverAdapter:
    """Small binder for a future external solver module.

    The imported module must expose ``solve_spot(spot)`` and return either a
    ``SolverDecision`` or a dict with ``action``, ``raise_amount``,
    ``confidence`` and ``raw`` keys.
    """

    def __init__(self, module_name: str, *, name: str | None = None, version: str | None = None) -> None:
        self.module_name = module_name
        self.module = importlib.import_module(module_name)
        self.name = name or getattr(self.module, "SOLVER_NAME", module_name)
        self.version = version or getattr(self.module, "SOLVER_VERSION", "external_solver")

    def solve_spot(self, spot: SolverSpot) -> SolverDecision:
        result = self.module.solve_spot(spot)
        if isinstance(result, SolverDecision):
            return result
        if not isinstance(result, dict):
            raise TypeError("external solver must return SolverDecision or dict")
        return SolverDecision(
            action=str(result.get("action") or ""),
            raise_amount=_float_or_none(result.get("raise_amount")),
            confidence=_float_or_none(result.get("confidence")),
            raw=dict(result.get("raw") or result),
        )


def normalize_solver_action(action: Any) -> str | None:
    normalized = str(action or "").strip().upper()
    if normalized == "BET":
        return "RAISE"
    if normalized in TRAINING_ACTIONS:
        return normalized
    return None


def action_allowed_by_buttons(action: str, buttons_active: list[str] | None) -> bool:
    states = {str(state).lower() for state in (buttons_active or [])}
    if action == "FOLD":
        return bool(states & FOLD_BUTTON_STATES)
    if action == "CHECK":
        return bool(states & CHECK_BUTTON_STATES)
    if action == "CALL":
        return bool(states & CALL_BUTTON_STATES)
    if action == "RAISE":
        return bool(states & RAISE_BUTTON_STATES)
    return False


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

