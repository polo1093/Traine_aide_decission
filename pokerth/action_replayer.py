"""Replay parsed PokerTH actions to reconstruct hero decision contexts."""

from __future__ import annotations

from typing import Any


STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")
DECISION_ACTIONS = {"calls", "bets", "raises", "checks", "folds"}
POT_ACTIONS = {"posts small blind", "posts big blind", "calls", "bets", "raises"}


def replay_hero_decisions(hand_summary: dict[str, Any], *, hero_name: str = "polo") -> dict[str, Any]:
    """Replay one parsed hand and return hero decision contexts plus snapshots."""

    try:
        validation_error = _validate_hand_summary(hand_summary, hero_name)
        if validation_error is not None:
            return _result("failed", [], [], validation_error, validation_error)

        players = list(hand_summary["players"])
        active_players = set(players)
        contributions = {player: 0.0 for player in players}
        pot = 0.0
        contexts: list[dict[str, Any]] = []

        for street in STREETS:
            if street != "PREFLOP":
                contributions = {player: 0.0 for player in players}
            for action in hand_summary.get("actions_by_street", {}).get(street, []):
                player = action.get("player")
                action_name = str(action.get("action", "")).lower()
                if action_name in {"shows", "wins", "collected"}:
                    continue
                if action_name not in POT_ACTIONS and action_name not in {"checks", "folds"}:
                    return _result("failed", contexts, [], "unknown_action", f"unknown_action:{action_name}")
                if player not in contributions:
                    return _result("failed", contexts, [], "multiway_context_not_supported", "multiway_context_not_supported")

                if player == hero_name and action_name in DECISION_ACTIONS:
                    contexts.append(_decision_context(hand_summary, street, action, pot, contributions, active_players, hero_name))

                apply_error, chips_added = _apply_action(action, contributions, active_players)
                if apply_error is not None:
                    return _result("failed", contexts, [], apply_error, apply_error)
                pot += chips_added

        snapshots = [_snapshot_from_context(hand_summary, ctx, index) for index, ctx in enumerate(contexts) if ctx["street"] != "PREFLOP"]
        return _result("ok", contexts, snapshots, None, None, pot)
    except Exception as exc:  # noqa: BLE001
        return _result("failed", [], [], "replay_exception", _format_error(exc))


def _validate_hand_summary(hand_summary: dict[str, Any], hero_name: str) -> str | None:
    if not isinstance(hand_summary, dict):
        return "invalid_hand_summary"
    if hand_summary.get("has_side_pot"):
        return "side_pot_not_supported"
    players = hand_summary.get("players") or []
    if len(players) != 2:
        return "multiway_context_not_supported"
    if not hand_summary.get("hero_hand"):
        return "hero_hand_missing"
    if not hand_summary.get("villain_hand"):
        return "villain_hand_missing"
    if hero_name not in players:
        return "hero_hand_missing"
    return None


def _decision_context(
    hand_summary: dict[str, Any],
    street: str,
    action: dict[str, Any],
    pot: float,
    contributions: dict[str, float],
    active_players: set[str],
    hero_name: str,
) -> dict[str, Any]:
    to_call = _to_call(hero_name, contributions)
    hero_action = _normalize_action(action["action"])
    return {
        "game_id": hand_summary.get("game_id"),
        "hand_id": hand_summary.get("hand_id"),
        "hero_name": hero_name,
        "street": street,
        "hero_action": hero_action,
        "pot_before_action": float(pot),
        "to_call": float(to_call),
        "can_check": to_call == 0,
        "can_call": to_call > 0,
        "can_raise": True,
        "active_opponents": max(0, len(active_players) - 1),
        "board_cards": _board_for_street(hand_summary, street),
        "hero_cards": list(hand_summary.get("hero_hand") or []),
        "villain_hand": list(hand_summary.get("villain_hand") or []),
        "decision_context_known": True,
        "raw_action": action.get("raw"),
    }


def _apply_action(
    action: dict[str, Any],
    contributions: dict[str, float],
    active_players: set[str],
) -> tuple[str | None, float]:
    player = action["player"]
    action_name = str(action["action"]).lower()
    amount = action.get("amount")
    to_call = _to_call(player, contributions)

    if action_name in {"posts small blind", "posts big blind", "calls", "bets", "raises"} and amount is None:
        return "amount_parse_failed", 0.0
    if action_name == "checks":
        if to_call != 0:
            return "pot_reconstruction_failed", 0.0
        return None, 0.0
    if action_name == "folds":
        active_players.discard(player)
        return None, 0.0
    if action_name in {"posts small blind", "posts big blind"}:
        contributions[player] += float(amount)
        return None, float(amount)
    if action_name == "calls":
        if float(amount) != to_call:
            return ("all_in_complex_not_supported" if float(amount) < to_call else "pot_reconstruction_failed"), 0.0
        contributions[player] += float(amount)
        return None, float(amount)
    if action_name == "bets":
        if to_call == 0:
            contributions[player] += float(amount)
            return None, float(amount)
        target = float(amount)
        previous = contributions[player]
        if target <= max(contributions.values()):
            return "pot_reconstruction_failed", 0.0
        contributions[player] = target
        return None, target - previous
    if action_name == "raises":
        target = float(amount)
        previous = contributions[player]
        if target <= max(contributions.values()):
            return "pot_reconstruction_failed", 0.0
        contributions[player] = target
        return None, target - previous
    return "unknown_action", 0.0


def _to_call(player: str, contributions: dict[str, float]) -> float:
    return max(contributions.values(), default=0.0) - contributions.get(player, 0.0)


def _normalize_action(action: str) -> str:
    mapping = {"calls": "CALL", "bets": "BET", "raises": "RAISE", "checks": "CHECK", "folds": "FOLD"}
    return mapping.get(str(action).lower(), str(action).upper())


def _board_for_street(hand_summary: dict[str, Any], street: str) -> list[str]:
    board = hand_summary.get("board") or {}
    if street == "PREFLOP":
        return []
    if street == "FLOP":
        return list(board.get("flop") or [])
    if street == "TURN":
        return list((board.get("flop") or []) + (board.get("turn") or []))
    if street == "RIVER":
        return list((board.get("flop") or []) + (board.get("turn") or []) + (board.get("river") or []))
    return []


def _snapshot_from_context(hand_summary: dict[str, Any], context: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "schema_version": "ml_dataset_v1",
        "snapshot_id": f"pokerth_game{context['game_id']}_hand{context['hand_id']}_{context['street'].lower()}_decision{index}",
        "metadata": {
            "source_type": "pokerth_history",
            "source_reliability": "reconstructed_history",
            "game_id": context["game_id"],
            "hand_id": context["hand_id"],
            "street": context["street"],
            "observed_hero_action": context["hero_action"],
        },
        "features": {
            "hero_cards": context["hero_cards"],
            "villain_hand": context["villain_hand"],
            "board_cards": context["board_cards"],
            "pot": context["pot_before_action"],
            "to_call": context["to_call"],
            "to_call_is_estimated": False,
            "decision_context_known": True,
            "active_opponents": context["active_opponents"],
            "units": "chips",
            "pot_is_estimated": True,
            "pot_reconstruction_method": hand_summary.get("pot_reconstruction_method"),
        },
        "quality_flags": {
            "usable_for_training": False,
            "usable_for_solver": True,
        },
        "labels": {
            "label_source": "pokerth_history",
            "label_quality": "history_reconstructed",
            "training_label": None,
        },
        "confidence": {"overall": 0.8},
    }


def _result(
    status: str,
    decision_contexts: list[dict[str, Any]],
    decision_snapshots: list[dict[str, Any]],
    rejection_reason: str | None,
    error: str | None,
    pot_reconstructed: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision_contexts": decision_contexts,
        "decision_snapshots": decision_snapshots,
        "pot_reconstructed": pot_reconstructed,
        "rejection_reason": rejection_reason,
        "error": error,
        "warnings": [],
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
