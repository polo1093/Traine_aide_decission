"""Build solver-ready ML snapshots from parsed PokerTH hand summaries."""

from __future__ import annotations

from typing import Any


BOARD_COUNTS = {"FLOP": 3, "TURN": 4, "RIVER": 5}


def build_snapshot_from_hand_summary(
    hand_summary: dict[str, Any],
    *,
    street: str,
    to_call: float | None,
) -> dict[str, Any]:
    """Build one solver-ready snapshot if the parsed hand is simple enough."""

    try:
        if not isinstance(hand_summary, dict):
            return _result(None, "invalid_hand_summary", "invalid_hand_summary")

        game_id = hand_summary.get("game_id")
        hand_id = hand_summary.get("hand_id")
        normalized_street = str(street).upper()
        if normalized_street not in BOARD_COUNTS:
            return _result(None, "unsupported_street", f"unsupported_street:{street}", game_id, hand_id)
        if hand_summary.get("has_side_pot"):
            return _result(None, "side_pot_not_supported", "side_pot_not_supported", game_id, hand_id)
        if not hand_summary.get("hero_hand"):
            return _result(None, "hero_hand_missing", "hero_hand_missing", game_id, hand_id)
        if not hand_summary.get("villain_hand"):
            return _result(None, "villain_hand_missing", "villain_hand_missing", game_id, hand_id)
        if to_call is None:
            return _result(None, "to_call_unknown", "to_call_unknown", game_id, hand_id)

        board_cards = _board_for_street(hand_summary, normalized_street)
        if len(board_cards) != BOARD_COUNTS[normalized_street]:
            return _result(None, "invalid_board", "invalid_board", game_id, hand_id)

        active_players = hand_summary.get("active_players_by_street", {}).get(normalized_street)
        if not active_players:
            return _result(None, "street_activity_unknown", "street_activity_unknown", game_id, hand_id)
        if len(active_players) != 2:
            return _result(None, "multiway_context_not_supported", "multiway_context_not_supported", game_id, hand_id)
        expected_players = {hand_summary.get("hero_name"), hand_summary.get("villain_name")}
        if set(active_players) != expected_players:
            return _result(None, "multiway_context_not_supported", "multiway_context_not_supported", game_id, hand_id)

        pot = hand_summary.get("pot_final")
        if pot is None or float(pot) <= 0:
            return _result(None, "pot_reconstruction_failed", "pot_reconstruction_failed", game_id, hand_id)

        snapshot = {
            "schema_version": "ml_dataset_v1",
            "snapshot_id": f"pokerth_game{game_id}_hand{hand_id}_{normalized_street.lower()}",
            "metadata": {
                "source_type": "pokerth_history",
                "source_reliability": "reconstructed_history",
                "game_id": game_id,
                "hand_id": hand_id,
                "street": normalized_street,
            },
            "features": {
                "hero_cards": hand_summary["hero_hand"],
                "villain_hand": hand_summary["villain_hand"],
                "board_cards": board_cards,
                "pot": float(pot),
                "to_call": float(to_call),
                "to_call_is_estimated": False,
                "decision_context_known": True,
                "active_opponents": 1,
                "units": "chips",
                "pot_is_estimated": bool(hand_summary.get("pot_is_estimated", True)),
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
            "confidence": {
                "overall": 0.8,
            },
        }
        return _result(snapshot, None, None, game_id, hand_id)
    except Exception as exc:  # noqa: BLE001
        return _result(None, "snapshot_builder_exception", _format_error(exc))


def _board_for_street(hand_summary: dict[str, Any], street: str) -> list[str]:
    board = hand_summary.get("board") or {}
    if street == "FLOP":
        return list(board.get("flop") or [])
    if street == "TURN":
        return list((board.get("flop") or []) + (board.get("turn") or []))
    if street == "RIVER":
        return list((board.get("flop") or []) + (board.get("turn") or []) + (board.get("river") or []))
    return []


def _result(
    snapshot: dict[str, Any] | None,
    rejection_reason: str | None,
    error: str | None,
    game_id: int | None = None,
    hand_id: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed" if error else "ok",
        "game_id": game_id,
        "hand_id": hand_id,
        "snapshot": snapshot,
        "rejection_reason": rejection_reason,
        "error": error,
        "warnings": [],
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
