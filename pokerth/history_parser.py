"""Conservative PokerTH text hand-history parser."""

from __future__ import annotations

import re
from typing import Any


HEADER_RE = re.compile(r"##\s*Game:\s*(?P<game>\d+)\s*\|\s*Hand:\s*(?P<hand>\d+)\s*##", re.IGNORECASE)
ACTION_RE = re.compile(
    r"^(?P<player>.+?)\s+"
    r"(?P<action>posts small blind|posts big blind|bets|calls|raises|checks|folds|wins|collected)"
    r"(?:\s+\(?\$?(?P<amount>[\d.,]+)\)?)?",
    re.IGNORECASE,
)
SHOW_RE = re.compile(r"^(?P<player>.+?)\s+shows\s+\[(?P<cards>[^\]]+)\]", re.IGNORECASE)
WIN_RE = re.compile(r"^(?P<player>.+?)\s+(?:wins|collected)\s+\$?(?P<amount>[\d.,]+)", re.IGNORECASE)
BOARD_RE = re.compile(r"\b(?P<street>flop|turn|river)\b.*?\[(?P<cards>[^\]]+)\]", re.IGNORECASE)
BOARD_ALL_RE = re.compile(r"\bboard\b.*?\[(?P<cards>[^\]]+)\]", re.IGNORECASE)

STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")
POT_ACTIONS = {"posts small blind", "posts big blind", "bets", "calls", "raises"}


def parse_pokerth_history(text: str, *, hero_name: str = "polo") -> dict[str, Any]:
    """Parse one text blob containing one or more PokerTH hands."""

    try:
        sections = _split_hands(text)
        if not sections:
            return _history_result("failed", [], [], "no_hands_found")
        hands: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for section in sections:
            parsed = parse_pokerth_hand(section, hero_name=hero_name)
            if parsed["status"] == "ok":
                hands.append(parsed["hand_summary"])
            else:
                rejections.append(
                    {
                        "game_id": parsed.get("game_id"),
                        "hand_id": parsed.get("hand_id"),
                        "rejection_reason": parsed.get("rejection_reason"),
                        "error": parsed.get("error"),
                    }
                )
        if hands and rejections:
            status = "partial"
        elif hands:
            status = "ok"
        else:
            status = "failed"
        return _history_result(status, hands, rejections, None if hands else "all_hands_rejected")
    except Exception as exc:  # noqa: BLE001
        return _history_result("failed", [], [], _format_error(exc))


def parse_pokerth_hand(text: str, *, hero_name: str = "polo") -> dict[str, Any]:
    """Parse one PokerTH hand and return a stable result."""

    game_id = None
    hand_id = None
    try:
        header = HEADER_RE.search(text or "")
        if header is None:
            return _hand_result("failed", None, "invalid_hand_header", "invalid_hand_header")
        game_id = int(header.group("game"))
        hand_id = int(header.group("hand"))

        if re.search(r"\bside\s+pot\b", text, re.IGNORECASE):
            return _hand_result("failed", None, "side_pot_not_supported", "side_pot_not_supported", game_id, hand_id)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        current_street = "PREFLOP"
        board_by_street: dict[str, list[str]] = {"FLOP": [], "TURN": [], "RIVER": []}
        actions_by_street: dict[str, list[dict[str, Any]]] = {street: [] for street in STREETS}
        active_players: set[str] = set()
        active_by_street: dict[str, list[str]] = {}
        players_seen: set[str] = set()
        shown_hands: dict[str, list[str]] = {}
        blinds: dict[str, dict[str, Any]] = {}
        winners: list[dict[str, Any]] = []
        pot_total = 0.0
        pot_sources: list[dict[str, Any]] = []

        for line in lines:
            if HEADER_RE.search(line):
                continue

            board_event = _parse_board_line(line)
            if board_event is not None:
                current_street = board_event["street"]
                _merge_board(board_by_street, board_event["street"], board_event["cards"])
                active_by_street.setdefault(current_street, sorted(active_players))
                continue

            show = SHOW_RE.match(line)
            if show:
                player = show.group("player").strip()
                cards = _parse_cards(show.group("cards"))
                players_seen.add(player)
                active_players.add(player)
                shown_hands[player] = cards
                actions_by_street[current_street].append({"player": player, "action": "shows", "cards": cards, "raw": line})
                continue

            winner = WIN_RE.match(line)
            if winner:
                player = winner.group("player").strip()
                amount = _parse_amount(winner.group("amount"))
                winners.append({"player": player, "amount": amount})

            action = ACTION_RE.match(line)
            if action:
                player = action.group("player").strip()
                action_name = action.group("action").lower()
                amount = _parse_amount(action.group("amount"))
                players_seen.add(player)
                if action_name != "folds":
                    active_players.add(player)
                if action_name == "folds" and player in active_players:
                    active_players.remove(player)
                if action_name in POT_ACTIONS and amount is not None:
                    pot_total += amount
                    pot_sources.append({"player": player, "action": action_name, "amount": amount, "street": current_street})
                if action_name == "posts small blind":
                    blinds["small_blind"] = {"player": player, "amount": amount}
                if action_name == "posts big blind":
                    blinds["big_blind"] = {"player": player, "amount": amount}
                actions_by_street[current_street].append(
                    {"player": player, "action": action_name, "amount": amount, "raw": line}
                )

        if len(shown_hands) < 2:
            return _hand_result("failed", None, "showdown_missing", "showdown_missing", game_id, hand_id)
        hero_hand = shown_hands.get(hero_name)
        if hero_hand is None:
            return _hand_result("failed", None, "hero_hand_missing", "hero_hand_missing", game_id, hand_id)
        villain_entries = [(player, cards) for player, cards in shown_hands.items() if player != hero_name]
        if len(villain_entries) != 1:
            return _hand_result("failed", None, "multiway_not_supported", "multiway_not_supported", game_id, hand_id)

        villain_name, villain_hand = villain_entries[0]
        for street in ("FLOP", "TURN", "RIVER"):
            active_by_street.setdefault(street, sorted(active_players))

        hand_summary = {
            "game_id": game_id,
            "hand_id": hand_id,
            "hero_name": hero_name,
            "villain_name": villain_name,
            "players": sorted(players_seen | set(shown_hands)),
            "blinds": blinds,
            "actions_by_street": actions_by_street,
            "board": _board_summary(board_by_street),
            "hero_hand": hero_hand,
            "villain_hand": villain_hand,
            "shown_hands": shown_hands,
            "winner": winners[0] if winners else None,
            "winners": winners,
            "pot_final": pot_total if pot_total > 0 else None,
            "pot_is_estimated": True,
            "pot_reconstruction_method": "sum_posted_bet_call_raise_amounts",
            "pot_reconstruction_sources": pot_sources,
            "active_players_by_street": active_by_street,
            "streets_available": [street for street in ("FLOP", "TURN", "RIVER") if _board_for_street(board_by_street, street)],
            "has_side_pot": False,
            "source_reliability": "reconstructed_history",
        }
        return _hand_result("ok", hand_summary, None, None, game_id, hand_id)
    except Exception as exc:  # noqa: BLE001
        return _hand_result("failed", None, "parser_exception", _format_error(exc), game_id, hand_id)


def _split_hands(text: str) -> list[str]:
    matches = list(HEADER_RE.finditer(text or ""))
    sections: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[match.start() : end].strip())
    return sections


def _parse_board_line(line: str) -> dict[str, Any] | None:
    all_board = BOARD_ALL_RE.search(line)
    if all_board:
        cards = _parse_cards(all_board.group("cards"))
        if len(cards) >= 3:
            return {"street": "RIVER" if len(cards) >= 5 else "TURN" if len(cards) == 4 else "FLOP", "cards": cards}
    board = BOARD_RE.search(line)
    if board:
        return {"street": board.group("street").upper(), "cards": _parse_cards(board.group("cards"))}
    return None


def _merge_board(board_by_street: dict[str, list[str]], street: str, cards: list[str]) -> None:
    if street == "FLOP":
        board_by_street["FLOP"] = cards[:3]
    elif street == "TURN":
        if len(cards) >= 4:
            board_by_street["FLOP"] = cards[:3]
            board_by_street["TURN"] = cards[3:4]
        else:
            board_by_street["TURN"] = cards[:1]
    elif street == "RIVER":
        if len(cards) >= 5:
            board_by_street["FLOP"] = cards[:3]
            board_by_street["TURN"] = cards[3:4]
            board_by_street["RIVER"] = cards[4:5]
        else:
            board_by_street["RIVER"] = cards[:1]


def _board_summary(board_by_street: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        "flop": list(board_by_street["FLOP"]),
        "turn": list(board_by_street["TURN"]),
        "river": list(board_by_street["RIVER"]),
        "full": _board_for_street(board_by_street, "RIVER"),
    }


def _board_for_street(board_by_street: dict[str, list[str]], street: str) -> list[str]:
    if street == "FLOP":
        return list(board_by_street["FLOP"])
    if street == "TURN":
        return list(board_by_street["FLOP"] + board_by_street["TURN"])
    if street == "RIVER":
        return list(board_by_street["FLOP"] + board_by_street["TURN"] + board_by_street["RIVER"])
    return []


def _parse_cards(text: str) -> list[str]:
    return [_normalize_card(part) for part in re.split(r"[, ]+", text.strip()) if part.strip()]


def _normalize_card(value: str) -> str:
    token = value.strip().replace("[", "").replace("]", "")
    token = token.replace("10", "T")
    suit_map = {
        "♥": "h",
        "♡": "h",
        "♠": "s",
        "♤": "s",
        "♦": "d",
        "♢": "d",
        "♣": "c",
        "♧": "c",
    }
    for raw, normalized in suit_map.items():
        token = token.replace(raw, normalized)
    if len(token) != 2:
        raise ValueError(f"invalid_card:{value}")
    rank = token[0].upper()
    suit = token[1].lower()
    if rank not in "23456789TJQKA" or suit not in "hdcs":
        raise ValueError(f"invalid_card:{value}")
    return rank + suit


def _parse_amount(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value.replace(",", "").strip().rstrip("."))


def _history_result(status: str, hands: list[dict[str, Any]], rejections: list[dict[str, Any]], error: str | None) -> dict[str, Any]:
    return {"status": status, "hands": hands, "rejections": rejections, "error": error, "warnings": []}


def _hand_result(
    status: str,
    hand_summary: dict[str, Any] | None,
    rejection_reason: str | None,
    error: str | None,
    game_id: int | None = None,
    hand_id: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "game_id": game_id,
        "hand_id": hand_id,
        "hand_summary": hand_summary,
        "rejection_reason": rejection_reason,
        "error": error,
        "warnings": [],
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
