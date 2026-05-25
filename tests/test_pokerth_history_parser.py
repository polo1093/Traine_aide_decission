from __future__ import annotations

from pokerth.history_parser import parse_pokerth_hand, parse_pokerth_history


SIMPLE_HAND = """
## Game: 4 | Hand: 38 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9♣, 5♣, 8♠]
polo bets $480.
Player 3 calls $480.
*** TURN *** [K♦]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8♥,9♠]
Player 3 shows [8♣,K♣]
polo wins $1920.
"""


MULTIWAY_HAND = """
## Game: 4 | Hand: 39 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
Player 4 calls $320.
polo calls $160.
Player 3 checks.
*** FLOP *** [9♣, 5♣, 8♠]
Player 4 folds.
polo checks.
Player 3 checks.
*** TURN *** [K♦]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8♥,9♠]
Player 3 shows [8♣,K♣]
polo wins $1440.
"""


def test_parse_simple_heads_up_showdown() -> None:
    result = parse_pokerth_hand(SIMPLE_HAND)

    assert result["status"] == "ok", result["error"]
    summary = result["hand_summary"]
    assert summary["game_id"] == 4
    assert summary["hand_id"] == 38
    assert summary["hero_name"] == "polo"
    assert summary["villain_name"] == "Player 3"
    assert summary["hero_hand"] == ["8h", "9s"]
    assert summary["villain_hand"] == ["8c", "Kc"]
    assert summary["pot_is_estimated"] is True
    assert summary["pot_reconstruction_method"] == "sum_posted_bet_call_raise_amounts"
    assert summary["pot_final"] > 0


def test_parse_board_flop_turn_river() -> None:
    summary = parse_pokerth_hand(SIMPLE_HAND)["hand_summary"]

    assert summary["board"]["flop"] == ["9c", "5c", "8s"]
    assert summary["board"]["turn"] == ["Kd"]
    assert summary["board"]["river"] == ["2h"]
    assert summary["board"]["full"] == ["9c", "5c", "8s", "Kd", "2h"]
    assert summary["streets_available"] == ["FLOP", "TURN", "RIVER"]


def test_parse_history_multiple_hands_with_partial_rejection() -> None:
    text = SIMPLE_HAND + "\n" + """
## Game: 4 | Hand: 40 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo folds.
"""

    result = parse_pokerth_history(text)

    assert result["status"] == "partial"
    assert len(result["hands"]) == 1
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["rejection_reason"] == "showdown_missing"


def test_parse_no_showdown_rejected() -> None:
    text = """
## Game: 4 | Hand: 41 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo folds.
"""

    result = parse_pokerth_hand(text)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "showdown_missing"


def test_parse_side_pot_rejected() -> None:
    text = SIMPLE_HAND.replace("polo wins $1920.", "Side pot $100.\npolo wins $1920.")

    result = parse_pokerth_hand(text)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "side_pot_not_supported"


def test_parse_multiway_context_keeps_active_counts_for_builder_rejection() -> None:
    result = parse_pokerth_hand(MULTIWAY_HAND)

    assert result["status"] == "ok", result["error"]
    summary = result["hand_summary"]
    assert summary["active_players_by_street"]["FLOP"] == ["Player 3", "Player 4", "polo"]
    assert summary["active_players_by_street"]["TURN"] == ["Player 3", "polo"]
    assert summary["active_players_by_street"]["RIVER"] == ["Player 3", "polo"]


def test_parse_invalid_text_has_stable_failure() -> None:
    result = parse_pokerth_hand("not a pokerth hand")

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "invalid_hand_header"
