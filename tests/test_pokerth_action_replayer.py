from __future__ import annotations

from pokerth.action_replayer import replay_hero_decisions
from pokerth.history_parser import parse_pokerth_hand
from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job


SIMPLE_HAND = """
## Game: 4 | Hand: 38 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 3 bets $480.
polo calls $480.
*** TURN *** [Kd]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
Player 3 checks.
polo checks.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
polo wins $1920.
"""


RAISE_HAND = """
## Game: 4 | Hand: 39 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
polo bets $480.
Player 3 raises $1280.
polo calls $800.
*** TURN *** [Kd]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
Player 3 wins $3200.
"""


FOLD_HAND = """
## Game: 4 | Hand: 40 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 3 bets $480.
polo folds.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
Player 3 wins $1120.
"""


SIMPLE_ALL_IN_HAND = """
## Game: 4 | Hand: 42 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 3 bets $480.
polo calls $480 all-in.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
polo wins $1600.
"""


COMPLEX_ALL_IN_HAND = """
## Game: 4 | Hand: 43 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 3 bets $480.
polo calls $300 all-in.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
Player 3 wins $1420.
"""


MULTIWAY_HAND = """
## Game: 4 | Hand: 41 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
Player 4 calls $320.
polo calls $160.
Player 3 checks.
*** FLOP *** [9c, 5c, 8s]
Player 4 folds.
polo checks.
Player 3 checks.
*** TURN *** [Kd]
polo checks.
Player 3 checks.
*** RIVER *** [2h]
polo checks.
Player 3 checks.
polo shows [8h,9s]
Player 3 shows [8c,Kc]
polo wins $1440.
"""


def hand_summary(text: str = SIMPLE_HAND, *, hero_name: str = "polo") -> dict:
    parsed = parse_pokerth_hand(text, hero_name=hero_name)
    assert parsed["status"] == "ok", parsed["error"]
    return parsed["hand_summary"]


def find_context(result: dict, *, street: str, hero_action: str) -> dict:
    for context in result["decision_contexts"]:
        if context["street"] == street and context["hero_action"] == hero_action:
            return context
    raise AssertionError(f"context not found: {street} {hero_action}")


def test_blinds_create_initial_pot_before_first_hero_decision() -> None:
    result = replay_hero_decisions(hand_summary())

    assert result["status"] == "ok", result["error"]
    first = result["decision_contexts"][0]
    assert first["street"] == "PREFLOP"
    assert first["hero_action"] == "CALL"
    assert first["pot_before_action"] == 480.0
    assert first["to_call"] == 160.0


def test_big_blind_free_check_has_zero_to_call() -> None:
    result = replay_hero_decisions(hand_summary(SIMPLE_HAND, hero_name="Player 3"), hero_name="Player 3")

    assert result["status"] == "ok", result["error"]
    check = find_context(result, street="PREFLOP", hero_action="CHECK")
    assert check["to_call"] == 0.0
    assert check["can_check"] is True
    assert check["can_call"] is False


def test_villain_bet_creates_correct_hero_to_call() -> None:
    result = replay_hero_decisions(hand_summary())

    call = find_context(result, street="FLOP", hero_action="CALL")
    assert call["pot_before_action"] == 1120.0
    assert call["to_call"] == 480.0
    assert call["can_call"] is True


def test_villain_raise_creates_correct_hero_to_call() -> None:
    result = replay_hero_decisions(hand_summary(RAISE_HAND))

    assert result["status"] == "ok", result["error"]
    call = find_context(result, street="FLOP", hero_action="CALL")
    assert call["pot_before_action"] == 2400.0
    assert call["to_call"] == 800.0


def test_hero_call_updates_pot_for_next_street() -> None:
    result = replay_hero_decisions(hand_summary())

    turn_check = find_context(result, street="TURN", hero_action="CHECK")
    assert turn_check["pot_before_action"] == 1600.0
    assert turn_check["to_call"] == 0.0


def test_hero_fold_context_is_captured_before_fold() -> None:
    result = replay_hero_decisions(hand_summary(FOLD_HAND))

    fold = find_context(result, street="FLOP", hero_action="FOLD")
    assert fold["pot_before_action"] == 1120.0
    assert fold["to_call"] == 480.0
    assert fold["decision_context_known"] is True


def test_simple_all_in_call_is_replayed_when_amount_matches_to_call() -> None:
    result = replay_hero_decisions(hand_summary(SIMPLE_ALL_IN_HAND))

    assert result["status"] == "ok", result["error"]
    call = find_context(result, street="FLOP", hero_action="CALL")
    assert call["to_call"] == 480.0


def test_partial_all_in_call_is_rejected_as_complex() -> None:
    result = replay_hero_decisions(hand_summary(COMPLEX_ALL_IN_HAND))

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "all_in_complex_not_supported"


def test_street_changes_include_board_cards() -> None:
    result = replay_hero_decisions(hand_summary())

    flop = find_context(result, street="FLOP", hero_action="CALL")
    turn = find_context(result, street="TURN", hero_action="CHECK")
    river = find_context(result, street="RIVER", hero_action="CHECK")
    assert flop["board_cards"] == ["9c", "5c", "8s"]
    assert turn["board_cards"] == ["9c", "5c", "8s", "Kd"]
    assert river["board_cards"] == ["9c", "5c", "8s", "Kd", "2h"]


def test_heads_up_complete_hand_builds_solver_ready_decision_snapshot() -> None:
    result = replay_hero_decisions(hand_summary())

    assert result["status"] == "ok", result["error"]
    assert result["pot_reconstructed"] == 1600.0
    assert result["decision_snapshots"]
    snapshot = result["decision_snapshots"][0]
    assert snapshot["schema_version"] == "ml_dataset_v1"
    assert snapshot["features"]["to_call"] == 480.0
    assert snapshot["features"]["to_call_is_estimated"] is False
    assert snapshot["features"]["decision_context_known"] is True
    assert snapshot["quality_flags"]["usable_for_training"] is False
    assert snapshot["labels"]["training_label"] is None

    mapped = map_snapshot_to_solver_job(snapshot)
    assert mapped["status"] == "ok", mapped["error"]


def test_side_pot_is_rejected() -> None:
    summary = hand_summary()
    summary["has_side_pot"] = True

    result = replay_hero_decisions(summary)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "side_pot_not_supported"


def test_multiway_is_rejected() -> None:
    result = replay_hero_decisions(hand_summary(MULTIWAY_HAND))

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "multiway_context_not_supported"


def test_unknown_action_is_rejected_without_raw_exception() -> None:
    summary = hand_summary()
    summary["actions_by_street"]["FLOP"].insert(
        0,
        {"player": "Player 3", "action": "dances", "raw": "Player 3 dances."},
    )

    result = replay_hero_decisions(summary)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "unknown_action"
    assert "Traceback" not in result["error"]


def test_invalid_input_returns_stable_failure() -> None:
    result = replay_hero_decisions("not a summary")  # type: ignore[arg-type]

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "invalid_hand_summary"
    assert result["decision_contexts"] == []
