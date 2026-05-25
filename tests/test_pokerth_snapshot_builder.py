from __future__ import annotations

from pokerth.history_parser import parse_pokerth_hand
from pokerth.snapshot_builder import build_snapshot_from_hand_summary
from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job


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


def simple_summary() -> dict:
    parsed = parse_pokerth_hand(SIMPLE_HAND)
    assert parsed["status"] == "ok", parsed["error"]
    return parsed["hand_summary"]


def test_build_solver_ready_snapshot() -> None:
    result = build_snapshot_from_hand_summary(simple_summary(), street="FLOP", to_call=0.0)

    assert result["status"] == "ok", result["error"]
    snapshot = result["snapshot"]
    assert snapshot["schema_version"] == "ml_dataset_v1"
    assert snapshot["snapshot_id"] == "pokerth_game4_hand38_flop"
    assert snapshot["metadata"]["source_type"] == "pokerth_history"
    assert snapshot["metadata"]["source_reliability"] == "reconstructed_history"
    assert snapshot["features"]["hero_cards"] == ["8h", "9s"]
    assert snapshot["features"]["villain_hand"] == ["8c", "Kc"]
    assert snapshot["features"]["board_cards"] == ["9c", "5c", "8s"]
    assert snapshot["features"]["pot"] > 0
    assert snapshot["features"]["pot_is_estimated"] is True
    assert snapshot["features"]["pot_reconstruction_method"] == "sum_posted_bet_call_raise_amounts"
    assert snapshot["features"]["to_call"] == 0.0
    assert snapshot["features"]["to_call_is_estimated"] is False
    assert snapshot["features"]["decision_context_known"] is True
    assert snapshot["quality_flags"]["usable_for_solver"] is True
    assert snapshot["quality_flags"]["usable_for_training"] is False
    assert snapshot["labels"]["training_label"] is None


def test_snapshot_compatible_with_solver_job_mapper() -> None:
    snapshot = build_snapshot_from_hand_summary(simple_summary(), street="RIVER", to_call=0.0)["snapshot"]

    result = map_snapshot_to_solver_job(snapshot)

    assert result["status"] == "ok", result["error"]
    assert result["solver_job"]["source_type"] == "pokerth_history"
    assert result["solver_job"]["street"] == "RIVER"
    assert result["solver_job"]["board"] == ["9c", "5c", "8s", "Kd", "2h"]


def test_multiway_flop_rejected_but_no_automatic_snapshot_created() -> None:
    parsed = parse_pokerth_hand(MULTIWAY_HAND)
    assert parsed["status"] == "ok", parsed["error"]

    result = build_snapshot_from_hand_summary(parsed["hand_summary"], street="FLOP", to_call=0.0)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "multiway_context_not_supported"
    assert result["snapshot"] is None


def test_multiway_preflop_heads_up_river_can_be_built_only_when_requested() -> None:
    parsed = parse_pokerth_hand(MULTIWAY_HAND)
    assert parsed["status"] == "ok", parsed["error"]

    result = build_snapshot_from_hand_summary(parsed["hand_summary"], street="RIVER", to_call=0.0)

    assert result["status"] == "ok", result["error"]
    assert result["snapshot"]["metadata"]["street"] == "RIVER"


def test_to_call_unknown_rejected_without_inventing_zero() -> None:
    result = build_snapshot_from_hand_summary(simple_summary(), street="FLOP", to_call=None)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "to_call_unknown"
    assert result["snapshot"] is None


def test_to_call_unknown_snapshot_is_not_mapped() -> None:
    result = build_snapshot_from_hand_summary(simple_summary(), street="FLOP", to_call=None)

    assert result["status"] == "failed"
    assert result["snapshot"] is None
    assert result["error"] == "to_call_unknown"


def test_invalid_board_rejected() -> None:
    summary = simple_summary()
    summary["board"]["flop"] = []

    result = build_snapshot_from_hand_summary(summary, street="FLOP", to_call=0.0)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "invalid_board"


def test_pot_reconstruction_failed_rejected() -> None:
    summary = simple_summary()
    summary["pot_final"] = None

    result = build_snapshot_from_hand_summary(summary, street="FLOP", to_call=0.0)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "pot_reconstruction_failed"


def test_winner_does_not_create_training_label() -> None:
    result = build_snapshot_from_hand_summary(simple_summary(), street="RIVER", to_call=0.0)

    assert result["status"] == "ok", result["error"]
    labels = result["snapshot"]["labels"]
    assert labels["label_source"] == "pokerth_history"
    assert labels["training_label"] is None
    assert "label_action" not in labels
    assert "gto_label" not in labels


def test_snapshot_rejection_has_stable_reason() -> None:
    summary = simple_summary()
    summary["villain_hand"] = None

    result = build_snapshot_from_hand_summary(summary, street="FLOP", to_call=0.0)

    assert result["status"] == "failed"
    assert result["rejection_reason"] == "villain_hand_missing"
    assert result["error"] == "villain_hand_missing"
