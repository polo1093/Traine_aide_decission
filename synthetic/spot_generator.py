"""Deterministic synthetic postflop spot generation for solver_job_v1."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from solver_jobs.job_schema import (
    MAX_ITERATIONS,
    MAX_TIMEOUT_S,
    SCHEMA_VERSION,
    validate_solver_job,
)
from synthetic.deck import RANKS, SUITS, draw_cards, full_deck, validate_unique_cards


SUPPORTED_PROFILES = {
    "random_flop_spot",
    "random_turn_spot",
    "random_river_spot",
    "drawy_board_spot",
    "paired_board_spot",
    "made_hand_vs_draw_spot",
    "top_pair_spot",
    "two_pair_plus_spot",
}
DEFAULT_MAX_COUNT = 1000
DEFAULT_CREATED_AT = "2026-05-25T00:00:00+00:00"
DEFAULT_BET_SIZE_POOLS = (
    (0.33,),
    (0.5,),
    (0.66,),
    (0.33, 0.66),
    (0.5, 1.0),
)
STREET_BOARD_COUNTS = {"FLOP": 3, "TURN": 4, "RIVER": 5}
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}


def generate_solver_jobs(
    count: int,
    seed: int,
    profile: str,
    iterations: int = 25,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Return deterministic synthetic ``solver_job_v1`` dictionaries.

    The generator only creates jobs. It never calls the solver runner and never
    promotes solver output to ML labels.
    """

    safe_count = _validate_count(count)
    safe_seed = _validate_seed(seed)
    safe_profile = _validate_profile(profile)
    safe_iterations = _validate_iterations(iterations)
    safe_timeout_s = _validate_timeout(timeout_s)

    return [
        _generate_one_job(
            seed=safe_seed,
            profile=safe_profile,
            index=index,
            iterations=safe_iterations,
            timeout_s=safe_timeout_s,
        )
        for index in range(safe_count)
    ]


def is_paired_board(board: list[str] | tuple[str, ...]) -> bool:
    """Return whether the board contains at least one paired rank."""

    ranks = [card[0].upper() for card in board]
    return len(set(ranks)) < len(ranks)


def is_drawy_board(board: list[str] | tuple[str, ...]) -> bool:
    """Return a deliberately simple draw texture check.

    This is a texture helper, not a full poker evaluator. It flags boards with
    at least two suited cards or with three ranks close enough to suggest a
    straight draw.
    """

    if len(board) < 3:
        return False
    suits = [card[1].lower() for card in board]
    if any(suits.count(suit) >= 2 for suit in set(suits)):
        return True

    indexes = sorted({RANK_INDEX[card[0].upper()] for card in board})
    for start in range(len(indexes) - 2):
        window = indexes[start : start + 3]
        if window[-1] - window[0] <= 4:
            return True
    return False


def _generate_one_job(
    *,
    seed: int,
    profile: str,
    index: int,
    iterations: int,
    timeout_s: float,
) -> dict[str, Any]:
    rng = random.Random(_derived_seed(seed, profile, index))
    street, hero_hand, villain_hand, board = _generate_cards(rng, profile)
    pot, to_call, stack, bet_sizes = _generate_amounts(rng)

    id_suffix = f"{profile}_seed_{seed}_{index:06d}"
    job = {
        "solver_job_id": f"synthetic_solver_job_{id_suffix}",
        "source_snapshot_id": f"synthetic_snapshot_{id_suffix}",
        "created_at": DEFAULT_CREATED_AT,
        "schema_version": SCHEMA_VERSION,
        "source_type": "synthetic",
        "units": "chips",
        "street": street,
        "hero_hand": hero_hand,
        "villain_hand": villain_hand,
        "villain_range": None,
        "board": board,
        "pot": pot,
        "to_call": to_call,
        "stack": stack,
        "bet_sizes": bet_sizes,
        "iterations": iterations,
        "timeout_s": timeout_s,
        "backend": "rust",
        "label_intent": "solver_smoke",
        "generation_seed": seed,
        "generation_profile": profile,
        "generation_index": index,
    }

    validation = validate_solver_job(job)
    if validation["status"] != "ok":
        raise ValueError(f"generated_invalid_solver_job:{validation['error']}")
    return validation["job"]


def _generate_cards(rng: random.Random, profile: str) -> tuple[str, list[str], list[str], list[str]]:
    if profile == "random_flop_spot":
        return _random_spot(rng, "FLOP")
    if profile == "random_turn_spot":
        return _random_spot(rng, "TURN")
    if profile == "random_river_spot":
        return _random_spot(rng, "RIVER")
    if profile == "drawy_board_spot":
        return _drawy_board_spot(rng)
    if profile == "paired_board_spot":
        return _paired_board_spot(rng)
    if profile == "made_hand_vs_draw_spot":
        return _made_hand_vs_draw_spot(rng)
    if profile == "top_pair_spot":
        return _top_pair_spot(rng)
    if profile == "two_pair_plus_spot":
        return _two_pair_plus_spot(rng)
    raise ValueError(f"unsupported_generation_profile:{profile}")


def _random_spot(rng: random.Random, street: str) -> tuple[str, list[str], list[str], list[str]]:
    cards = draw_cards(rng, 4 + STREET_BOARD_COUNTS[street])
    hero_hand = cards[:2]
    villain_hand = cards[2:4]
    board = cards[4:]
    return street, hero_hand, villain_hand, board


def _drawy_board_spot(rng: random.Random) -> tuple[str, list[str], list[str], list[str]]:
    suit = rng.choice(SUITS)
    off_suit = rng.choice([candidate for candidate in SUITS if candidate != suit])
    start = rng.randrange(0, len(RANKS) - 3)
    board = [
        f"{RANKS[start + 1]}{suit}",
        f"{RANKS[start + 2]}{suit}",
        f"{RANKS[start + 3]}{off_suit}",
    ]
    return _complete_flop_spot(rng, board)


def _paired_board_spot(rng: random.Random) -> tuple[str, list[str], list[str], list[str]]:
    pair_rank = rng.choice(RANKS)
    pair_suits = rng.sample(list(SUITS), 2)
    kicker_rank = rng.choice([rank for rank in RANKS if rank != pair_rank])
    kicker_suit = rng.choice(SUITS)
    board = [f"{pair_rank}{pair_suits[0]}", f"{pair_rank}{pair_suits[1]}", f"{kicker_rank}{kicker_suit}"]
    return _complete_flop_spot(rng, board)


def _made_hand_vs_draw_spot(rng: random.Random) -> tuple[str, list[str], list[str], list[str]]:
    suit = rng.choice(SUITS)
    off_suit = rng.choice([candidate for candidate in SUITS if candidate != suit])
    made_rank = rng.choice("9TJQKA")
    draw_ranks = _nearby_draw_ranks(made_rank)
    board = [f"{made_rank}{suit}", f"{draw_ranks[0]}{suit}", f"2{off_suit}"]
    hero_hand = [f"{made_rank}{off_suit}", f"7{rng.choice([candidate for candidate in SUITS if candidate != off_suit])}"]
    villain_hand = [
        f"{draw_ranks[1]}{suit}",
        f"{draw_ranks[2]}{suit}",
    ]
    return _complete_constrained_spot(rng, "FLOP", hero_hand, villain_hand, board)


def _top_pair_spot(rng: random.Random) -> tuple[str, list[str], list[str], list[str]]:
    top_rank = rng.choice("TJQKA")
    lower_ranks = [rank for rank in RANKS if RANK_INDEX[rank] < RANK_INDEX[top_rank]]
    board_suit = rng.choice(SUITS)
    hero_suit = rng.choice([suit for suit in SUITS if suit != board_suit])
    top_cards = [f"{top_rank}{board_suit}", f"{top_rank}{hero_suit}"]
    lower_deck = [card for card in full_deck() if card[0] in lower_ranks and card not in top_cards]
    rng.shuffle(lower_deck)
    board = [top_cards[0], lower_deck[0], lower_deck[1]]
    hero_hand = [top_cards[1], lower_deck[2]]
    return _complete_partial_spot(rng, "FLOP", hero_hand, board)


def _two_pair_plus_spot(rng: random.Random) -> tuple[str, list[str], list[str], list[str]]:
    ranks = rng.sample(list(RANKS), 3)
    first_suits = rng.sample(list(SUITS), 2)
    second_suits = rng.sample(list(SUITS), 2)
    board = [
        f"{ranks[0]}{first_suits[0]}",
        f"{ranks[1]}{second_suits[0]}",
        f"{ranks[2]}{rng.choice(SUITS)}",
    ]
    hero_hand = [f"{ranks[0]}{first_suits[1]}", f"{ranks[1]}{second_suits[1]}"]
    return _complete_partial_spot(rng, "FLOP", hero_hand, board)


def _complete_flop_spot(
    rng: random.Random,
    board: list[str],
) -> tuple[str, list[str], list[str], list[str]]:
    validate_unique_cards(board)
    hands = draw_cards(rng, 4, excluded=board)
    return "FLOP", hands[:2], hands[2:4], board


def _complete_partial_spot(
    rng: random.Random,
    street: str,
    hero_hand: list[str],
    board: list[str],
) -> tuple[str, list[str], list[str], list[str]]:
    validate_unique_cards(hero_hand + board)
    villain_hand = draw_cards(rng, 2, excluded=hero_hand + board)
    return street, hero_hand, villain_hand, board


def _complete_constrained_spot(
    rng: random.Random,
    street: str,
    hero_hand: list[str],
    villain_hand: list[str],
    board: list[str],
) -> tuple[str, list[str], list[str], list[str]]:
    validate_unique_cards(hero_hand + villain_hand + board)
    expected_board_count = STREET_BOARD_COUNTS[street]
    if len(board) < expected_board_count:
        board = board + draw_cards(rng, expected_board_count - len(board), excluded=hero_hand + villain_hand + board)
    return street, hero_hand, villain_hand, board


def _nearby_draw_ranks(anchor_rank: str) -> list[str]:
    index = RANK_INDEX[anchor_rank]
    if index >= len(RANKS) - 3:
        return [RANKS[index - 1], RANKS[index - 2], RANKS[index - 3]]
    return [RANKS[index + 1], RANKS[index + 2], RANKS[index + 3]]


def _generate_amounts(rng: random.Random) -> tuple[float, float, float, list[float]]:
    pot = float(rng.choice([40, 60, 80, 100, 120, 160, 200]))
    stack = float(rng.choice([400, 600, 800, 1000, 1500, 2000]))
    to_call = float(rng.choice([0, round(pot * 0.25, 2), round(pot * 0.5, 2)]))
    bet_sizes = [float(size) for size in rng.choice(DEFAULT_BET_SIZE_POOLS)]
    if len(bet_sizes) > 5:
        raise ValueError("bet_sizes_exceeds_limit:5")
    return pot, to_call, stack, bet_sizes


def _validate_count(count: int) -> int:
    try:
        safe_count = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("count_must_be_integer") from exc
    if safe_count <= 0:
        raise ValueError("count_must_be_positive")
    if safe_count > DEFAULT_MAX_COUNT:
        raise ValueError(f"count_exceeds_limit:{DEFAULT_MAX_COUNT}")
    return safe_count


def _validate_seed(seed: int) -> int:
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed_must_be_integer") from exc


def _validate_profile(profile: str) -> str:
    safe_profile = str(profile).strip()
    if safe_profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported_generation_profile:{safe_profile}")
    return safe_profile


def _validate_iterations(iterations: int) -> int:
    try:
        safe_iterations = int(iterations)
    except (TypeError, ValueError) as exc:
        raise ValueError("iterations_must_be_integer") from exc
    if safe_iterations <= 0:
        raise ValueError("iterations_must_be_positive")
    if safe_iterations > MAX_ITERATIONS:
        raise ValueError(f"iterations_exceeds_limit:{MAX_ITERATIONS}")
    return safe_iterations


def _validate_timeout(timeout_s: float) -> float:
    try:
        safe_timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_s_must_be_number") from exc
    if safe_timeout <= 0:
        raise ValueError("timeout_s_must_be_positive")
    if safe_timeout > MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s_exceeds_limit:{MAX_TIMEOUT_S:g}")
    return safe_timeout


def _derived_seed(seed: int, profile: str, index: int) -> int:
    material = f"{seed}:{profile}:{index}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def _assert_deck_is_complete() -> None:
    deck = full_deck()
    if len(deck) != 52:
        raise ValueError("deck_size_invalid")
    validate_unique_cards(deck)


_assert_deck_is_complete()
