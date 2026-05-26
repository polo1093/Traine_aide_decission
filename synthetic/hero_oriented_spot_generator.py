"""Deterministic synthetic hero-oriented solver job generation."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from solver_jobs.hero_oriented_builder import (
    build_hero_oriented_solver_job,
    validate_hero_root_alignment,
)
from solver_jobs.job_schema import MAX_ITERATIONS, MAX_TIMEOUT_S, validate_solver_job
from synthetic.deck import draw_cards, validate_unique_cards


SUPPORTED_HERO_CONTEXTS = {
    "hero_oop_check_or_bet",
    "hero_ip_facing_bet",
    "hero_oop_facing_bet",
}
SUPPORTED_HERO_STREETS = {"RIVER", "TURN"}
DEFAULT_CREATED_AT = "2026-05-25T00:00:00+00:00"
DEFAULT_MAX_COUNT = 1000
STREET_BOARD_COUNTS = {"TURN": 4, "RIVER": 5}


def generate_hero_oriented_solver_jobs(
    *,
    count: int,
    seed: int,
    context: str,
    street: str = "RIVER",
    iterations: int = 25,
    timeout_s: float = 5.0,
    backend: str = "rust",
) -> list[dict[str, Any]]:
    """Return synthetic jobs whose solver root is validated as hero.

    The generator only creates bounded solver jobs. It never solves, labels, or
    marks output as a label candidate.
    """

    safe_count = _validate_count(count)
    safe_seed = _validate_seed(seed)
    safe_context = _validate_context(context)
    safe_street = _validate_street(street)
    safe_iterations = _validate_iterations(iterations)
    safe_timeout_s = _validate_timeout(timeout_s)

    return [
        _generate_one_job(
            seed=safe_seed,
            context=safe_context,
            street=safe_street,
            index=index,
            iterations=safe_iterations,
            timeout_s=safe_timeout_s,
            backend=backend,
        )
        for index in range(safe_count)
    ]


def _generate_one_job(
    *,
    seed: int,
    context: str,
    street: str,
    index: int,
    iterations: int,
    timeout_s: float,
    backend: str,
) -> dict[str, Any]:
    rng = random.Random(_derived_seed(seed, context, street, index))
    board_count = STREET_BOARD_COUNTS[street]
    cards = draw_cards(rng, 4 + board_count)
    hero_hand = cards[:2]
    villain_hand = cards[2:4]
    board = cards[4:]
    validate_unique_cards(hero_hand + villain_hand + board)

    pot, to_call = _amounts(rng, context)
    position, decision_type = _context_fields(context)
    id_suffix = f"{context}_{street.lower()}_seed_{seed}_{index:06d}"
    built = build_hero_oriented_solver_job(
        solver_job_id=f"synthetic_hero_solver_job_{id_suffix}",
        source_snapshot_id=f"synthetic_hero_snapshot_{id_suffix}",
        created_at=DEFAULT_CREATED_AT,
        source_type="synthetic",
        units="bb",
        street=street,
        hero_hand=hero_hand,
        villain_hand=villain_hand,
        board=board,
        pot=pot,
        to_call=to_call,
        stack=100.0,
        bet_sizes=[0.33],
        iterations=iterations,
        timeout_s=timeout_s,
        backend=backend,
        hero_position_model=position,
        decision_context_type=decision_type,
        root_must_be_hero=True,
    )
    if built["status"] != "ok":
        raise ValueError(f"generated_invalid_hero_solver_job:{built['error']}")

    job = dict(built["job"])
    job.update(
        {
            "generation_seed": seed,
            "generation_profile": context,
            "generation_index": index,
            "generation_street": street,
            "hero_oriented": True,
        }
    )
    validation = validate_solver_job(job)
    if validation["status"] != "ok":
        raise ValueError(f"generated_invalid_solver_job:{validation['error']}")
    root_validation = validate_hero_root_alignment(validation["job"])
    if root_validation["status"] != "ok":
        raise ValueError(f"generated_root_alignment_failed:{root_validation['error']}")
    return validation["job"]


def _amounts(rng: random.Random, context: str) -> tuple[float, float]:
    if context == "hero_oop_check_or_bet":
        return float(rng.choice([10, 12, 16, 20])), 0.0
    pot = float(rng.choice([12, 16, 20, 24]))
    to_call = float(rng.choice([2, 4, 6]))
    if to_call >= pot:
        to_call = pot / 4.0
    return pot, to_call


def _context_fields(context: str) -> tuple[str, str]:
    if context == "hero_oop_check_or_bet":
        return "OOP", "hero_check_or_bet"
    if context == "hero_ip_facing_bet":
        return "IP", "hero_facing_bet"
    if context == "hero_oop_facing_bet":
        return "OOP", "hero_facing_bet"
    raise ValueError(f"unsupported_hero_context:{context}")


def _validate_count(count: int) -> int:
    safe_count = int(count)
    if safe_count <= 0:
        raise ValueError("count_must_be_positive")
    if safe_count > DEFAULT_MAX_COUNT:
        raise ValueError(f"count_exceeds_limit:{DEFAULT_MAX_COUNT}")
    return safe_count


def _validate_seed(seed: int) -> int:
    return int(seed)


def _validate_context(context: str) -> str:
    safe_context = str(context).strip()
    if safe_context not in SUPPORTED_HERO_CONTEXTS:
        raise ValueError(f"unsupported_hero_context:{safe_context}")
    return safe_context


def _validate_street(street: str) -> str:
    safe_street = str(street).strip().upper()
    if safe_street not in SUPPORTED_HERO_STREETS:
        raise ValueError(f"unsupported_hero_street:{safe_street}")
    return safe_street


def _validate_iterations(iterations: int) -> int:
    safe_iterations = int(iterations)
    if safe_iterations <= 0:
        raise ValueError("iterations_must_be_positive")
    if safe_iterations > MAX_ITERATIONS:
        raise ValueError(f"iterations_exceeds_limit:{MAX_ITERATIONS}")
    return safe_iterations


def _validate_timeout(timeout_s: float) -> float:
    safe_timeout = float(timeout_s)
    if safe_timeout <= 0:
        raise ValueError("timeout_s_must_be_positive")
    if safe_timeout > MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s_exceeds_limit:{MAX_TIMEOUT_S:g}")
    return safe_timeout


def _derived_seed(seed: int, context: str, street: str, index: int) -> int:
    material = f"{seed}:{context}:{street}:{index}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)
