from __future__ import annotations

import time

import pytest

from solver_jobs.job_schema import validate_solver_job
from synthetic.deck import full_deck
from synthetic.spot_generator import (
    SUPPORTED_PROFILES,
    generate_solver_jobs,
    is_drawy_board,
    is_paired_board,
)


def all_cards(job: dict) -> list[str]:
    return job["hero_hand"] + job["villain_hand"] + job["board"]


def test_deck_has_52_unique_cards() -> None:
    deck = full_deck()

    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_generation_is_reproducible_for_seed_profile_and_count() -> None:
    first = generate_solver_jobs(count=5, seed=42, profile="random_flop_spot", iterations=25, timeout_s=5)
    second = generate_solver_jobs(count=5, seed=42, profile="random_flop_spot", iterations=25, timeout_s=5)

    assert first == second


def test_generated_jobs_have_no_duplicate_cards() -> None:
    for profile in SUPPORTED_PROFILES:
        jobs = generate_solver_jobs(count=20, seed=7, profile=profile, iterations=25, timeout_s=5)
        for job in jobs:
            cards = all_cards(job)
            assert len(cards) == len(set(cards))


@pytest.mark.parametrize(
    ("profile", "street", "board_count"),
    [
        ("random_flop_spot", "FLOP", 3),
        ("random_turn_spot", "TURN", 4),
        ("random_river_spot", "RIVER", 5),
        ("drawy_board_spot", "FLOP", 3),
        ("paired_board_spot", "FLOP", 3),
        ("made_hand_vs_draw_spot", "FLOP", 3),
        ("top_pair_spot", "FLOP", 3),
        ("two_pair_plus_spot", "FLOP", 3),
    ],
)
def test_board_size_matches_street(profile: str, street: str, board_count: int) -> None:
    job = generate_solver_jobs(count=1, seed=11, profile=profile, iterations=25, timeout_s=5)[0]

    assert job["street"] == street
    assert len(job["board"]) == board_count


def test_every_generated_job_passes_solver_job_validator() -> None:
    for profile in SUPPORTED_PROFILES:
        jobs = generate_solver_jobs(count=10, seed=99, profile=profile, iterations=25, timeout_s=5)
        for job in jobs:
            validation = validate_solver_job(job)
            assert validation["status"] == "ok", validation["error"]
            assert validation["job"]["source_type"] == "synthetic"


def test_amounts_and_solver_bounds_are_valid() -> None:
    jobs = generate_solver_jobs(count=100, seed=123, profile="random_river_spot", iterations=25, timeout_s=5)

    for job in jobs:
        assert job["pot"] > 0
        assert job["stack"] > 0
        assert job["to_call"] >= 0
        assert 0 < len(job["bet_sizes"]) <= 5
        assert all(size > 0 for size in job["bet_sizes"])
        assert job["iterations"] <= 100
        assert job["timeout_s"] <= 10


def test_generation_of_100_jobs_is_fast_and_does_not_call_solver(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("generator must not call run_solver_job")

    monkeypatch.setattr("solver_jobs.job_runner.run_solver_job", fail_if_called)
    started = time.perf_counter()

    jobs = generate_solver_jobs(count=100, seed=321, profile="drawy_board_spot", iterations=25, timeout_s=5)

    assert len(jobs) == 100
    assert time.perf_counter() - started < 2.0


def test_drawy_profile_produces_drawy_board_texture() -> None:
    jobs = generate_solver_jobs(count=25, seed=55, profile="drawy_board_spot", iterations=25, timeout_s=5)

    assert all(is_drawy_board(job["board"]) for job in jobs)


def test_paired_profile_produces_paired_board() -> None:
    jobs = generate_solver_jobs(count=25, seed=56, profile="paired_board_spot", iterations=25, timeout_s=5)

    assert all(is_paired_board(job["board"]) for job in jobs)


def test_unknown_profile_is_rejected_cleanly() -> None:
    with pytest.raises(ValueError, match="unsupported_generation_profile"):
        generate_solver_jobs(count=1, seed=1, profile="unknown_profile", iterations=25, timeout_s=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0, "seed": 1, "profile": "random_flop_spot", "iterations": 25, "timeout_s": 5}, "count_must_be_positive"),
        ({"count": 1001, "seed": 1, "profile": "random_flop_spot", "iterations": 25, "timeout_s": 5}, "count_exceeds_limit"),
        ({"count": 1, "seed": 1, "profile": "random_flop_spot", "iterations": 101, "timeout_s": 5}, "iterations_exceeds_limit"),
        ({"count": 1, "seed": 1, "profile": "random_flop_spot", "iterations": 25, "timeout_s": 10.1}, "timeout_s_exceeds_limit"),
    ],
)
def test_invalid_parameters_are_rejected_cleanly(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_solver_jobs(**kwargs)
