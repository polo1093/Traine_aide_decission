"""Hero-oriented solver job construction and root-alignment validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from solvers import poker_solver_adapter as adapter
from solver_jobs.job_builder import build_solver_job
from solver_jobs.job_schema import validate_solver_job


ACTION_LABELS = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "BET_33",
    4: "BET_75",
    5: "BET_100",
    6: "BET_150",
    7: "BET_200",
    8: "RAISE_33",
    9: "RAISE_75",
    10: "RAISE_100",
    11: "RAISE_150",
    12: "RAISE_200",
    13: "ALL_IN",
}


def build_hero_oriented_solver_job(
    *,
    source_snapshot_id: str,
    street: str,
    hero_hand: list[str] | tuple[str, ...],
    villain_hand: list[str] | tuple[str, ...],
    board: list[str] | tuple[str, ...],
    pot: float,
    stack: float,
    hero_position_model: str,
    decision_context_type: str,
    to_call: float = 0.0,
    bet_sizes: list[float] | tuple[float, ...] = (0.33,),
    iterations: int = 25,
    timeout_s: float = 5.0,
    backend: str = "rust",
    solver_job_id: str | None = None,
    created_at: str | None = None,
    source_type: str = "manual_fixture",
    units: str = "bb",
    root_must_be_hero: bool = True,
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a solver job whose root player is intentionally the hero.

    The returned job remains a solver-inspection artifact only. It contains no
    training label and is never marked as a label candidate.
    """

    try:
        hero_solver_player = _hero_solver_player(
            hero_position_model=hero_position_model,
            decision_context_type=decision_context_type,
        )
        villain_solver_player = 1 - hero_solver_player
        initial_hole_cards = _ordered_hole_cards(
            hero_hand=hero_hand,
            villain_hand=villain_hand,
            hero_solver_player=hero_solver_player,
        )
        initial_contributions = _initial_contributions(
            pot=pot,
            to_call=to_call,
            hero_solver_player=hero_solver_player,
            decision_context_type=decision_context_type,
        )

        built = build_solver_job(
            solver_job_id=solver_job_id,
            source_snapshot_id=source_snapshot_id,
            created_at=created_at,
            source_type=source_type,
            units=units,
            street=street,
            hero_hand=hero_hand,
            villain_hand=villain_hand,
            villain_range=None,
            board=board,
            pot=pot,
            to_call=to_call,
            stack=stack,
            bet_sizes=bet_sizes,
            iterations=iterations,
            timeout_s=timeout_s,
            backend=backend,
            label_intent="solver_smoke",
            hero_solver_player=hero_solver_player,
            villain_solver_player=villain_solver_player,
            decision_actor="hero",
            root_must_be_hero=root_must_be_hero,
            hero_position_model=hero_position_model,
            decision_context_type=decision_context_type,
            initial_hole_cards=initial_hole_cards,
            initial_contributions=initial_contributions,
        )
        if built["status"] != "ok":
            return {**built, "root_validation": None}

        root_validation = validate_hero_root_alignment(built["job"], solver_path=solver_path)
        if root_validation["status"] != "ok":
            return {
                "status": "failed",
                "job": None,
                "error": root_validation["error"],
                "root_validation": root_validation,
            }
        return {**built, "root_validation": root_validation}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "job": None,
            "error": _format_error(exc),
            "root_validation": None,
        }


def validate_hero_root_alignment(
    job: dict[str, Any],
    *,
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild ``HUNLPoker(config)`` and verify the root actor before solve."""

    validation = validate_solver_job(job)
    if validation["status"] != "ok":
        return _root_result(None, None, [], validation["error"])

    normalized = validation["job"]
    try:
        solver, error, _ = adapter._load_poker_solver(solver_path)  # noqa: SLF001
        if error is not None:
            return _root_result(None, normalized, [], error)

        config = _hunl_config_from_job(solver=solver, job=normalized)
        game = solver.HUNLPoker(config)
        state = game.initial_state()
        root_player = game.current_player(state)
        actions = list(game.legal_actions(state))
        root_matches_hero = root_player == normalized["hero_solver_player"]
        if normalized["root_must_be_hero"] and not root_matches_hero:
            return _root_result(root_player, normalized, actions, "root_player_not_hero")
        return _root_result(root_player, normalized, actions, None)
    except Exception as exc:  # noqa: BLE001
        return _root_result(None, normalized, [], _format_error(exc))


def _hunl_config_from_job(*, solver: Any, job: dict[str, Any]) -> Any:
    board = tuple(solver.parse_board(adapter._cards_to_solver_string(job["board"])))  # noqa: SLF001
    street = adapter._street_from_board_length(solver, len(board))  # noqa: SLF001
    if street is None:
        raise ValueError("unsupported_board_length_for_postflop")
    hole_cards = (
        tuple(solver.parse_hand(adapter._cards_to_solver_string(job["initial_hole_cards"][0]))),  # noqa: SLF001
        tuple(solver.parse_hand(adapter._cards_to_solver_string(job["initial_hole_cards"][1]))),  # noqa: SLF001
    )
    pot_cents = adapter._bb_to_cents(job["pot"])  # noqa: SLF001
    stack_cents = max(adapter._bb_to_cents(job["stack"]), pot_cents, 100)  # noqa: SLF001
    contribution_cents = (
        adapter._bb_to_cents(job["initial_contributions"][0]),  # noqa: SLF001
        adapter._bb_to_cents(job["initial_contributions"][1]),  # noqa: SLF001
    )
    return solver.HUNLConfig(
        starting_stack=stack_cents,
        starting_street=street,
        initial_board=board,
        initial_pot=pot_cents,
        initial_contributions=contribution_cents,
        initial_hole_cards=hole_cards,
        bet_size_fractions=tuple(float(size) for size in job["bet_sizes"]),
    )


def _hero_solver_player(*, hero_position_model: str, decision_context_type: str) -> int:
    if decision_context_type == "hero_check_or_bet":
        if hero_position_model != "OOP":
            raise ValueError("hero_check_or_bet_requires_hero_oop")
        return 1
    if decision_context_type == "hero_facing_bet":
        if hero_position_model == "IP":
            return 0
        if hero_position_model == "OOP":
            return 1
        raise ValueError("hero_facing_bet_requires_known_position")
    raise ValueError(f"unsupported_decision_context_type:{decision_context_type}")


def _ordered_hole_cards(
    *,
    hero_hand: list[str] | tuple[str, ...],
    villain_hand: list[str] | tuple[str, ...],
    hero_solver_player: int,
) -> list[list[str]]:
    if hero_solver_player == 0:
        return [list(hero_hand), list(villain_hand)]
    return [list(villain_hand), list(hero_hand)]


def _initial_contributions(
    *,
    pot: float,
    to_call: float,
    hero_solver_player: int,
    decision_context_type: str,
) -> list[float]:
    pot_value = float(pot)
    to_call_value = float(to_call)
    if decision_context_type == "hero_check_or_bet":
        if to_call_value != 0:
            raise ValueError("hero_check_or_bet_requires_zero_to_call")
        left = pot_value / 2.0
        return [left, pot_value - left]
    if decision_context_type == "hero_facing_bet":
        if to_call_value <= 0:
            raise ValueError("hero_facing_bet_requires_positive_to_call")
        if pot_value < to_call_value:
            raise ValueError("pot_must_cover_to_call")
        lower = (pot_value - to_call_value) / 2.0
        higher = lower + to_call_value
        if hero_solver_player == 0:
            return [lower, higher]
        return [higher, lower]
    raise ValueError(f"unsupported_decision_context_type:{decision_context_type}")


def _root_result(
    root_player: Any,
    job: dict[str, Any] | None,
    actions: list[int],
    error: str | None,
) -> dict[str, Any]:
    hero_solver_player = None if job is None else job.get("hero_solver_player")
    root_matches_hero = None if root_player is None or hero_solver_player is None else root_player == hero_solver_player
    root_player_role = "unknown"
    if root_matches_hero is True:
        root_player_role = "hero"
    elif root_matches_hero is False:
        root_player_role = "villain"
    return {
        "status": "failed" if error else "ok",
        "error": error,
        "root_player": root_player,
        "hero_solver_player": hero_solver_player,
        "root_matches_hero": root_matches_hero,
        "root_player_role": root_player_role,
        "legal_action_ids": [int(action) for action in actions],
        "legal_action_labels": [ACTION_LABELS.get(int(action), f"ACTION_{int(action)}") for action in actions],
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
