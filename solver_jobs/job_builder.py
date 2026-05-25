"""Builders for small, traceable PokerSolver jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from solver_jobs.job_schema import SCHEMA_VERSION, validate_solver_job


def build_solver_job(
    *,
    source_snapshot_id: str,
    street: str,
    hero_hand: list[str] | tuple[str, ...],
    board: list[str] | tuple[str, ...],
    pot: float,
    to_call: float,
    stack: float,
    villain_hand: list[str] | tuple[str, ...] | None = None,
    villain_range: str | None = None,
    bet_sizes: list[float] | tuple[float, ...] = (0.33,),
    iterations: int = 25,
    timeout_s: float = 5.0,
    backend: str = "rust",
    label_intent: str = "solver_smoke",
    solver_job_id: str | None = None,
    created_at: str | None = None,
    source_type: str = "manual_fixture",
    units: str = "chips",
) -> dict[str, Any]:
    """Build and validate one solver job without leaking raw exceptions."""

    try:
        job = {
            "solver_job_id": solver_job_id or f"solver_job_{uuid4().hex}",
            "source_snapshot_id": source_snapshot_id,
            "created_at": created_at or datetime.now(UTC).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "source_type": source_type,
            "units": units,
            "street": street,
            "hero_hand": list(hero_hand),
            "villain_hand": None if villain_hand is None else list(villain_hand),
            "villain_range": villain_range,
            "board": list(board),
            "pot": pot,
            "to_call": to_call,
            "stack": stack,
            "bet_sizes": list(bet_sizes),
            "iterations": iterations,
            "timeout_s": timeout_s,
            "backend": backend,
            "label_intent": label_intent,
        }
        validation = validate_solver_job(job)
        if validation["status"] == "failed":
            return {"status": "failed", "job": None, "error": validation["error"]}
        return {"status": "ok", "job": validation["job"], "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "job": None, "error": _format_error(exc)}


def manual_fixture_spots() -> list[dict[str, Any]]:
    """Return tiny hand-crafted job build results for tests and smoke runs."""

    return [
        build_solver_job(
            solver_job_id="solver_job_fixture_flop_simple",
            source_snapshot_id="snapshot_fixture_flop_simple",
            street="FLOP",
            hero_hand=["Ah", "Kh"],
            villain_hand=["Qd", "Qc"],
            board=["2h", "7h", "9d"],
            pot=100,
            to_call=20,
            stack=1000,
            iterations=25,
        ),
        build_solver_job(
            solver_job_id="solver_job_fixture_two_pair_drawy",
            source_snapshot_id="snapshot_fixture_two_pair_drawy",
            street="TURN",
            hero_hand=["9h", "7c"],
            villain_hand=["As", "Ad"],
            board=["9d", "7h", "6h", "2c"],
            pot=160,
            to_call=0,
            stack=900,
            iterations=25,
        ),
        build_solver_job(
            solver_job_id="solver_job_fixture_top_pair_river",
            source_snapshot_id="snapshot_fixture_top_pair_river",
            street="RIVER",
            hero_hand=["Ah", "Ts"],
            villain_hand=["Kc", "Kd"],
            board=["Ad", "7c", "2h", "4s", "9d"],
            pot=240,
            to_call=0,
            stack=1200,
            iterations=25,
        ),
        build_solver_job(
            solver_job_id="solver_job_fixture_invalid_board_missing",
            source_snapshot_id="snapshot_fixture_invalid_board_missing",
            street="FLOP",
            hero_hand=["Ah", "Kh"],
            villain_hand=["Qd", "Qc"],
            board=[],
            pot=100,
            to_call=0,
            stack=1000,
            iterations=25,
        ),
        build_solver_job(
            solver_job_id="solver_job_fixture_range_unsupported",
            source_snapshot_id="snapshot_fixture_range_unsupported",
            street="FLOP",
            hero_hand=["Ah", "Kh"],
            villain_range="QQ+,AKs",
            board=["2h", "7h", "9d"],
            pot=100,
            to_call=20,
            stack=1000,
            iterations=25,
        ),
    ]


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
