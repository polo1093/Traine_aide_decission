"""Run bounded solver jobs through the PokerSolver adapter."""

from __future__ import annotations

import time
from typing import Any

from solvers.poker_solver_adapter import solve_tiny_postflop_spot
from solver_jobs.hero_oriented_builder import validate_hero_root_alignment
from solver_jobs.job_schema import validate_solver_job


RUNNER_KEYS = ("status", "solver_job_id", "input", "output", "error", "duration_ms", "quality")


def run_solver_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run a validated solver job and always return a stable result."""

    started = time.perf_counter()
    solver_job_id = _job_id(job)
    try:
        validation = validate_solver_job(job)
        if validation["status"] == "failed":
            return _runner_result(
                started,
                solver_job_id,
                dict(job) if isinstance(job, dict) else {"raw_job": job},
                None,
                validation["error"],
                _quality(None, None, "job_validation_failed"),
            )

        normalized_job = validation["job"]
        root_validation = validate_hero_root_alignment(normalized_job)
        if root_validation["status"] != "ok":
            return _runner_result(
                started,
                normalized_job["solver_job_id"],
                normalized_job,
                {
                    "root_player": root_validation.get("root_player"),
                    "hero_solver_player": root_validation.get("hero_solver_player"),
                    "root_matches_hero": root_validation.get("root_matches_hero"),
                    "root_player_role": root_validation.get("root_player_role"),
                    "legal_action_ids": root_validation.get("legal_action_ids", []),
                    "legal_action_labels": root_validation.get("legal_action_labels", []),
                },
                root_validation.get("error") or "root_validation_failed",
                _quality(normalized_job["iterations"], None, "root_validation_failed"),
            )

        adapter_result = solve_tiny_postflop_spot(
            normalized_job["hero_hand"],
            normalized_job["villain_hand"],
            villain_range=normalized_job["villain_range"],
            board=normalized_job["board"],
            pot=normalized_job["pot"],
            stack=normalized_job["stack"],
            bet_sizes=normalized_job["bet_sizes"],
            iterations=normalized_job["iterations"],
            backend=normalized_job["backend"],
            timeout_s=normalized_job["timeout_s"],
            hero_solver_player=normalized_job["hero_solver_player"],
            decision_actor=normalized_job["decision_actor"],
            root_must_be_hero=normalized_job["root_must_be_hero"],
            initial_hole_cards=normalized_job["initial_hole_cards"],
            initial_contributions=normalized_job["initial_contributions"],
        )

        output = adapter_result.get("output")
        if adapter_result.get("status") != "ok":
            return _runner_result(
                started,
                normalized_job["solver_job_id"],
                normalized_job,
                output,
                adapter_result.get("error") or "solver_failed",
                _quality(normalized_job["iterations"], output, "solver_failed"),
            )

        reason = _quality_exclusion_reason(normalized_job["iterations"], output, None)
        return _runner_result(
            started,
            normalized_job["solver_job_id"],
            normalized_job,
            output,
            None,
            _quality(normalized_job["iterations"], output, reason),
        )
    except Exception as exc:  # noqa: BLE001
        return _runner_result(
            started,
            solver_job_id,
            dict(job) if isinstance(job, dict) else {"raw_job": job},
            None,
            _format_error(exc),
            _quality(None, None, "runner_exception"),
        )


def _quality(iterations: int | None, output: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    exploitability_last = None
    if output:
        history = output.get("exploitability_history") or []
        if history:
            exploitability_last = history[-1]
    return {
        "iterations": iterations,
        "exploitability_last": exploitability_last,
        "is_label_candidate": False,
        "exclusion_reason": reason,
    }


def _quality_exclusion_reason(
    iterations: int,
    output: dict[str, Any] | None,
    error: str | None,
) -> str:
    if error is not None:
        return "solver_failed"
    if output is None:
        return "result_incomplete"
    if iterations < 100:
        return "iterations_too_low"
    if not output.get("exploitability_history"):
        return "exploitability_missing"
    if output.get("strategy_entry_count", 0) <= 0:
        return "strategy_entry_count_too_low"
    return "labeling_disabled"


def _runner_result(
    started: float,
    solver_job_id: str | None,
    input_payload: dict[str, Any],
    output: dict[str, Any] | None,
    error: str | None,
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "failed" if error else "ok",
        "solver_job_id": solver_job_id,
        "input": input_payload,
        "output": output,
        "error": error,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "quality": quality,
    }


def _job_id(job: Any) -> str | None:
    if isinstance(job, dict):
        value = job.get("solver_job_id")
        return str(value) if value is not None else None
    return None


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
