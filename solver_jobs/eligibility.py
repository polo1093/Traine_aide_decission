"""Eligibility rules for sending synthetic jobs to the solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solver_jobs.hero_oriented_builder import validate_hero_root_alignment
from solver_jobs.job_schema import validate_solver_job


SOLVER_SAFE_PROFILES = {"random_turn_spot", "random_river_spot"}
HERO_ORIENTED_SAFE_PROFILES = {
    "hero_oop_check_or_bet",
    "hero_ip_facing_bet",
    "hero_oop_facing_bet",
}
SOLVER_SAFE_STREETS = {"TURN", "RIVER"}
MAX_ELIGIBLE_ITERATIONS = 5
MAX_HERO_ORIENTED_ITERATIONS = 25
MAX_ELIGIBLE_TIMEOUT_S = 5.0
MAX_ELIGIBLE_BET_SIZES = 2


def evaluate_solver_eligibility(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return whether a job is safe to solve under current calibration rules."""

    validation = validate_solver_job(job)
    if validation["status"] != "ok":
        return _result(False, "invalid_solver_job", [validation["error"]])

    normalized = validation["job"]
    warnings: list[str] = []

    if normalized["source_type"] != "synthetic":
        return _result(False, "profile_not_solver_safe", ["source_type_not_synthetic"])

    profile = normalized.get("generation_profile")
    if profile in HERO_ORIENTED_SAFE_PROFILES:
        return _evaluate_hero_oriented_eligibility(job, normalized)

    street = normalized["street"]
    if street == "FLOP":
        return _result(False, "flop_solver_timeout_risk", warnings)
    if profile not in SOLVER_SAFE_PROFILES:
        return _result(False, "profile_not_solver_safe", warnings)
    if street not in SOLVER_SAFE_STREETS:
        return _result(False, "street_not_solver_safe", warnings)
    if normalized["iterations"] > MAX_ELIGIBLE_ITERATIONS:
        return _result(False, "iterations_too_high_for_calibration", warnings)
    if normalized["timeout_s"] > MAX_ELIGIBLE_TIMEOUT_S:
        return _result(False, "timeout_too_high_for_calibration", warnings)
    if len(normalized["bet_sizes"]) > MAX_ELIGIBLE_BET_SIZES:
        return _result(False, "profile_not_solver_safe", ["bet_sizes_too_wide_for_calibration"])

    if profile == "random_turn_spot" and street != "TURN":
        warnings.append("profile_street_mismatch")
    if profile == "random_river_spot" and street != "RIVER":
        warnings.append("profile_street_mismatch")

    return _result(True, None, warnings)


def _evaluate_hero_oriented_eligibility(
    raw_job: Mapping[str, Any],
    normalized: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    if raw_job.get("root_must_be_hero") is not True:
        return _result(False, "root_must_be_hero_required", warnings)
    if raw_job.get("decision_actor") != "hero":
        return _result(False, "decision_actor_must_be_hero", warnings)
    if normalized["street"] not in SOLVER_SAFE_STREETS:
        return _result(False, "street_not_solver_safe", warnings)
    if normalized["iterations"] > MAX_HERO_ORIENTED_ITERATIONS:
        return _result(False, "iterations_too_high_for_calibration", warnings)
    if normalized["timeout_s"] > MAX_ELIGIBLE_TIMEOUT_S:
        return _result(False, "timeout_too_high_for_calibration", warnings)
    if len(normalized["bet_sizes"]) > MAX_ELIGIBLE_BET_SIZES:
        return _result(False, "profile_not_solver_safe", ["bet_sizes_too_wide_for_calibration"])

    root_validation = validate_hero_root_alignment(normalized)
    if root_validation["status"] != "ok":
        return _result(False, root_validation["error"] or "root_alignment_failed", warnings)
    if root_validation.get("root_matches_hero") is not True:
        return _result(False, "root_player_not_hero", warnings)
    return _result(True, None, warnings)


def _result(eligible: bool, reason: str | None, warnings: list[str]) -> dict[str, Any]:
    return {
        "eligible": bool(eligible),
        "reason": reason,
        "warnings": list(warnings),
    }
