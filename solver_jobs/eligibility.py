"""Eligibility rules for sending synthetic jobs to the solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solver_jobs.job_schema import validate_solver_job


SOLVER_SAFE_PROFILES = {"random_turn_spot", "random_river_spot"}
SOLVER_SAFE_STREETS = {"TURN", "RIVER"}
MAX_ELIGIBLE_ITERATIONS = 5
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


def _result(eligible: bool, reason: str | None, warnings: list[str]) -> dict[str, Any]:
    return {
        "eligible": bool(eligible),
        "reason": reason,
        "warnings": list(warnings),
    }
