"""Build non-label solver action candidates from hero-aligned strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solver_jobs.strategy_extractor import extract_root_strategy


MIN_DOMINANT_FREQUENCY = 0.60
MIN_ITERATIONS = 25
LABEL_QUALITY = "solver_candidate_untrusted"
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def build_solver_action_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a guarded action candidate without creating an ML label."""

    try:
        solver_job_id = _solver_job_id(payload)
        solver_status = _solver_status(payload)
        if solver_status != "ok":
            return _failed(solver_job_id, "solver_not_ok")

        strategy = _strategy_payload(payload)
        if strategy.get("status") != "ok":
            error = strategy.get("error") or "strategy_not_available"
            if error in {"root_player_not_hero", "hero_solver_player_unknown"}:
                return _failed(solver_job_id, "root_not_hero")
            if "frequency" in str(error):
                return _failed(solver_job_id, "invalid_frequencies")
            return _failed(solver_job_id, "strategy_not_available")

        if strategy.get("root_player_role") != "hero":
            return _failed(solver_job_id, "root_not_hero")

        frequencies = strategy.get("action_frequencies")
        if not _valid_frequencies(frequencies):
            return _failed(solver_job_id, "invalid_frequencies")

        dominant_action = strategy.get("dominant_action")
        dominant_frequency = _float_or_none(strategy.get("dominant_action_frequency"))
        if dominant_action is None or dominant_frequency is None:
            return _failed(solver_job_id, "strategy_not_available")
        if dominant_frequency < MIN_DOMINANT_FREQUENCY:
            return _failed(solver_job_id, "dominant_action_too_weak")

        iterations = _iterations(payload)
        if iterations is None or iterations < MIN_ITERATIONS:
            return _failed(solver_job_id, "iterations_too_low")

        confidence = str(strategy.get("confidence") or _confidence(dominant_frequency))
        if confidence not in ALLOWED_CONFIDENCE:
            confidence = _confidence(dominant_frequency)

        return _candidate(
            status="ok",
            solver_job_id=solver_job_id,
            candidate_action=str(dominant_action),
            candidate_frequency=round(float(dominant_frequency), 6),
            candidate_confidence=confidence,
            exclusion_reason=None,
            warnings=[],
        )
    except Exception as exc:  # noqa: BLE001 - stable public boundary
        return _failed(_solver_job_id(payload), f"{type(exc).__name__}:{exc}")


def _strategy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "action_frequencies" in payload or "dominant_action" in payload:
        return dict(payload)
    return extract_root_strategy(payload)


def _solver_status(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("solver_status")
    if value is not None:
        return str(value)
    solver_result = payload.get("solver_result")
    if isinstance(solver_result, Mapping):
        value = solver_result.get("status")
        if value is not None:
            return str(value)
    status = payload.get("status")
    if status is not None:
        return str(status)
    return "ok"


def _solver_job_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("solver_job_id")
    if value is not None:
        return str(value)
    solver_result = payload.get("solver_result")
    if isinstance(solver_result, Mapping) and solver_result.get("solver_job_id") is not None:
        return str(solver_result["solver_job_id"])
    solver_job = payload.get("solver_job")
    if isinstance(solver_job, Mapping) and solver_job.get("solver_job_id") is not None:
        return str(solver_job["solver_job_id"])
    return None


def _iterations(payload: Mapping[str, Any]) -> int | None:
    for value in (
        payload.get("iterations"),
        _mapping_get(payload.get("quality"), "iterations"),
        _mapping_get(payload.get("solver_result"), "iterations"),
        _mapping_get(_mapping_get(payload.get("solver_result"), "output"), "iterations"),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _valid_frequencies(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    total = 0.0
    for raw in value.values():
        frequency = _float_or_none(raw)
        if frequency is None or frequency < 0.0 or frequency > 1.0:
            return False
        total += frequency
    return 0.98 <= total <= 1.02


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence(frequency: float) -> str:
    if frequency < 0.55:
        return "low"
    if frequency <= 0.75:
        return "medium"
    return "high"


def _failed(solver_job_id: str | None, reason: str) -> dict[str, Any]:
    return _candidate(
        status="failed",
        solver_job_id=solver_job_id,
        candidate_action=None,
        candidate_frequency=0.0,
        candidate_confidence="low",
        exclusion_reason=reason,
        warnings=[],
    )


def _candidate(
    *,
    status: str,
    solver_job_id: str | None,
    candidate_action: str | None,
    candidate_frequency: float,
    candidate_confidence: str,
    exclusion_reason: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "solver_job_id": solver_job_id,
        "candidate_action": candidate_action,
        "candidate_frequency": float(candidate_frequency),
        "candidate_confidence": candidate_confidence,
        "is_training_label": False,
        "label_quality": LABEL_QUALITY,
        "exclusion_reason": exclusion_reason,
        "warnings": list(warnings),
    }
