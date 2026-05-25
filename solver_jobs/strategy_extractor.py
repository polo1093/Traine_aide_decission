"""Extract inspectable root strategy data from solver_run_result records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MIN_FREQUENCY_SUM = 0.98
MAX_FREQUENCY_SUM = 1.02
LOW_CONFIDENCE_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.75


def extract_root_strategy(run_result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable strategy-inspection payload without creating labels."""

    try:
        solver_job_id = _solver_job_id(run_result)
        root_strategy = _find_root_strategy(run_result)
        if root_strategy is None:
            return _failed(solver_job_id, "strategy_not_available")
        alignment_error = _alignment_error(root_strategy)
        if alignment_error is not None:
            return _failed(solver_job_id, alignment_error)

        frequencies, evs, error = _parse_root_strategy(root_strategy)
        if error is not None:
            return _failed(solver_job_id, error)
        if not frequencies:
            return _failed(solver_job_id, "strategy_not_available")

        total = sum(frequencies.values())
        if total < MIN_FREQUENCY_SUM or total > MAX_FREQUENCY_SUM:
            return _failed(solver_job_id, "invalid_frequency_sum")

        normalized = {action: round(value / total, 6) for action, value in sorted(frequencies.items())}
        dominant_action, dominant_frequency = max(normalized.items(), key=lambda item: item[1])

        return {
            "status": "ok",
            "solver_job_id": solver_job_id,
            "available": True,
            "root_strategy": root_strategy,
            "root_player": _root_player(root_strategy),
            "root_player_role": _root_player_role(root_strategy),
            "action_frequencies": normalized,
            "action_evs": evs or None,
            "dominant_action": dominant_action,
            "dominant_action_frequency": dominant_frequency,
            "confidence": _confidence(dominant_frequency),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - public inspection boundary
        return _failed(_solver_job_id(run_result), f"{type(exc).__name__}:{exc}")


def _find_root_strategy(run_result: Mapping[str, Any]) -> Any:
    solver_result = run_result.get("solver_result")
    if isinstance(solver_result, Mapping):
        found = _find_strategy_in_mapping(solver_result)
        if found is not None:
            return found

    return _find_strategy_in_mapping(run_result)


def _find_strategy_in_mapping(payload: Mapping[str, Any]) -> Any:
    output = payload.get("output")
    if isinstance(output, Mapping):
        found = _find_strategy_in_mapping(output)
        if found is not None:
            return found

    for key in ("root_strategy_raw",):
        value = payload.get(key)
        if value is not None:
            return value

    return None


def _parse_root_strategy(value: Any) -> tuple[dict[str, float], dict[str, float], str | None]:
    if isinstance(value, Mapping):
        if "action_labels" in value and "frequencies" in value:
            return _parse_raw_strategy(value)

    return {}, {}, "strategy_not_available"


def _alignment_error(root_strategy: Any) -> str | None:
    if not isinstance(root_strategy, Mapping):
        return "strategy_not_available"
    if root_strategy.get("hero_solver_player") is None:
        return "hero_solver_player_unknown"
    if root_strategy.get("root_player_role") == "unknown":
        return "hero_solver_player_unknown"
    if root_strategy.get("root_matches_hero") is not True:
        return "root_player_not_hero"
    if root_strategy.get("root_player_role") != "hero":
        return "root_player_not_hero"
    return None


def _parse_raw_strategy(value: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], str | None]:
    labels = value.get("action_labels")
    frequencies = value.get("frequencies")
    if not isinstance(labels, list) or not isinstance(frequencies, list):
        return {}, {}, "strategy_not_available"
    if len(labels) != len(frequencies):
        return {}, {}, "invalid_frequency_shape"

    mapped: dict[str, float] = {}
    for label, raw_frequency in zip(labels, frequencies):
        try:
            frequency = float(raw_frequency)
        except (TypeError, ValueError):
            return {}, {}, "invalid_frequency_value"
        if frequency < 0.0 or frequency > 1.0:
            return {}, {}, "invalid_frequency_value"
        mapped[_normalize_action(label)] = frequency
    return mapped, {}, None


def _confidence(dominant_frequency: float) -> str:
    if dominant_frequency < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if dominant_frequency <= HIGH_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def _root_player(root_strategy: Any) -> int | None:
    if not isinstance(root_strategy, Mapping):
        return None
    value = root_strategy.get("root_player", root_strategy.get("player"))
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _root_player_role(root_strategy: Any) -> str:
    if not isinstance(root_strategy, Mapping):
        return "unknown"
    value = root_strategy.get("root_player_role")
    if value is None:
        return "unknown"
    return str(value)


def _solver_job_id(run_result: Mapping[str, Any]) -> str | None:
    value = run_result.get("solver_job_id")
    if value is not None:
        return str(value)
    solver_job = run_result.get("solver_job")
    if isinstance(solver_job, Mapping) and solver_job.get("solver_job_id") is not None:
        return str(solver_job["solver_job_id"])
    solver_result = run_result.get("solver_result")
    if isinstance(solver_result, Mapping) and solver_result.get("solver_job_id") is not None:
        return str(solver_result["solver_job_id"])
    return None


def _failed(solver_job_id: str | None, error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "solver_job_id": solver_job_id,
        "available": False,
        "root_strategy": None,
        "root_player": None,
        "root_player_role": "unknown",
        "action_frequencies": {},
        "action_evs": None,
        "dominant_action": None,
        "dominant_action_frequency": None,
        "confidence": "low",
        "error": error,
    }


def _normalize_action(value: Any) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CHECK_CALL": "CALL",
        "BET33": "BET_33",
        "BET75": "BET_75",
        "BET100": "BET_100",
        "BET150": "BET_150",
        "BET200": "BET_200",
        "RAISE33": "RAISE_33",
        "RAISE75": "RAISE_75",
        "RAISE100": "RAISE_100",
        "RAISE150": "RAISE_150",
        "RAISE200": "RAISE_200",
        "ALLIN": "ALL_IN",
    }
    return aliases.get(text, text)
