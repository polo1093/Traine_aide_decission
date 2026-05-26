"""Quality/stability checks for non-label solver action candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from solver_jobs.action_candidate import build_solver_action_candidate


MIN_RUNS = 2
MIN_DOMINANT_FREQUENCY = 0.60
MIN_ITERATIONS = 25
LABEL_QUALITY = "solver_candidate_untrusted"


def evaluate_candidate_quality(
    runs: Sequence[Mapping[str, Any]],
    *,
    short_stack_explicit: bool = False,
) -> dict[str, Any]:
    """Evaluate multi-run candidate stability without producing labels."""

    try:
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            return _failed(None, "runs_must_be_sequence", ["single_run_only"])
        if len(runs) < MIN_RUNS:
            return _failed(None, "single_run_only", ["single_run_only"])

        observations: list[dict[str, Any]] = []
        danger_flags: list[str] = []
        solver_job_id = _solver_job_id(runs[0])

        for run in runs:
            run_error = _run_error(run)
            if run_error is not None:
                return _failed(solver_job_id, run_error, _dedupe([*danger_flags, run_error]))

            candidate = _candidate_payload(run)
            if candidate.get("status") != "ok":
                reason = candidate.get("exclusion_reason") or "candidate_not_available"
                mapped = _map_candidate_reason(str(reason))
                return _failed(solver_job_id, mapped, _dedupe([*danger_flags, _flag_for_reason(mapped)]))

            if _root_not_hero(run):
                return _failed(solver_job_id, "root_not_hero", _dedupe([*danger_flags, "root_not_hero"]))

            iterations = _iterations(run)
            if iterations is None or iterations < MIN_ITERATIONS:
                danger_flags.append("iterations_too_low")
                return _failed(solver_job_id, "iterations_too_low", _dedupe(danger_flags))

            exploitability = _exploitability_last(run)
            if exploitability is None:
                danger_flags.append("exploitability_missing")
                return _failed(solver_job_id, "exploitability_missing", _dedupe(danger_flags))

            action = candidate.get("candidate_action")
            frequency = _float_or_none(candidate.get("candidate_frequency"))
            if not action or frequency is None:
                return _failed(solver_job_id, "candidate_not_available", _dedupe(danger_flags))
            if frequency < MIN_DOMINANT_FREQUENCY:
                danger_flags.append("frequency_too_low")
                return _failed(solver_job_id, "frequency_too_low", _dedupe(danger_flags))
            if action == "ALL_IN" and not short_stack_explicit:
                danger_flags.append("extreme_action_all_in")

            observations.append(
                {
                    "candidate_action": str(action),
                    "candidate_frequency": float(frequency),
                    "iterations": iterations,
                    "exploitability_last": exploitability,
                }
            )

        actions = [obs["candidate_action"] for obs in observations]
        consistency = actions.count(actions[0]) / len(actions)
        if len(set(actions)) != 1:
            danger_flags.append("dominant_action_unstable")
            return _quality(
                status="failed",
                solver_job_id=solver_job_id,
                candidate_action=None,
                stable_action=False,
                observations=observations,
                danger_flags=danger_flags,
                exclusion_reason="dominant_action_unstable",
            )

        return _quality(
            status="ok",
            solver_job_id=solver_job_id,
            candidate_action=actions[0],
            stable_action=True,
            observations=observations,
            danger_flags=danger_flags,
            exclusion_reason=None,
            consistency=consistency,
        )
    except Exception as exc:  # noqa: BLE001 - stable public boundary
        return _failed(None, f"{type(exc).__name__}:{exc}", [])


def _candidate_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    candidate = run.get("action_candidate")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return build_solver_action_candidate(run)


def _run_error(run: Mapping[str, Any]) -> str | None:
    solver_status = _solver_status(run)
    if solver_status == "timeout":
        return "timeout"
    if solver_status != "ok":
        return "solver_not_ok"
    if run.get("error"):
        return "solver_not_ok"
    return None


def _root_not_hero(run: Mapping[str, Any]) -> bool:
    if run.get("root_matches_hero") is False or run.get("root_player_role") == "villain":
        return True
    strategy = run.get("root_strategy")
    if isinstance(strategy, Mapping):
        if strategy.get("root_matches_hero") is False or strategy.get("root_player_role") == "villain":
            return True
    solver_result = run.get("solver_result")
    output = solver_result.get("output") if isinstance(solver_result, Mapping) else None
    if isinstance(output, Mapping):
        if output.get("root_matches_hero") is False or output.get("root_player_role") == "villain":
            return True
    return False


def _solver_status(run: Mapping[str, Any]) -> str | None:
    if run.get("solver_status") is not None:
        return str(run["solver_status"])
    solver_result = run.get("solver_result")
    if isinstance(solver_result, Mapping) and solver_result.get("status") is not None:
        return str(solver_result["status"])
    if run.get("status") is not None:
        return str(run["status"])
    return "ok"


def _solver_job_id(run: Mapping[str, Any]) -> str | None:
    if run.get("solver_job_id") is not None:
        return str(run["solver_job_id"])
    solver_result = run.get("solver_result")
    if isinstance(solver_result, Mapping) and solver_result.get("solver_job_id") is not None:
        return str(solver_result["solver_job_id"])
    return None


def _iterations(run: Mapping[str, Any]) -> int | None:
    for value in (
        run.get("iterations"),
        _mapping_get(run.get("quality"), "iterations"),
        _mapping_get(run.get("action_candidate"), "iterations"),
        _mapping_get(run.get("solver_result"), "iterations"),
        _mapping_get(_mapping_get(run.get("solver_result"), "output"), "iterations"),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _exploitability_last(run: Mapping[str, Any]) -> float | None:
    for value in (
        run.get("exploitability_last"),
        _mapping_get(run.get("quality"), "exploitability_last"),
    ):
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    output = _mapping_get(run.get("solver_result"), "output")
    history = _mapping_get(output, "exploitability_history")
    if isinstance(history, list) and history:
        return _float_or_none(history[-1])
    return None


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_candidate_reason(reason: str) -> str:
    if reason in {"root_not_hero", "solver_not_ok", "iterations_too_low"}:
        return reason
    if reason in {"dominant_action_too_weak", "frequency_too_low"}:
        return "frequency_too_low"
    return "candidate_not_available"


def _flag_for_reason(reason: str) -> str:
    mapping = {
        "frequency_too_low": "frequency_too_low",
        "iterations_too_low": "iterations_too_low",
        "timeout": "timeout",
        "root_not_hero": "root_not_hero",
    }
    return mapping.get(reason, reason)


def _failed(solver_job_id: str | None, reason: str, danger_flags: list[str]) -> dict[str, Any]:
    return _quality(
        status="failed",
        solver_job_id=solver_job_id,
        candidate_action=None,
        stable_action=False,
        observations=[],
        danger_flags=danger_flags,
        exclusion_reason=reason,
    )


def _quality(
    *,
    status: str,
    solver_job_id: str | None,
    candidate_action: str | None,
    stable_action: bool,
    observations: list[dict[str, Any]],
    danger_flags: list[str],
    exclusion_reason: str | None,
    consistency: float | None = None,
) -> dict[str, Any]:
    frequencies = [float(obs["candidate_frequency"]) for obs in observations]
    if consistency is None:
        consistency = 0.0 if not observations else 1.0
    return {
        "status": status,
        "solver_job_id": solver_job_id,
        "candidate_action": candidate_action,
        "stable_action": bool(stable_action),
        "dominant_frequency_avg": round(sum(frequencies) / len(frequencies), 6) if frequencies else 0.0,
        "dominant_frequency_min": round(min(frequencies), 6) if frequencies else 0.0,
        "dominant_action_consistency": round(float(consistency), 6),
        "danger_flags": _dedupe(danger_flags),
        "is_training_label": False,
        "label_quality": LABEL_QUALITY,
        "exclusion_reason": exclusion_reason,
        "warnings": [],
        "run_observations": observations,
        "recommendation": "usable_for_candidate_analysis" if status == "ok" else "not_usable_for_candidate_analysis",
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
