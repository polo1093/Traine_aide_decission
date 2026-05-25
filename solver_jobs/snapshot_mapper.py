"""Map exported ML snapshots into bounded solver jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solver_jobs.job_builder import build_solver_job


SNAPSHOT_SCHEMA_VERSION = "ml_dataset_v1"
DEFAULT_STACK = 1000.0
DEFAULT_BET_SIZES = [0.33]
DEFAULT_ITERATIONS = 25
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_BACKEND = "rust"
DEFAULT_LABEL_INTENT = "solver_smoke"
DEFAULT_UNITS = "chips"
DEFAULT_MIN_CONFIDENCE = 0.5


def map_snapshot_to_solver_job(
    snapshot: Mapping[str, Any],
    *,
    stack_fallback: float = DEFAULT_STACK,
    bet_sizes: list[float] | tuple[float, ...] = tuple(DEFAULT_BET_SIZES),
    iterations: int = DEFAULT_ITERATIONS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: str = DEFAULT_BACKEND,
    label_intent: str = DEFAULT_LABEL_INTENT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Map one exported snapshot to ``solver_job_v1`` without leaking exceptions."""

    warnings: list[str] = []
    source_snapshot_id = None
    try:
        if not isinstance(snapshot, Mapping):
            return _result(None, None, "snapshot_must_be_mapping", warnings)

        source_snapshot_id = snapshot.get("snapshot_id")
        if not source_snapshot_id:
            return _result(None, None, "snapshot_id_missing", warnings)

        if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            return _result(source_snapshot_id, None, "unsupported_snapshot_schema_version", warnings)

        quality_flags = snapshot.get("quality_flags") or {}
        if not isinstance(quality_flags, Mapping):
            return _result(source_snapshot_id, None, "quality_flags_must_be_mapping", warnings)
        usable_for_training = quality_flags.get("usable_for_training") is True
        usable_for_solver = quality_flags.get("usable_for_solver") is True
        if not usable_for_training and not usable_for_solver:
            return _result(source_snapshot_id, None, "snapshot_not_usable_for_training", warnings)

        low_confidence = _first_low_confidence(snapshot.get("confidence"), min_confidence)
        if low_confidence is not None:
            return _result(source_snapshot_id, None, f"confidence_too_low:{low_confidence}", warnings)

        metadata = snapshot.get("metadata") or {}
        features = snapshot.get("features") or {}
        if not isinstance(metadata, Mapping):
            return _result(source_snapshot_id, None, "metadata_must_be_mapping", warnings)
        if not isinstance(features, Mapping):
            return _result(source_snapshot_id, None, "features_must_be_mapping", warnings)

        street = metadata.get("street") or features.get("street")
        if street is None:
            return _result(source_snapshot_id, None, "street_missing", warnings)
        street = str(street).upper()
        if street not in {"FLOP", "TURN", "RIVER"}:
            return _result(source_snapshot_id, None, f"unsupported_street:{street}", warnings)

        active_opponents = features.get("active_opponents")
        if active_opponents != 1:
            return _result(source_snapshot_id, None, "non_heads_up_snapshot", warnings)

        hero_cards = features.get("hero_cards")
        if hero_cards is None:
            return _result(source_snapshot_id, None, "hero_cards_missing", warnings)
        board_cards = features.get("board_cards")
        if board_cards is None:
            return _result(source_snapshot_id, None, "board_cards_missing", warnings)

        if features.get("villain_range") not in (None, ""):
            return _result(source_snapshot_id, None, "villain_range_not_supported", warnings)
        villain_hand = features.get("villain_hand")
        if villain_hand is None:
            return _result(source_snapshot_id, None, "villain_hand_missing", warnings)

        if "pot" not in features:
            return _result(source_snapshot_id, None, "pot_missing", warnings)

        decision_context_known = features.get("decision_context_known") is True
        if not decision_context_known:
            return _result(source_snapshot_id, None, "decision_context_unknown", warnings)
        if "to_call" not in features or features.get("to_call") is None:
            return _result(source_snapshot_id, None, "to_call_unknown", warnings)
        to_call = features["to_call"]

        stack = features.get("stack", features.get("effective_stack"))
        if stack is None:
            stack = stack_fallback
            warnings.append(f"stack_missing_defaulted_to_{stack_fallback:g}")

        units = features.get("units") or metadata.get("units") or DEFAULT_UNITS
        if "units" not in features and "units" not in metadata:
            warnings.append(f"units_missing_defaulted_to_{DEFAULT_UNITS}")

        source_type = metadata.get("source_type") or "ml_snapshot"
        iterations_value = features.get("solver_iterations", iterations)
        timeout_value = features.get("solver_timeout_s", timeout_s)
        build_result = build_solver_job(
            solver_job_id=f"solver_job_from_{source_snapshot_id}",
            source_snapshot_id=str(source_snapshot_id),
            source_type=str(source_type),
            units=str(units),
            street=street,
            hero_hand=hero_cards,
            villain_hand=villain_hand,
            villain_range=None,
            board=board_cards,
            pot=features["pot"],
            to_call=to_call,
            stack=stack,
            bet_sizes=bet_sizes,
            iterations=iterations_value,
            timeout_s=timeout_value,
            backend=backend,
            label_intent=label_intent,
        )
        if build_result["status"] == "failed":
            return _result(source_snapshot_id, None, build_result["error"], warnings)
        return _result(source_snapshot_id, build_result["job"], None, warnings)
    except Exception as exc:  # noqa: BLE001
        return _result(source_snapshot_id, None, _format_error(exc), warnings)


def _first_low_confidence(value: Any, min_confidence: float) -> str | None:
    if value in (None, {}):
        return None
    if not isinstance(value, Mapping):
        return None
    for key, raw in value.items():
        if isinstance(raw, Mapping):
            nested = _first_low_confidence(raw, min_confidence)
            if nested is not None:
                return f"{key}.{nested}"
            continue
        if isinstance(raw, (int, float)) and float(raw) < min_confidence:
            return str(key)
    return None


def _result(
    source_snapshot_id: Any,
    solver_job: dict[str, Any] | None,
    error: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": "failed" if error else "ok",
        "source_snapshot_id": None if source_snapshot_id is None else str(source_snapshot_id),
        "solver_job": solver_job,
        "error": error,
        "warnings": list(warnings),
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
