"""End-to-end PokerTH text -> solver run JSONL pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pokerth.history_parser import parse_pokerth_history
from pokerth.snapshot_builder import build_snapshot_from_hand_summary
from solver_jobs.batch_runner import run_solver_batch, write_solver_batch_jsonl


def run_pokerth_solver_pipeline(
    *,
    text: str | None = None,
    path: str | Path | None = None,
    hero_name: str = "polo",
    max_hands: int = 10,
    street: str = "RIVER",
    to_call_by_street: dict[str, float] | None = None,
    iterations: int = 25,
    timeout_s: float = 5.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Parse PokerTH text, build strict snapshots, run solver batch, optionally write JSONL."""

    try:
        history_text = _read_input(text=text, path=path)
        if history_text is None:
            return _summary_result(
                status="failed",
                hands_total=0,
                hands_parsed=0,
                hands_rejected=0,
                snapshots_built=0,
                snapshots_rejected=0,
                jobs_mapped=0,
                jobs_solved=0,
                solver_failed=0,
                output_path=None,
                results=[_rejection_row("input", None, None, "input_missing", "input_missing")],
            )

        parsed = parse_pokerth_history(history_text, hero_name=hero_name)
        parse_rejections = list(parsed.get("rejections") or [])
        hands = list(parsed.get("hands") or [])[: max(0, int(max_hands))]
        street_name = str(street).upper()
        to_call = _to_call_for_street(to_call_by_street, street_name)

        results: list[dict[str, Any]] = []
        for rejection in parse_rejections:
            results.append(
                _rejection_row(
                    "parse",
                    rejection.get("game_id"),
                    rejection.get("hand_id"),
                    rejection.get("rejection_reason") or "parse_failed",
                    rejection.get("error") or rejection.get("rejection_reason") or "parse_failed",
                )
            )

        snapshots: list[dict[str, Any]] = []
        snapshot_rejections = 0
        for hand in hands:
            built = build_snapshot_from_hand_summary(hand, street=street_name, to_call=to_call)
            if built["status"] == "ok":
                snapshot = built["snapshot"]
                snapshot["features"]["solver_iterations"] = int(iterations)
                snapshot["features"]["solver_timeout_s"] = float(timeout_s)
                snapshots.append(snapshot)
                results.append(
                    {
                        "stage": "snapshot",
                        "status": "ok",
                        "game_id": built.get("game_id"),
                        "hand_id": built.get("hand_id"),
                        "source_snapshot_id": snapshot.get("snapshot_id"),
                        "rejection_reason": None,
                        "error": None,
                        "snapshot": snapshot,
                        "batch_result": None,
                    }
                )
            else:
                snapshot_rejections += 1
                results.append(
                    _rejection_row(
                        "snapshot",
                        built.get("game_id"),
                        built.get("hand_id"),
                        built.get("rejection_reason") or "snapshot_rejected",
                        built.get("error") or built.get("rejection_reason") or "snapshot_rejected",
                    )
                )

        batch = run_solver_batch(snapshots) if snapshots else _empty_batch()
        for row in batch.get("results", []):
            results.append(
                {
                    "stage": "solver",
                    "status": row.get("solver_status"),
                    "game_id": _job_meta(row, "game_id"),
                    "hand_id": _job_meta(row, "hand_id"),
                    "source_snapshot_id": row.get("source_snapshot_id"),
                    "rejection_reason": None if row.get("error") is None else row.get("error"),
                    "error": row.get("error"),
                    "snapshot": None,
                    "batch_result": row,
                }
            )

        write_path = None
        if output_path is not None:
            write_result = write_solver_batch_jsonl(batch, output_path)
            write_path = write_result.get("output_path") if write_result.get("status") == "ok" else None
            if write_result.get("status") != "ok":
                results.append(_rejection_row("write", None, None, "jsonl_write_failed", write_result.get("error")))

        hands_total = len(parse_rejections) + len(hands)
        hands_parsed = len(hands)
        hands_rejected = len(parse_rejections)
        snapshots_built = len(snapshots)
        jobs_mapped = int(batch.get("mapped", 0))
        jobs_solved = int(batch.get("solved", 0))
        solver_failed = int(batch.get("solver_failed", 0))
        status = _pipeline_status(
            jobs_solved=jobs_solved,
            failures=hands_rejected + snapshot_rejections + solver_failed,
        )
        return _summary_result(
            status=status,
            hands_total=hands_total,
            hands_parsed=hands_parsed,
            hands_rejected=hands_rejected,
            snapshots_built=snapshots_built,
            snapshots_rejected=snapshot_rejections,
            jobs_mapped=jobs_mapped,
            jobs_solved=jobs_solved,
            solver_failed=solver_failed,
            output_path=write_path,
            results=results,
        )
    except Exception as exc:  # noqa: BLE001
        return _summary_result(
            status="failed",
            hands_total=0,
            hands_parsed=0,
            hands_rejected=0,
            snapshots_built=0,
            snapshots_rejected=0,
            jobs_mapped=0,
            jobs_solved=0,
            solver_failed=0,
            output_path=None,
            results=[_rejection_row("pipeline", None, None, "pipeline_exception", _format_error(exc))],
        )


def _read_input(*, text: str | None, path: str | Path | None) -> str | None:
    if text is not None:
        return text
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _to_call_for_street(to_call_by_street: dict[str, float] | None, street: str) -> float | None:
    if not to_call_by_street:
        return None
    for key, value in to_call_by_street.items():
        if str(key).upper() == street:
            return float(value)
    return None


def _empty_batch() -> dict[str, Any]:
    return {
        "status": "failed",
        "total": 0,
        "mapped": 0,
        "solved": 0,
        "mapping_failed": 0,
        "solver_failed": 0,
        "failed_total": 0,
        "results": [],
    }


def _pipeline_status(*, jobs_solved: int, failures: int) -> str:
    if jobs_solved <= 0:
        return "failed"
    if failures > 0:
        return "partial"
    return "ok"


def _summary_result(
    *,
    status: str,
    hands_total: int,
    hands_parsed: int,
    hands_rejected: int,
    snapshots_built: int,
    snapshots_rejected: int,
    jobs_mapped: int,
    jobs_solved: int,
    solver_failed: int,
    output_path: str | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "hands_total": hands_total,
        "hands_parsed": hands_parsed,
        "hands_rejected": hands_rejected,
        "snapshots_built": snapshots_built,
        "snapshots_rejected": snapshots_rejected,
        "jobs_mapped": jobs_mapped,
        "jobs_solved": jobs_solved,
        "solver_failed": solver_failed,
        "output_path": output_path,
        "results": results,
    }


def _rejection_row(
    stage: str,
    game_id: Any,
    hand_id: Any,
    rejection_reason: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "failed",
        "game_id": game_id,
        "hand_id": hand_id,
        "source_snapshot_id": None,
        "rejection_reason": rejection_reason,
        "error": error or rejection_reason,
        "snapshot": None,
        "batch_result": None,
    }


def _job_meta(row: dict[str, Any], key: str) -> Any:
    job = row.get("solver_job") or {}
    metadata = job.get("input", {}).get("metadata", {}) if "input" in job else {}
    return (job.get("metadata") or metadata or {}).get(key)


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__
