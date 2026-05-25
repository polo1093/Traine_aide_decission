"""Analyze solver_run_result JSONL files without creating ML labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_solver_run_results(paths: list[str | Path]) -> dict[str, Any]:
    """Return aggregate counters for one or more solver_run_result JSONL files."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_rows(Path(path)))

    total = len(rows)
    solved = sum(1 for row in rows if _row_state(row) == "solved")
    skipped = sum(1 for row in rows if _row_state(row) == "skipped")
    timeouts = sum(1 for row in rows if _row_state(row) == "timeout")
    errors = sum(1 for row in rows if _row_state(row) == "error")
    durations = [_duration_ms(row) for row in rows]
    durations = [value for value in durations if value is not None]

    summary = {
        "total": total,
        "solved": solved,
        "skipped": skipped,
        "timeouts": timeouts,
        "errors": errors,
        "avg_duration_ms": _average(durations),
        "profiles": _group_counts(rows, _profile),
        "iterations": _group_counts(rows, _iterations),
        "recommendation": "",
    }
    summary["recommendation"] = _recommendation(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze solver_run_result JSONL files. This never creates labels or datasets."
    )
    parser.add_argument("paths", nargs="+", help="solver_run_result JSONL file(s) to analyze.")
    args = parser.parse_args()

    summary = analyze_solver_run_results(args.paths)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                text = line.strip()
                if not text:
                    rows.append(_invalid_row(path, line_index, "jsonl_empty_line"))
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    rows.append(_invalid_row(path, line_index, f"JSONDecodeError:{exc.msg}"))
                    continue
                if not isinstance(value, dict):
                    rows.append(_invalid_row(path, line_index, "jsonl_record_must_be_object"))
                    continue
                rows.append(value)
    except OSError as exc:
        rows.append(_invalid_row(path, None, f"{type(exc).__name__}:{exc}"))
    return rows


def _invalid_row(path: Path, line_index: int | None, error: str) -> dict[str, Any]:
    return {
        "record_type": "solver_run_result",
        "solver_status": "failed",
        "solver_job": None,
        "solver_result": None,
        "quality": {
            "iterations": None,
            "exploitability_last": None,
            "is_label_candidate": False,
            "exclusion_reason": "jsonl_invalid",
        },
        "error": error,
        "warnings": [f"path:{path}", f"line_index:{line_index}" if line_index is not None else "line_index:unknown"],
        "_invalid_jsonl": True,
    }


def _row_state(row: dict[str, Any]) -> str:
    if row.get("_invalid_jsonl"):
        return "error"
    status = str(row.get("solver_status") or "").lower()
    if status == "ok":
        return "solved"
    if status == "skipped":
        return "skipped"
    if status == "timeout":
        return "timeout"
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    error = str(row.get("error") or "").lower()
    warnings = " ".join(str(item).lower() for item in row.get("warnings") or [])
    if quality.get("exclusion_reason") == "timeout" or "timeout" in error or "timeout" in warnings:
        return "timeout"
    return "error"


def _profile(row: dict[str, Any]) -> str:
    job = row.get("solver_job")
    if isinstance(job, dict) and job.get("generation_profile") is not None:
        return str(job["generation_profile"])
    return "unknown"


def _iterations(row: dict[str, Any]) -> str:
    quality = row.get("quality")
    if isinstance(quality, dict) and quality.get("iterations") is not None:
        return str(quality["iterations"])
    job = row.get("solver_job")
    if isinstance(job, dict) and job.get("iterations") is not None:
        return str(job["iterations"])
    result = row.get("solver_result")
    if isinstance(result, dict) and result.get("input") and isinstance(result["input"], dict):
        value = result["input"].get("iterations")
        if value is not None:
            return str(value)
    return "unknown"


def _duration_ms(row: dict[str, Any]) -> float | None:
    for value in (
        row.get("duration_ms"),
        _nested(row, "solver_result", "duration_ms"),
    ):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(row: dict[str, Any], outer: str, inner: str) -> Any:
    value = row.get(outer)
    if isinstance(value, dict):
        return value.get(inner)
    return None


def _group_counts(rows: list[dict[str, Any]], key_func) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_func(row), []).append(row)

    return {key: _counts(group) for key, group in sorted(groups.items())}


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [_duration_ms(row) for row in rows]
    durations = [value for value in durations if value is not None]
    return {
        "total": len(rows),
        "solved": sum(1 for row in rows if _row_state(row) == "solved"),
        "skipped": sum(1 for row in rows if _row_state(row) == "skipped"),
        "timeouts": sum(1 for row in rows if _row_state(row) == "timeout"),
        "errors": sum(1 for row in rows if _row_state(row) == "error"),
        "avg_duration_ms": _average(durations),
    }


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _recommendation(summary: dict[str, Any]) -> str:
    total = int(summary["total"])
    if total == 0:
        return "no_solver_run_results: run a tiny eligible batch before changing scope."

    timeout_rate = summary["timeouts"] / total
    success_rate = summary["solved"] / total
    if timeout_rate > 0.30:
        return "reduce_profiles_or_iterations: timeout_rate_above_30_percent; keep runs tiny and avoid label generation."
    if success_rate > 0.80:
        return "stable_for_smoke_runs: success_rate_above_80_percent; keep is_label_candidate=false and do not create training_label."
    return "keep_controlled_smoke_scope: inspect skipped/errors before expanding; this is still not an ML dataset."


if __name__ == "__main__":
    raise SystemExit(main())
