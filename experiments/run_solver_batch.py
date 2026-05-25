"""Run a tiny snapshot -> solver job -> solver result batch and write JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver_jobs.batch_runner import run_solver_batch, write_solver_batch_jsonl


def _snapshots() -> list[dict]:
    return [
        {
            "schema_version": "ml_dataset_v1",
            "snapshot_id": "snapshot_batch_flop",
            "metadata": {"street": "FLOP"},
            "features": {
                "hero_cards": ["Ah", "Kh"],
                "villain_hand": ["Qd", "Qc"],
                "board_cards": ["2h", "7h", "9d"],
                "pot": 100.0,
                "to_call": 20.0,
                "to_call_is_estimated": False,
                "decision_context_known": True,
                "stack": 1000.0,
                "active_opponents": 1,
                "hero_position": "BTN",
                "units": "chips",
            },
            "labels": {},
            "confidence": {"overall": 0.95},
            "quality_flags": {"usable_for_training": True},
        },
        {
            "schema_version": "ml_dataset_v1",
            "snapshot_id": "snapshot_batch_turn",
            "metadata": {"street": "TURN"},
            "features": {
                "hero_cards": ["Ah", "Kh"],
                "villain_hand": ["Qd", "Qc"],
                "board_cards": ["2h", "7h", "9d", "4c"],
                "pot": 140.0,
                "to_call": 0.0,
                "to_call_is_estimated": False,
                "decision_context_known": True,
                "stack": 1000.0,
                "active_opponents": 1,
                "hero_position": "BTN",
                "units": "chips",
            },
            "labels": {},
            "confidence": {"overall": 0.95},
            "quality_flags": {"usable_for_training": True},
        },
        {
            "schema_version": "ml_dataset_v1",
            "snapshot_id": "snapshot_batch_missing_villain",
            "metadata": {"street": "FLOP"},
            "features": {
                "hero_cards": ["Ah", "Kh"],
                "board_cards": ["2h", "7h", "9d"],
                "pot": 100.0,
                "to_call": 20.0,
                "to_call_is_estimated": False,
                "decision_context_known": True,
                "active_opponents": 1,
            },
            "labels": {},
            "confidence": {"overall": 0.95},
            "quality_flags": {"usable_for_training": True},
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny bounded solver batch")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    batch = run_solver_batch(_snapshots())
    write_result = write_solver_batch_jsonl(batch, args.output)
    print(json.dumps({"batch": batch, "write": write_result}, indent=2, ensure_ascii=False, default=str))
    return 0 if write_result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
