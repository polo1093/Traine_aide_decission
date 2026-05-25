"""Parse PokerTH history text and write solver-ready snapshots as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokerth.history_parser import parse_pokerth_history
from pokerth.snapshot_builder import build_snapshot_from_hand_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Import simple PokerTH histories into solver-ready snapshots")
    parser.add_argument("history_file")
    parser.add_argument("--hero-name", default="polo")
    parser.add_argument("--street", default="RIVER", choices=("FLOP", "TURN", "RIVER"))
    parser.add_argument("--to-call", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = Path(args.history_file).read_text(encoding="utf-8")
    parsed = parse_pokerth_history(text, hero_name=args.hero_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for hand in parsed.get("hands", []):
        built = build_snapshot_from_hand_summary(hand, street=args.street, to_call=args.to_call)
        records.append(built)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(json.dumps({"parse": parsed, "records_written": len(records), "output": str(output_path)}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
