"""Unified CLI for external poker oracle dataset adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external_oracle_sources.hf_gtow_llama_sft_v3 import export_gtow_llama_sft_v3  # noqa: E402
from datasets.external_oracle_sources.hf_poker_gto_100k import export_poker_gto_100k  # noqa: E402
from datasets.external_oracle_sources.phh_acpc_scaffold import write_phh_scaffold  # noqa: E402


DEFAULT_OUTPUTS = {
    "poker_gto_100k": "outputs/readiness/poker_gto_100k_oracle_v1",
    "gtow_llama_sft_v3": "outputs/readiness/gtow_llama_sft_v3_oracle_v1",
    "phh_acpc_scaffold": "outputs/readiness/phh_acpc_scaffold_v1",
}


def export_external_oracle_dataset(
    *,
    source: str,
    output_dir: str | Path | None = None,
    sample_size: int | None = None,
    no_download: bool = False,
    force: bool = False,
    label_mode: str = "3intent",
) -> dict[str, Any]:
    if label_mode != "3intent":
        raise ValueError(f"unsupported_label_mode:{label_mode}")
    target_dir = Path(output_dir or DEFAULT_OUTPUTS[source])
    if source == "poker_gto_100k":
        return export_poker_gto_100k(
            output_dir=target_dir,
            sample_size=sample_size,
            no_download=no_download,
            force=force,
        )
    if source == "gtow_llama_sft_v3":
        return export_gtow_llama_sft_v3(
            output_dir=target_dir,
            sample_size=sample_size,
            no_download=no_download,
            force=force,
        )
    if source == "phh_acpc_scaffold":
        return write_phh_scaffold(output_dir=target_dir, force=force)
    raise ValueError(f"unsupported_source:{source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(DEFAULT_OUTPUTS), required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--label-mode", choices=["3intent"], default="3intent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = export_external_oracle_dataset(
            source=args.source,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            no_download=args.no_download,
            force=args.force,
            label_mode=args.label_mode,
        )
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "hint": "For Hugging Face exports, remove --no-download. This adapter uses the Hugging Face Dataset Viewer API and does not train a model.",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
