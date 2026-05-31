"""V2 full merged oracle preflop/postflop training and final report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.merged_oracle_preflop_postflop_v1 import run_merged_oracle_preflop_postflop_v1


DEFAULT_DATA_DIR = Path("outputs/readiness/merged_oracle_3intent_v2_full")
DEFAULT_OUTPUT_ROOT = Path("outputs/readiness")
DEFAULT_PREFLOP_DIR = Path("outputs/readiness/merged_oracle_preflop_model_v2_full")
DEFAULT_POSTFLOP_DIR = Path("outputs/readiness/merged_oracle_postflop_model_v2_full")
DEFAULT_FINAL_REPORT = Path("outputs/readiness/merged_oracle_v2_full_final_report.md")


def run_v2_full_training(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    preflop_output_dir: str | Path = DEFAULT_PREFLOP_DIR,
    postflop_output_dir: str | Path = DEFAULT_POSTFLOP_DIR,
    final_report_path: str | Path = DEFAULT_FINAL_REPORT,
    force: bool = False,
    random_seed: int = 17,
) -> dict[str, Any]:
    final_path = Path(final_report_path)
    result = run_merged_oracle_preflop_postflop_v1(
        data_dir=data_dir,
        output_root=output_root,
        preflop_output_dir=preflop_output_dir,
        postflop_output_dir=postflop_output_dir,
        comparison_path=final_path,
        force=force,
        random_seed=random_seed,
    )
    merge_report = read_json(Path(data_dir) / "merge_report.json")
    preflop_report = read_json(Path(preflop_output_dir) / "training_report.json")
    postflop_report = read_json(Path(postflop_output_dir) / "training_report.json")
    final_path.write_text(render_final_report(merge_report, preflop_report, postflop_report), encoding="utf-8")
    result["final_report"] = str(final_path)
    return result


def render_final_report(merge: dict[str, Any], preflop: dict[str, Any], postflop: dict[str, Any]) -> str:
    baseline = read_json(Path("outputs/readiness/pokerbench_oracle_3intent_v1/training_report.json"))
    lines = [
        "# Merged Oracle V2 Full Final Report",
        "",
        "## Dataset Composition",
        "",
        f"- total rows: `{sum((merge.get('source_rows') or {}).values())}`",
        f"- preflop rows: `{(merge.get('stage_group_distribution') or {}).get('PREFLOP')}`",
        f"- postflop rows: `{(merge.get('stage_group_distribution') or {}).get('POSTFLOP')}`",
        f"- unknown rows excluded: `{merge.get('unknown_stage_rows_excluded')}`",
        "",
        "### Rows By Source",
        "",
        json_block(merge.get("source_rows", {})),
        "",
        "### Rows By Stage",
        "",
        json_block(merge.get("stage_group_distribution", {})),
        "",
        "### Rows By Source And Stage",
        "",
        json_block(merge.get("rows_by_stage_and_source", {})),
        "",
        "### Label Distribution By Source",
        "",
        json_block(merge.get("label_distribution_by_source", {})),
        "",
        "### Label Distribution By Stage",
        "",
        json_block(merge.get("label_distribution_by_stage", {})),
        "",
        "## Model Comparison",
        "",
        "| model | rows | accuracy | macro_f1 | weighted_f1 | recall_NO_INVEST | recall_CALL | recall_RAISE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline:
        lines.append(
            f"| PokerBench 3intent baseline | {baseline.get('rows_usable')} | {baseline.get('accuracy')} | {baseline.get('macro_f1')} | {baseline.get('weighted_f1')} | "
            f"{recall_for(baseline, 'NO_INVEST')} | {recall_for(baseline, 'CALL')} | {recall_for(baseline, 'RAISE')} |"
        )
    for name, report in (("V2 preflop", preflop), ("V2 postflop", postflop)):
        rows = int(report.get("rows_train", 0)) + int(report.get("rows_validation", 0)) + int(report.get("rows_test", 0))
        lines.append(
            f"| {name} | {rows} | {report.get('accuracy')} | {report.get('macro_f1')} | {report.get('weighted_f1')} | "
            f"{report.get('recall_NO_INVEST')} | {report.get('recall_CALL')} | {report.get('recall_RAISE')} |"
        )
    lines.extend(
        [
            "",
            "## Source-Level Performance",
            "",
            "### Preflop",
            "",
            json_block(preflop.get("performance_by_source_dataset", {})),
            "",
            "### Postflop",
            "",
            json_block(postflop.get("performance_by_source_dataset", {})),
            "",
            "## Warnings",
            "",
            "- Domain shift is expected between PokerBench, poker-gto-100k, and gtow-llama-sft-v3.",
            "- External source license and GTO Wizard terms must be verified before redistribution or commercial use.",
            "- PHH/ACPC is excluded from supervised training because it is a scaffold/expert history source, not a direct oracle-label source.",
            "- Raw prompt/instruction/output/chosen/rejected text is excluded from model inputs.",
            "- `source_dataset` is excluded from model features and used only for grouped evaluation.",
            "",
            "## Final Recommendation",
            "",
            "- V2 improves coverage by adding external HU/GTO-style data, but source imbalance remains dominated by PokerBench unless full external exports are much larger.",
            "- The preflop model is the stronger offline candidate if source-level performance is stable.",
            "- The postflop model remains weaker and likely needs card, board texture, equity, range, and sizing features beyond the 20-feature contract.",
            "- For V3, compare source weighting/downsampling and consider excluding or separately routing any source that degrades source-level recall.",
            "",
        ]
    )
    return "\n".join(lines)


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"


def recall_for(report: dict[str, Any], label: str) -> float:
    return round(float(report.get("classification_report", {}).get(label, {}).get("recall", 0.0)), 6)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--preflop-output-dir", default=str(DEFAULT_PREFLOP_DIR))
    parser.add_argument("--postflop-output-dir", default=str(DEFAULT_POSTFLOP_DIR))
    parser.add_argument("--final-report", default=str(DEFAULT_FINAL_REPORT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_v2_full_training(
        data_dir=args.data_dir,
        output_root=args.output_root,
        preflop_output_dir=args.preflop_output_dir,
        postflop_output_dir=args.postflop_output_dir,
        final_report_path=args.final_report,
        force=args.force,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
