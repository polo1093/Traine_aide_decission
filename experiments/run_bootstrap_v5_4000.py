"""Generate, train, and compare the offline dist-aligned bootstrap v5_4000 run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.export_candidate_dataset import export_candidate_dataset
from experiments.dist_schema_alignment import DEFAULT_DIST_DIR, flatten_schema_paths, infer_json_schema, load_dist_references
from experiments.generate_bootstrap_solver_candidates import build_solver_candidate_plan, build_solver_candidate_report, run_one_solver_candidate_row, write_records
from experiments.train_bootstrap_v5 import dist_sample_prediction_report, train_bootstrap_v5
from datasets.export_candidate_dataset import export_dist_aligned_candidate_csv


DEFAULT_OUTPUT_DIR = Path("outputs/readiness/bootstrap_candidate_dataset_v5_4000")
DEFAULT_MODEL_DIR = Path("outputs/readiness/bootstrap_model_v5_large")
DEFAULT_BASELINE_MODEL_DIR = Path("outputs/readiness/bootstrap_model_v5")
DEFAULT_DIST_SAMPLE = Path("dist/ml_dataset_export/example_training_dataset.jsonl")


def run_bootstrap_v5_4000(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    baseline_model_dir: str | Path = DEFAULT_BASELINE_MODEL_DIR,
    target_solves: int = 4000,
    seed: int = 15400,
    timeout_s: float = 5.0,
    backend: str = "rust",
    min_dominant_frequency: float = 0.70,
    min_usable_rows: int = 4000,
    class_floor: int = 400,
    min_training_rows: int = 100,
    random_seed: int = 17,
    dist_dir: str | Path = DEFAULT_DIST_DIR,
    dist_sample: str | Path = DEFAULT_DIST_SAMPLE,
) -> dict[str, Any]:
    out = Path(output_dir)
    model_out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    sensitivity_path = out / "candidate_sensitivity_results.jsonl"
    raw_jsonl = out / "raw_candidates.jsonl"
    raw_csv = out / "raw_candidates.csv"
    candidates_csv = out / "candidates.csv"
    dataset_report_path = out / "dataset_report.json"
    comparison_path = model_out / "comparison_v5_vs_v5_4000.json"

    started = time.perf_counter()
    records = []
    with sensitivity_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in build_solver_candidate_plan(target_solves=target_solves, seed=seed):
            record = normalize_v5_record(run_one_solver_candidate_row(row, timeout_s=timeout_s, backend=backend))
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    write_records(sensitivity_path, records)
    raw_export_summary = export_candidate_dataset(
        [sensitivity_path],
        output_jsonl=raw_jsonl,
        output_csv=raw_csv,
        include_weak_rules=True,
        min_usable_rows=min_usable_rows,
        class_floor=class_floor,
        min_dominant_frequency=min_dominant_frequency,
        balance_weak_rules=True,
    )
    raw_report = build_solver_candidate_report(
        records,
        raw_export_summary,
        stage_name="v5_4000",
        requested_solves=target_solves,
        output_dir=out,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )

    dist_flat_columns = dist_reference_flat_columns(Path(dist_dir))
    dist_export_summary = export_dist_aligned_candidate_csv(
        raw_csv,
        candidates_csv,
        fieldnames=dist_flat_columns,
    )
    report_outputs = {
        key: value
        for key, value in raw_report["outputs"].items()
        if key != "dataset_report_solver"
    }
    dataset_report = {
        **raw_report,
        "schema": "dist_aligned_v5",
        "target_solves": target_solves,
        "raw_export_summary": raw_export_summary,
        "dist_export_summary": dist_export_summary,
        "outputs": {
            **report_outputs,
            "dataset_report": str(dataset_report_path),
            "raw_candidates_csv": str(raw_csv),
            "raw_candidates_jsonl": str(raw_jsonl),
            "candidates_csv": str(candidates_csv),
        },
    }
    write_json(dataset_report, dataset_report_path)
    if dataset_report["critical_warnings"]:
        return {
            "status": "failed",
            "dataset_report": dataset_report,
            "training": None,
            "comparison": None,
            "dist_sample_prediction": None,
        }

    training = train_bootstrap_v5(
        input_csv=candidates_csv,
        output_dir=model_out,
        min_rows=min_training_rows,
        random_seed=random_seed,
    )
    dist_prediction = dist_sample_prediction_report(model_dir=model_out, dist_sample=dist_sample)
    comparison = compare_reports(
        baseline_model_dir=Path(baseline_model_dir),
        candidate_model_dir=model_out,
        candidate_report=dataset_report,
        candidate_training=training,
        dist_prediction=dist_prediction,
    )
    write_json(comparison, comparison_path)
    write_json(training, model_out / "training_report.json")

    return {
        "status": "ok",
        "dataset_report": dataset_report,
        "training": training,
        "comparison": comparison,
        "dist_sample_prediction": dist_prediction,
        "heatmaps": training.get("heatmaps", {}),
        "outputs": {
            "candidates_csv": str(candidates_csv),
            "dataset_report": str(dataset_report_path),
            "model": str(model_out / "model.joblib"),
            "training_report": str(model_out / "training_report.json"),
            "feature_contract": str(model_out / "feature_contract.json"),
            "preprocessing_schema": str(model_out / "preprocessing_schema.json"),
            "v5_input_numeric_by_output_heatmap": str(model_out / "v5_input_numeric_by_output_heatmap.svg"),
            "v5_input_category_by_output_heatmap": str(model_out / "v5_input_category_by_output_heatmap.svg"),
            "comparison_v5_vs_v5_4000": str(comparison_path),
        },
    }


def normalize_v5_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("scenario", "solver_job_id"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.replace("solver_", "v5_4000_")
    return normalized


def dist_reference_flat_columns(dist_dir: Path) -> list[str]:
    references = load_dist_references(dist_dir)
    schema = infer_json_schema([record for _, record in references])
    return sorted(flatten_schema_paths(schema))


def compare_reports(
    *,
    baseline_model_dir: Path,
    candidate_model_dir: Path,
    candidate_report: dict[str, Any],
    candidate_training: dict[str, Any],
    dist_prediction: dict[str, Any],
) -> dict[str, Any]:
    baseline_training = load_first_existing_json(
        baseline_model_dir / "training_report_v5.json",
        baseline_model_dir / "training_report.json",
        baseline_model_dir / "evaluation_report.json",
    )
    return {
        "status": "ok",
        "baseline_v5": model_summary(baseline_training),
        "v5_4000": model_summary(candidate_training),
        "dataset_v5_4000": {
            "root_player_not_hero_errors": candidate_report.get("root_player_not_hero_errors"),
            "critical_warnings": candidate_report.get("critical_warnings"),
            "class_distribution": candidate_report.get("class_distribution"),
            "source_distribution": candidate_report.get("label_source_distribution"),
            "usable": candidate_report.get("usable_rows"),
            "excluded": candidate_report.get("rejected_spots"),
            "total_spots_generated": candidate_report.get("total_spots_generated"),
        },
        "dist_sample_prediction": dist_prediction,
        "baseline_model_dir": str(baseline_model_dir),
        "candidate_model_dir": str(candidate_model_dir),
        "baseline_v5_not_overwritten": True,
    }


def model_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_size": report.get("dataset_size") or {
            "total": report.get("rows_total"),
            "usable": report.get("rows_used"),
            "excluded": report.get("rows_excluded"),
        },
        "label_distribution": report.get("label_distribution"),
        "source_distribution": report.get("label_source_distribution") or report.get("sources"),
        "confusion_matrix": report.get("confusion_matrix"),
        "classification_report": report.get("classification_report"),
        "leakage_columns_excluded": report.get("leakage_columns_excluded"),
        "leakage_used_by_model": report.get("leakage_columns_used_by_model", []),
        "selected_model": report.get("selected_model"),
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro_f1"),
    }


def load_first_existing_json(*paths: Path) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"missing_baseline_report:{paths[0].parent}")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--baseline-model-dir", default=str(DEFAULT_BASELINE_MODEL_DIR))
    parser.add_argument("--target-solves", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=15400)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--backend", choices=["rust", "python"], default="rust")
    parser.add_argument("--min-dominant-frequency", type=float, default=0.70)
    parser.add_argument("--min-usable-rows", type=int, default=4000)
    parser.add_argument("--class-floor", type=int, default=400)
    parser.add_argument("--min-training-rows", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST_DIR))
    parser.add_argument("--dist-sample", default=str(DEFAULT_DIST_SAMPLE))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_bootstrap_v5_4000(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        baseline_model_dir=args.baseline_model_dir,
        target_solves=args.target_solves,
        seed=args.seed,
        timeout_s=args.timeout_s,
        backend=args.backend,
        min_dominant_frequency=args.min_dominant_frequency,
        min_usable_rows=args.min_usable_rows,
        class_floor=args.class_floor,
        min_training_rows=args.min_training_rows,
        random_seed=args.random_seed,
        dist_dir=args.dist_dir,
        dist_sample=args.dist_sample,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
