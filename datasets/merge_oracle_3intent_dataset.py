"""Merge PokerBench and external oracle-like sources into one 3-intent dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pokerbench_oracle_baseline_v1 import FEATURE_COLUMNS


DEFAULT_OUTPUT_DIR = Path("outputs/readiness/merged_oracle_3intent_v1")
DEFAULT_INPUTS = {
    "pokerbench": Path("outputs/readiness/pokerbench_oracle_dataset_v1/model_input.csv"),
    "poker_gto_100k": Path("outputs/readiness/poker_gto_100k_oracle_v1/model_input.csv"),
    "gtow_llama_sft_v3": Path("outputs/readiness/gtow_llama_sft_v3_oracle_v1/model_input.csv"),
}
EXTERNAL_FEATURE_FILES = {
    "poker_gto_100k": Path("outputs/readiness/poker_gto_100k_oracle_v1/model_features_20.csv"),
    "gtow_llama_sft_v3": Path("outputs/readiness/gtow_llama_sft_v3_oracle_v1/model_features_20.csv"),
}
EXTERNAL_AUDIT_FILES = {
    "poker_gto_100k": Path("outputs/readiness/poker_gto_100k_oracle_v1/audit_candidates.csv"),
    "gtow_llama_sft_v3": Path("outputs/readiness/gtow_llama_sft_v3_oracle_v1/audit_candidates.csv"),
}
RAW_TEXT_COLUMNS = {
    "raw_prompt",
    "raw_instruction",
    "raw_response",
    "raw_output",
    "raw_chosen",
    "raw_rejected",
    "raw_metadata_json",
    "raw_record_json",
}
FORBIDDEN_MODEL_COLUMNS = {
    "source_dataset",
    "source_row_id",
    "raw_action",
    "normalized_action_4class",
    "label_3intent",
    "bootstrap_label",
    *RAW_TEXT_COLUMNS,
}
LABELS = ("NO_INVEST", "CALL", "RAISE")
SPLITS = ("train", "validation", "test")
STAGES = ("PREFLOP", "POSTFLOP")


def merge_oracle_3intent_dataset(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size_per_source: int | None = None,
    force: bool = False,
    random_seed: int = 17,
    input_paths: dict[str, Path] | None = None,
    external_feature_paths: dict[str, Path] | None = None,
    external_audit_paths: dict[str, Path] | None = None,
    pokerbench_dir: str | Path | None = None,
    poker_gto_dir: str | Path | None = None,
    gtow_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    reset_output_dir(output_path, force=force)

    inputs = input_paths or build_source_input_paths(
        pokerbench_dir=pokerbench_dir,
        poker_gto_dir=poker_gto_dir,
        gtow_dir=gtow_dir,
    )
    feature_files = external_feature_paths or build_external_feature_paths(
        poker_gto_dir=poker_gto_dir,
        gtow_dir=gtow_dir,
    )
    audit_files = external_audit_paths or build_external_audit_paths(
        poker_gto_dir=poker_gto_dir,
        gtow_dir=gtow_dir,
    )

    frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    load_warnings: list[str] = []
    for source_key, input_path in inputs.items():
        if not Path(input_path).exists():
            load_warnings.append(f"missing_source:{source_key}:{input_path}")
            continue
        if source_key == "pokerbench":
            frame, audit = load_pokerbench(Path(input_path), sample_size_per_source=sample_size_per_source)
        else:
            frame, audit = load_external(
                source_key,
                Path(input_path),
                feature_files.get(source_key),
                audit_files.get(source_key),
                sample_size_per_source=sample_size_per_source,
            )
        frames.append(frame)
        audit_frames.append(audit)

    if not frames:
        raise FileNotFoundError("no_merge_sources_available")

    merged = pd.concat(frames, ignore_index=True)
    audit = pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else merged.copy()
    merged["stage_group"] = merged.apply(stage_group_for_row, axis=1)
    merged["split_unit"] = merged.apply(split_unit_for_row, axis=1)
    merged["split"] = assign_splits(merged, random_seed=random_seed)
    training_rows = merged[merged["stage_group"].isin(STAGES)].copy()
    unknown_rows = merged[merged["stage_group"] == "UNKNOWN"].copy()

    write_outputs(merged=merged, training_rows=training_rows, audit=audit, output_path=output_path)
    report = build_merge_report(
        merged=merged,
        training_rows=training_rows,
        unknown_rows=unknown_rows,
        load_warnings=load_warnings,
        output_path=output_path,
        sample_size_per_source=sample_size_per_source,
        random_seed=random_seed,
    )
    (output_path / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "feature_contract.json").write_text(json.dumps(build_feature_contract(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "dataset_card.md").write_text(render_dataset_card(report), encoding="utf-8")
    return report


def build_source_input_paths(
    *,
    pokerbench_dir: str | Path | None,
    poker_gto_dir: str | Path | None,
    gtow_dir: str | Path | None,
) -> dict[str, Path]:
    return {
        "pokerbench": Path(pokerbench_dir) / "model_input.csv" if pokerbench_dir else DEFAULT_INPUTS["pokerbench"],
        "poker_gto_100k": Path(poker_gto_dir) / "model_input.csv" if poker_gto_dir else DEFAULT_INPUTS["poker_gto_100k"],
        "gtow_llama_sft_v3": Path(gtow_dir) / "model_input.csv" if gtow_dir else DEFAULT_INPUTS["gtow_llama_sft_v3"],
    }


def build_external_feature_paths(*, poker_gto_dir: str | Path | None, gtow_dir: str | Path | None) -> dict[str, Path]:
    return {
        "poker_gto_100k": Path(poker_gto_dir) / "model_features_20.csv" if poker_gto_dir else EXTERNAL_FEATURE_FILES["poker_gto_100k"],
        "gtow_llama_sft_v3": Path(gtow_dir) / "model_features_20.csv" if gtow_dir else EXTERNAL_FEATURE_FILES["gtow_llama_sft_v3"],
    }


def build_external_audit_paths(*, poker_gto_dir: str | Path | None, gtow_dir: str | Path | None) -> dict[str, Path]:
    return {
        "poker_gto_100k": Path(poker_gto_dir) / "audit_candidates.csv" if poker_gto_dir else EXTERNAL_AUDIT_FILES["poker_gto_100k"],
        "gtow_llama_sft_v3": Path(gtow_dir) / "audit_candidates.csv" if gtow_dir else EXTERNAL_AUDIT_FILES["gtow_llama_sft_v3"],
    }


def reset_output_dir(output_path: Path, *, force: bool) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        if not force:
            raise FileExistsError(f"output_dir_exists_use_force:{output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def load_pokerbench(path: Path, *, sample_size_per_source: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if sample_size_per_source is not None:
        df = df.head(sample_size_per_source).copy()
    out = pd.DataFrame()
    for feature in FEATURE_COLUMNS:
        out[feature] = df.get(feature)
    out["label_3intent"] = df.get("bootstrap_label")
    out["source_dataset"] = "PokerBench"
    out["source_row_id"] = df.get("snapshot_id")
    out["hand_id"] = df.get("hand_id")
    out["source_split_original"] = df.get("split")
    out["raw_action"] = None
    out["normalized_action_4class"] = None
    out["source_type"] = "solver_oracle"
    return coerce_feature_frame(out), out.copy()


def load_external(
    source_key: str,
    model_input_path: Path,
    feature_path: Path | None,
    audit_path: Path | None,
    *,
    sample_size_per_source: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_input = pd.read_csv(model_input_path)
    if feature_path is not None and Path(feature_path).exists():
        features = pd.read_csv(feature_path)
    else:
        features = normalize_external_without_feature_file(model_input)
    if sample_size_per_source is not None:
        features = features.head(sample_size_per_source).copy()
        model_input = model_input.head(sample_size_per_source).copy()
    out = pd.DataFrame()
    for feature in FEATURE_COLUMNS:
        out[feature] = features.get(feature)
    out["label_3intent"] = features.get("label_3intent", model_input.get("label_3intent"))
    out["source_dataset"] = features.get("source_dataset", model_input.get("source_dataset", source_key))
    out["source_row_id"] = features.get("source_row_id", model_input.get("source_row_id"))
    out["hand_id"] = model_input.get("hand_id")
    out["source_split_original"] = model_input.get("source_split")
    out["raw_action"] = model_input.get("raw_action")
    out["normalized_action_4class"] = model_input.get("normalized_action_4class")
    out["source_type"] = "external_solver_like"

    if audit_path is not None and Path(audit_path).exists():
        audit = pd.read_csv(audit_path)
        if sample_size_per_source is not None:
            audit = audit.head(sample_size_per_source).copy()
    else:
        audit = model_input.copy()
    return coerce_feature_frame(out), audit


def normalize_external_without_feature_file(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    pot = pd.to_numeric(df.get("pot_size"), errors="coerce")
    stack = pd.to_numeric(df.get("effective_stack"), errors="coerce")
    board_count = df.get("board_cards", pd.Series([""] * len(df))).fillna("").astype(str).str.count(r"[2-9TJQKA][cdhs]")
    out["features.pot_bb"] = pot / 100.0
    out["features.to_call_bb"] = 0.0
    out["features.hero_stack_bb"] = stack / 100.0
    out["features.effective_stack_bb"] = stack / 100.0
    out["features.stack_to_pot_ratio"] = stack / pot
    out["features.to_call_pot_ratio"] = 0.0
    legal = df.get("legal_actions", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
    out["features.has_check"] = legal.str.contains("check").astype(float)
    out["features.has_fold"] = legal.str.contains("fold").astype(float)
    out["features.has_call"] = legal.str.contains("call").astype(float)
    out["features.has_raise"] = legal.str.contains("bet|raise|all-in|allin").astype(float)
    out["features.num_players"] = 2.0
    out["features.num_bets"] = 0.0
    out["features.board_card_count"] = board_count.astype(float)
    for feature in ("features.action_count", "features.prior_check_count", "features.prior_call_count", "features.prior_bet_raise_count", "features.prior_fold_count"):
        out[feature] = 0.0
    out["metadata.street"] = df.get("street")
    out["features.hero_position"] = df.get("hero_position")
    out["label_3intent"] = df.get("label_3intent")
    out["source_dataset"] = df.get("source_dataset")
    out["source_row_id"] = df.get("source_row_id")
    return out


def coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in FEATURE_COLUMNS:
        if feature not in out.columns:
            out[feature] = pd.NA
    for feature in numeric_features():
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    out["metadata.street"] = out["metadata.street"].fillna("UNKNOWN").astype(str).str.upper()
    out["features.hero_position"] = out["features.hero_position"].fillna("UNKNOWN").astype(str).str.upper()
    out["label_3intent"] = out["label_3intent"].fillna("").astype(str).str.upper()
    return out


def numeric_features() -> list[str]:
    return [feature for feature in FEATURE_COLUMNS if feature not in {"metadata.street", "features.hero_position"}]


def stage_group_for_row(row: pd.Series) -> str:
    street = str(row.get("metadata.street") or "").upper()
    board_count = pd.to_numeric(row.get("features.board_card_count"), errors="coerce")
    if street == "PREFLOP":
        return "PREFLOP"
    if street in {"FLOP", "TURN", "RIVER"}:
        return "POSTFLOP"
    if pd.notna(board_count) and float(board_count) == 0.0:
        return "PREFLOP"
    if pd.notna(board_count) and float(board_count) >= 3.0:
        return "POSTFLOP"
    return "UNKNOWN"


def split_unit_for_row(row: pd.Series) -> str:
    source = str(row.get("source_dataset") or "UNKNOWN")
    hand_id = row.get("hand_id")
    if pd.notna(hand_id) and str(hand_id).strip():
        return f"{source}:hand:{hand_id}"
    return f"{source}:row:{row.get('source_row_id')}"


def stable_hash(value: str, *, random_seed: int = 17) -> str:
    return hashlib.sha256(f"{random_seed}:{value}".encode("utf-8")).hexdigest()


def assign_splits(df: pd.DataFrame, *, random_seed: int) -> pd.Series:
    split_by_unit: dict[str, str] = {}
    for stage in STAGES:
        stage_units = sorted({str(unit) for unit in df.loc[df["stage_group"] == stage, "split_unit"].dropna()}, key=lambda item: stable_hash(item, random_seed=random_seed))
        n = len(stage_units)
        if n == 0:
            continue
        if n < 3:
            for unit in stage_units:
                split_by_unit[unit] = "train"
            continue
        test_n = max(1, round(n * 0.20))
        validation_n = max(1, round(n * 0.10))
        train_n = max(1, n - test_n - validation_n)
        for unit in stage_units[:train_n]:
            split_by_unit[unit] = "train"
        for unit in stage_units[train_n : train_n + validation_n]:
            split_by_unit[unit] = "validation"
        for unit in stage_units[train_n + validation_n :]:
            split_by_unit[unit] = "test"
    return df["split_unit"].map(split_by_unit).fillna("")


def write_outputs(*, merged: pd.DataFrame, training_rows: pd.DataFrame, audit: pd.DataFrame, output_path: Path) -> None:
    model_columns = ["source_dataset", "source_row_id", "hand_id", "stage_group", "split", *FEATURE_COLUMNS, "label_3intent"]
    merged[model_columns].to_csv(output_path / "model_input.csv", index=False)
    for stage in STAGES:
        stage_name = stage.lower()
        stage_rows = training_rows[training_rows["stage_group"] == stage].copy()
        stage_rows[model_columns].to_csv(output_path / f"model_input_{stage_name}.csv", index=False)
        for split in SPLITS:
            split_rows = stage_rows[stage_rows["split"] == split]
            split_rows[list(FEATURE_COLUMNS)].to_csv(output_path / f"X_{split}_{stage_name}.csv", index=False)
            split_rows[["source_dataset", "source_row_id", "label_3intent"]].to_csv(output_path / f"y_{split}_{stage_name}.csv", index=False)
    audit_out = audit.copy()
    audit_out["audit_note"] = audit_out.get("audit_note", "source audit row; not model input")
    audit_out.to_csv(output_path / "audit_candidates.csv", index=False)


def build_merge_report(
    *,
    merged: pd.DataFrame,
    training_rows: pd.DataFrame,
    unknown_rows: pd.DataFrame,
    load_warnings: list[str],
    output_path: Path,
    sample_size_per_source: int | None,
    random_seed: int,
) -> dict[str, Any]:
    warnings = list(load_warnings)
    for stage in STAGES:
        stage_rows = training_rows[training_rows["stage_group"] == stage]
        if len(stage_rows) < 50:
            warnings.append(f"small_{stage.lower()}_dataset:{len(stage_rows)}")
        if set(stage_rows["split"]) - {""} != set(SPLITS):
            warnings.append(f"missing_split_for_{stage.lower()}:{sorted(set(SPLITS) - set(stage_rows['split']))}")
    schema_version = "merged_oracle_3intent_v2_full" if "v2" in str(output_path).lower() else "merged_oracle_3intent_v1"
    return {
        "status": "ok",
        "schema_version": schema_version,
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "output_dir": str(output_path),
        "sample_size_per_source": sample_size_per_source,
        "random_seed": random_seed,
        "source_rows": count_series(merged["source_dataset"]),
        "stage_group_distribution": count_series(merged["stage_group"]),
        "unknown_stage_rows_excluded": int(len(unknown_rows)),
        "excluded_sources": {
            "PHH / Zenodo ACPC HUNL": "excluded: scaffold/expert hand-history source, not direct solver/GTO labels",
        },
        "label_distribution": count_series(training_rows["label_3intent"]),
        "label_distribution_by_stage": nested_counts(training_rows, "stage_group", "label_3intent"),
        "label_distribution_by_source": nested_counts(training_rows, "source_dataset", "label_3intent"),
        "rows_by_stage_and_source": nested_counts(training_rows, "stage_group", "source_dataset"),
        "split_distribution_by_stage": nested_counts(training_rows, "stage_group", "split"),
        "missing_feature_rates": missing_feature_rates(training_rows),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "label_3intent",
        "leakage_columns_used_by_x": [column for column in FEATURE_COLUMNS if column in FORBIDDEN_MODEL_COLUMNS],
        "raw_text_columns_excluded_from_x": sorted(RAW_TEXT_COLUMNS),
        "source_dataset_excluded_from_x": True,
        "warnings": sorted(set(warnings)),
        "recommended_next_step": "Train separate preflop/postflop models and inspect source-level performance before any source merge is trusted.",
    }


def count_series(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(str(value) for value in series.fillna("")).items())}


def nested_counts(df: pd.DataFrame, outer: str, inner: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for outer_value, group in df.groupby(outer, dropna=False):
        result[str(outer_value)] = count_series(group[inner])
    return result


def missing_feature_rates(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {feature: 1.0 for feature in FEATURE_COLUMNS}
    rates = {}
    for feature in FEATURE_COLUMNS:
        missing = df[feature].isna() | (df[feature].astype(str).str.strip() == "")
        rates[feature] = round(float(missing.mean()), 6)
    return rates


def build_feature_contract() -> dict[str, Any]:
    return {
        "schema": "merged_oracle_3intent_v1",
        "features_model_used": list(FEATURE_COLUMNS),
        "numeric_features": numeric_features(),
        "categorical_features": ["metadata.street", "features.hero_position"],
        "target_column": "label_3intent",
        "allowed_labels": list(LABELS),
        "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        "source_dataset_kept_for_audit_and_grouped_eval": True,
        "phh_acpc_included": False,
    }


def render_dataset_card(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Merged Oracle 3-Intent Dataset V1",
            "",
            "Sources included: PokerBench, jevonmao/poker-gto-100k, jevonmao/gtow-llama-sft-v3.",
            "PHH / ACPC is explicitly excluded because it is not a direct solver/GTO label source in this repo.",
            "",
            "## Intended Use",
            "",
            "Offline supervised training for separate preflop and postflop 3-intent classifiers.",
            "",
            "## Labels",
            "",
            "CHECK/FOLD -> NO_INVEST, CALL -> CALL, BET/RAISE -> RAISE.",
            "",
            "## Rows",
            "",
            "```json",
            json.dumps(report["stage_group_distribution"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Leakage",
            "",
            "Raw prompt/output/action text and source identifiers are excluded from X_* files. Source IDs remain in y/audit files for grouped evaluation.",
            "",
            "## Limitations",
            "",
            "- Source/domain shift is expected between PokerBench and GTO-style HF datasets.",
            "- External compatibility features are best-effort normalized to the PokerBench 20-feature contract.",
            "- Missing features are reported instead of silently trusted.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-size-per-source", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--pokerbench-dir", default=None)
    parser.add_argument("--poker-gto-dir", default=None)
    parser.add_argument("--gtow-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = merge_oracle_3intent_dataset(
        output_dir=args.output_dir,
        sample_size_per_source=args.sample_size_per_source,
        force=args.force,
        random_seed=args.random_seed,
        pokerbench_dir=args.pokerbench_dir,
        poker_gto_dir=args.poker_gto_dir,
        gtow_dir=args.gtow_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
