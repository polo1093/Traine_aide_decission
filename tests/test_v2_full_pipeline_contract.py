from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.pokerbench_oracle_baseline_v1 import FEATURE_COLUMNS


def make_feature_rows(count: int, *, prefix: str, street: str, board_count: float) -> pd.DataFrame:
    labels = ["NO_INVEST", "CALL", "RAISE"]
    rows = []
    for index in range(count):
        row = {
            "snapshot_id": f"{prefix}:{index}",
            "source_row_id": f"{prefix}:{index}",
            "source_dataset": prefix,
            "split": "train",
            "hand_id": f"{prefix}_hand_{index}",
            "bootstrap_label": labels[index % 3],
            "label_3intent": labels[index % 3],
        }
        for feature in FEATURE_COLUMNS:
            if feature == "metadata.street":
                row[feature] = street
            elif feature == "features.hero_position":
                row[feature] = "BB"
            elif feature == "features.board_card_count":
                row[feature] = board_count
            elif feature.startswith("features.has_"):
                row[feature] = 1.0
            else:
                row[feature] = float(index % 9 + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_source_dirs(root: Path) -> tuple[Path, Path, Path]:
    pokerbench = root / "pokerbench"
    poker_gto = root / "poker_gto"
    gtow = root / "gtow"
    for path in (pokerbench, poker_gto, gtow):
        path.mkdir(parents=True)

    pb = pd.concat(
        [
            make_feature_rows(45, prefix="PokerBench", street="PREFLOP", board_count=0.0),
            make_feature_rows(45, prefix="PokerBench", street="FLOP", board_count=3.0),
        ],
        ignore_index=True,
    )
    pb.to_csv(pokerbench / "model_input.csv", index=False)

    gto = make_feature_rows(45, prefix="jevonmao/poker-gto-100k", street="TURN", board_count=4.0)
    gto.to_csv(poker_gto / "model_features_20.csv", index=False)
    gto[["source_dataset", "source_row_id", "label_3intent"]].to_csv(poker_gto / "model_input.csv", index=False)
    gto.to_csv(poker_gto / "audit_candidates.csv", index=False)

    gtow_df = pd.concat(
        [
            make_feature_rows(45, prefix="jevonmao/gtow-llama-sft-v3", street="PREFLOP", board_count=0.0),
            make_feature_rows(45, prefix="jevonmao/gtow-llama-sft-v3", street="RIVER", board_count=5.0),
        ],
        ignore_index=True,
    )
    gtow_df.to_csv(gtow / "model_features_20.csv", index=False)
    gtow_df[["source_dataset", "source_row_id", "label_3intent"]].to_csv(gtow / "model_input.csv", index=False)
    gtow_df.to_csv(gtow / "audit_candidates.csv", index=False)
    return pokerbench, poker_gto, gtow


def test_v2_cli_contract_outputs_and_final_report(tmp_path: Path) -> None:
    pokerbench, poker_gto, gtow = write_source_dirs(tmp_path / "sources")
    merged = tmp_path / "merged_oracle_3intent_v2_full"
    preflop = tmp_path / "merged_oracle_preflop_model_v2_full"
    postflop = tmp_path / "merged_oracle_postflop_model_v2_full"
    final_report = tmp_path / "merged_oracle_v2_full_final_report.md"

    subprocess.run(
        [
            sys.executable,
            "datasets/merge_oracle_3intent_dataset.py",
            "--pokerbench-dir",
            str(pokerbench),
            "--poker-gto-dir",
            str(poker_gto),
            "--gtow-dir",
            str(gtow),
            "--output-dir",
            str(merged),
            "--force",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "experiments/merged_oracle_preflop_postflop_v2.py",
            "--data-dir",
            str(merged),
            "--output-root",
            str(tmp_path),
            "--preflop-output-dir",
            str(preflop),
            "--postflop-output-dir",
            str(postflop),
            "--final-report",
            str(final_report),
            "--force",
        ],
        check=True,
    )

    report = json.loads((merged / "merge_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "merged_oracle_3intent_v2_full"
    assert "PHH / Zenodo ACPC HUNL" in report["excluded_sources"]
    for name in [
        "model_input.csv",
        "model_input_preflop.csv",
        "model_input_postflop.csv",
        "X_train_preflop.csv",
        "X_validation_preflop.csv",
        "X_test_preflop.csv",
        "X_train_postflop.csv",
        "X_validation_postflop.csv",
        "X_test_postflop.csv",
    ]:
        assert (merged / name).exists()

    x_header = pd.read_csv(merged / "X_train_preflop.csv", nrows=0).columns.tolist()
    forbidden = {
        "source_dataset",
        "source_row_id",
        "raw_prompt",
        "raw_response",
        "raw_action",
        "normalized_action_4class",
        "label_3intent",
        "source_license",
        "source_url",
        "leakage_risk_notes",
    }
    assert not (set(x_header) & forbidden)

    for model_dir in [preflop, postflop]:
        training_report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
        for key in [
            "rows_train",
            "rows_validation",
            "rows_test",
            "label_distribution_train",
            "source_distribution_test",
            "recall_NO_INVEST",
            "recall_CALL",
            "recall_RAISE",
            "precision_per_class",
            "performance_by_label_and_source",
            "model_path",
        ]:
            assert key in training_report
        assert training_report["leakage_columns_used_by_model"] == []

    assert final_report.exists()
    assert final_report.stat().st_mtime >= (preflop / "training_report.json").stat().st_mtime
    assert "PHH/ACPC is excluded" in final_report.read_text(encoding="utf-8")
