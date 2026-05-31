from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.pokerbench_oracle_baseline_v1 import (
    ALLOWED_LABELS,
    LIVE_BB_MODEL,
    normalize_label,
    run_pokerbench_oracle_baseline_v1,
)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_pokerbench_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    preflop = data_dir / "preflop_60k_train_set_game_scenario_information.csv"
    postflop = data_dir / "postflop_10k_test_set_game_scenario_information.csv"

    with preflop.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["", "prev_line", "hero_pos", "hero_holding", "correct_decision", "num_players", "num_bets", "available_moves", "pot_size"],
        )
        writer.writeheader()
        labels = ["check", "fold", "call", "10.0bb"] * 8
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "": index,
                    "prev_line": "HJ/2.0bb/CO/call/BTN/10.0bb",
                    "hero_pos": "HJ",
                    "hero_holding": "AhKc",
                    "correct_decision": label,
                    "num_players": 4,
                    "num_bets": 2,
                    "available_moves": "['call', 'fold', 'allin']",
                    "pot_size": 25.0 + index,
                }
            )

    with postflop.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "preflop_action",
                "board_flop",
                "board_turn",
                "board_river",
                "aggressor_position",
                "postflop_action",
                "evaluation_at",
                "available_moves",
                "pot_size",
                "hero_position",
                "holding",
                "correct_decision",
            ],
        )
        writer.writeheader()
        labels = ["Check", "Fold", "Call", "Raise 29"] * 8
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "": index,
                    "preflop_action": "CO/2.3bb/BTN/call",
                    "board_flop": "QcAdKh",
                    "board_turn": "Ts",
                    "board_river": "Tc",
                    "aggressor_position": "OOP",
                    "postflop_action": "OOP_CHECK/IP_BET_3/IP_RAISE_11",
                    "evaluation_at": "River",
                    "available_moves": "['Fold', 'Call', 'Raise 29']",
                    "pot_size": 20.0 + index,
                    "hero_position": "OOP",
                    "holding": "Kc5c",
                    "correct_decision": label,
                }
            )


def test_pokerbench_label_normalization_maps_raise_amounts() -> None:
    assert normalize_label("check") == "CHECK"
    assert normalize_label("fold") == "FOLD"
    assert normalize_label("call") == "CALL"
    assert normalize_label("bet 24") == "RAISE"
    assert normalize_label("Raise 29") == "RAISE"
    assert normalize_label("10.0bb") == "RAISE"


def test_pokerbench_oracle_pipeline_trains_four_classes_without_leakage_or_live_overwrite(tmp_path: Path) -> None:
    data_dir = tmp_path / "pokerbench"
    output_dir = tmp_path / "pokerbench_oracle_baseline_v1"
    write_pokerbench_fixture(data_dir)
    live_hash_before = sha256(LIVE_BB_MODEL)

    report = run_pokerbench_oracle_baseline_v1(data_dir=data_dir, output_dir=output_dir, download=False)

    assert report["status"] == "ok"
    assert report["allowed_predictions"] == list(ALLOWED_LABELS)
    assert set(report["label_distribution"]) == set(ALLOWED_LABELS)
    assert report["unmapped_count"] == 0
    assert report["label_source"] == "pokerbench_solver_oracle"
    assert report["offline_prediction"]["status"] == "ok"
    assert report["live_bb_baseline_v1_overwritten"] is False
    assert sha256(LIVE_BB_MODEL) == live_hash_before

    assert (output_dir / "candidates.csv").exists()
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "training_report.json").exists()
    assert (output_dir / "feature_contract.json").exists()
    assert (output_dir / "comparison_with_live_bb_baseline_v1.md").exists()
    assert (output_dir / "eda_label_distribution.svg").exists()
    assert (output_dir / "eda_street_distribution.svg").exists()
    assert (output_dir / "eda_label_by_street.svg").exists()
    assert (output_dir / "confusion_matrix.svg").exists()
    assert (output_dir / "feature_importance.svg").exists()
    assert (output_dir / "feature_correlation.svg").exists()
    assert (output_dir / "learning_curve.svg").exists()
    assert (output_dir / "graphical_study.md").exists()
    assert report["graphical_study"]["graphical_study_md"].endswith("graphical_study.md")

    contract = json.loads((output_dir / "feature_contract.json").read_text(encoding="utf-8"))
    assert contract["call_is_native_class"] is True
    assert contract["leakage_columns_used_by_model"] == []
    assert "pokerbench.correct_decision_raw" not in contract["features_model_used"]
    assert not any(feature.startswith(("labels.", "debug.", "audit.")) for feature in contract["features_model_used"])


def test_launch_notebook_is_clean_and_pokerbench_centered() -> None:
    notebook = json.loads(Path("launch.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "PokerBench" in source
    assert "RUN_TRAINING" in source
    assert "RUN_GRAPHICAL_STUDY" in source
    assert "FAST_MODE" in source
    assert "FORCE_REBUILD" in source
    assert "gtow_llama_sft_v3" in source
    assert "bootstrap_candidate_dataset_v3" not in source.lower()
    assert "bootstrap_candidate_dataset_v4" not in source.lower()
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert sum(len(cell.get("outputs", [])) for cell in notebook["cells"]) == 0
