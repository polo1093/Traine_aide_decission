from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasets.external_oracle_sources.common import NORMALIZED_COLUMNS
from datasets.external_oracle_sources.hf_gtow_llama_sft_v3 import export_gtow_llama_sft_v3
from datasets.external_oracle_sources.hf_poker_gto_100k import export_poker_gto_100k
from datasets.external_oracle_sources.phh_acpc_scaffold import export_phh_behavioral_dataset, parse_phh_file, write_phh_scaffold


def poker_gto_records() -> list[dict[str, object]]:
    return [
        {
            "_source_row_idx": 0,
            "_source_split": "train",
            "prompt": (
                "Here is a game summary:\n\nThe small blind is 50 chips and the big blind is 100 chips. "
                "In this hand, your position is SB, and your holding is [Eight of Club and King of Spade]. "
                "Before the flop, SB raised, BB called.\n\n"
                "Now it is your turn to make a move on the flop.\n"
                "To remind you, the current pot size is 500 chips.\n"
                "Your current stack: 19750 chips.\n"
                "Legal actions: check, bet 165, bet 375.\n\n"
                "GTO SOLVER CONTEXT:\nYour equity: 58.4%\nYour EV: 254 chips\n"
            ),
            "response": "<action>check</action>",
            "metadata": {"matchup": "SRP", "street": "flop", "board": "2c2d2h", "to_act": "SB", "hero_hand": "8cKs"},
        },
        {
            "_source_row_idx": 1,
            "_source_split": "train",
            "prompt": (
                "Here is a game summary:\n\nBefore the flop, SB raised, BB called.\n\n"
                "Now it is your turn to make a move on the turn.\n"
                "To remind you, the current pot size is 500 chips.\n"
                "Your current stack: 19750 chips.\n"
                "Legal actions: check, bet 375.\n\n"
                "GTO SOLVER CONTEXT:\nYour equity: 73.3%\nYour EV: 432 chips\n"
            ),
            "response": "<action>bet 375</action>",
            "metadata": {"matchup": "SRP", "street": "turn", "board": "2c2d2h3h", "to_act": "BB", "hero_hand": "8hAc"},
        },
    ]


def gtow_records() -> list[dict[str, object]]:
    return [
        {
            "_source_row_idx": 0,
            "_source_split": "train",
            "prompt": (
                "Here is a game summary:\n\nBefore the flop, SB raise 200 chips.\n\n"
                "Now it is your turn to make a move before the flop.\n"
                "To remind you, the current pot size is 300 chips.\n"
                "Your current stack: 19900 chips.\n"
                "Legal actions: call, fold, bet/raise.\n"
            ),
            "response": "<action>call</action>",
            "street": "preflop",
            "position": "BB",
            "action": "c",
            "bet_amount": None,
            "pot": 300.0,
            "board": "",
            "hole_cards": "Kh5c",
        },
        {
            "_source_row_idx": 1,
            "_source_split": "validation",
            "prompt": (
                "Here is a game summary:\n\nSB raise 275 chips.\n\n"
                "Now it is your turn to make a move before the flop.\n"
                "To remind you, the current pot size is 150 chips.\n"
                "Your current stack: 19950 chips.\n"
                "Legal actions: call, fold, bet/raise.\n"
            ),
            "response": "<action>raise 275</action>",
            "street": "preflop",
            "position": "SB",
            "action": "b275",
            "bet_amount": 275.0,
            "pot": 150.0,
            "board": "",
            "hole_cards": "Ts9h",
        },
    ]


def assert_common_export(output_dir: Path) -> None:
    model_input = output_dir / "model_input.csv"
    audit = output_dir / "audit_candidates.csv"
    report_path = output_dir / "dataset_report.json"
    assert model_input.exists()
    assert audit.exists()
    assert report_path.exists()
    assert (output_dir / "dataset_card.md").exists()
    assert (output_dir / "model_features_20.csv").exists()

    model_df = pd.read_csv(model_input)
    compat_df = pd.read_csv(output_dir / "model_features_20.csv")
    audit_df = pd.read_csv(audit)
    for column in NORMALIZED_COLUMNS:
        assert column in model_df.columns
    assert "raw_prompt" not in model_df.columns
    assert "raw_response" not in model_df.columns
    assert "raw_chosen" not in model_df.columns
    assert "raw_rejected" not in model_df.columns
    assert "raw_prompt" in audit_df.columns
    assert "raw_prompt" not in compat_df.columns
    assert "features.pot_bb" in compat_df.columns
    assert "label_3intent" in compat_df.columns

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["label_distribution_3intent"]
    assert report["source_license_or_terms"]
    assert report["schema_version"] == "external_poker_oracle_dataset_v1"


def test_poker_gto_100k_sample_export_schema_antileakage_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "poker_gto_100k"
    report = export_poker_gto_100k(output_dir=output_dir, records=poker_gto_records(), sample_size=2)

    assert report["rows_loaded"] == 2
    assert report["rows_usable"] == 2
    assert report["label_distribution_3intent"] == {"NO_INVEST": 1, "RAISE": 1}
    assert "agpl" in report["source_license_or_terms"].lower()
    assert_common_export(output_dir)


def test_gtow_llama_sft_v3_sample_export_schema_antileakage_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "gtow"
    report = export_gtow_llama_sft_v3(output_dir=output_dir, records=gtow_records(), sample_size=2)

    assert report["rows_loaded"] == 2
    assert report["rows_usable"] == 2
    assert report["label_distribution_3intent"] == {"CALL": 1, "RAISE": 1}
    assert "GTO Wizard API" in report["source_license_or_terms"]
    assert_common_export(output_dir)


def test_phh_acpc_scaffold_writes_report_and_empty_parser_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "phh"
    report = write_phh_scaffold(output_dir=output_dir)
    df = parse_phh_file(tmp_path / "sample.phh")

    assert report["rows_loaded"] == 0
    assert (output_dir / "dataset_report.json").exists()
    assert (output_dir / "dataset_card.md").exists()
    assert list(df.columns) == list(NORMALIZED_COLUMNS)


def test_phh_behavioral_export_parses_local_decision_snapshots(tmp_path: Path) -> None:
    input_dir = tmp_path / "pluribus"
    input_dir.mkdir()
    (input_dir / "0.phh").write_text(
        "\n".join(
            [
                "variant = 'NT'",
                "ante_trimming_status = true",
                "antes = [0, 0, 0, 0, 0, 0]",
                "blinds_or_straddles = [50, 100, 0, 0, 0, 0]",
                "min_bet = 100",
                "starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]",
                "actions = ['d dh p1 AhKh', 'd dh p2 ????', 'd dh p3 QcJc', 'd dh p4 2s2d', 'd dh p5 7h8h', 'd dh p6 Td9d', 'p3 f', 'p4 cbr 210', 'p5 f', 'p6 f', 'p1 cc', 'p2 f', 'd db 7d5h9d', 'p1 cc', 'p4 cc', 'd db Qh', 'p1 cbr 230', 'p4 f']",
                "hand = 0",
                "players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "export"

    report = export_phh_behavioral_dataset(
        source="phh_pluribus",
        input_path=input_dir,
        output_dir=output_dir,
        force=True,
    )

    assert report["rows_loaded"] == 10
    assert report["rows_usable"] == 10
    assert report["behavioral_not_solver_oracle"] is True
    assert report["source_dataset"] == "PHH Pluribus"
    assert report["label_distribution_3intent"] == {"CALL": 1, "NO_INVEST": 7, "RAISE": 2}

    model_df = pd.read_csv(output_dir / "model_input.csv")
    assert set(model_df["label_3intent"]) == {"NO_INVEST", "CALL", "RAISE"}
    assert "Ah Kh" in set(model_df["hero_cards"].dropna())
    assert "????" in set(model_df["hero_cards"].dropna())
    assert (output_dir / "model_features_20.csv").exists()
