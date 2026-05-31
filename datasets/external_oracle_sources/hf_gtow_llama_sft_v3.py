"""Adapter for jevonmao/gtow-llama-sft-v3."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    export_dataset,
    first_text,
    load_hf_viewer_rows,
    parse_action_history,
    parse_legal_actions,
    parse_number,
    parse_street,
)
from .hf_poker_gto_100k import infer_hu_villain_position
from .label_mapping import extract_bet_size, normalize_action_3intent, normalize_action_4class


SOURCE_DATASET = "jevonmao/gtow-llama-sft-v3"
SOURCE_URL = "https://huggingface.co/datasets/jevonmao/gtow-llama-sft-v3"
SOURCE_TERMS = "Derived from GTO Wizard API; verify terms before redistribution or commercial use."
DEFAULT_CONFIG = "default"
DEFAULT_SPLITS = ("train", "validation", "test")


def export_gtow_llama_sft_v3(
    *,
    output_dir: str | Path,
    sample_size: int | None = None,
    no_download: bool = False,
    force: bool = False,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_rows = list(records) if records is not None else load_hf_viewer_rows(
        dataset=SOURCE_DATASET,
        config=DEFAULT_CONFIG,
        splits=DEFAULT_SPLITS,
        sample_size=sample_size,
        no_download=no_download,
    )
    normalized, audit = normalize_records(raw_rows)
    return export_dataset(
        normalized_rows=normalized,
        audit_rows=audit,
        output_dir=output_dir,
        source_dataset=SOURCE_DATASET,
        source_url=SOURCE_URL,
        source_license_or_terms=SOURCE_TERMS,
        recommended_next_step="Audit terms and compare heads-up 200BB coverage against PokerBench before source mixing.",
        force=force,
    )


def normalize_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    audit = []
    for index, record in enumerate(records):
        row = normalize_record(record, fallback_index=index)
        normalized.append(row)
        audit.append(build_audit_row(row, record))
    return normalized, audit


def normalize_record(record: Mapping[str, Any], *, fallback_index: int) -> dict[str, Any]:
    prompt = first_text(record, ("prompt", "instruction"))
    raw_action = first_text(record, ("response", "output", "action"))
    four_class = normalize_action_4class(raw_action)
    source_split = str(record.get("_source_split") or record.get("source_split") or "train")
    source_row_idx = record.get("_source_row_idx", record.get("source_row_id", fallback_index))
    street = str(record.get("street") or parse_street(prompt) or "").upper() or None
    hero_position = record.get("position") or parse_position(prompt)
    return {
        "source_dataset": SOURCE_DATASET,
        "source_row_id": f"{source_split}:{source_row_idx}",
        "source_split": source_split,
        "hand_id": record.get("hand_id"),
        "street": street,
        "hero_position": hero_position,
        "villain_position": infer_hu_villain_position(hero_position),
        "hero_cards": record.get("hole_cards"),
        "board_cards": record.get("board"),
        "pot_size": record.get("pot") if record.get("pot") is not None else parse_number(r"current pot size is\s*(\d+(?:\.\d+)?)", prompt),
        "effective_stack": parse_number(r"current stack:\s*(\d+(?:\.\d+)?)", prompt),
        "action_history": parse_action_history(prompt),
        "legal_actions": parse_legal_actions(prompt),
        "raw_action": raw_action,
        "normalized_action_4class": four_class,
        "label_3intent": normalize_action_3intent(raw_action),
        "bet_size": record.get("bet_amount") if record.get("bet_amount") is not None else extract_bet_size(raw_action),
        "ev": None,
        "equity": None,
        "source_license": SOURCE_TERMS,
        "source_url": SOURCE_URL,
        "leakage_risk_notes": "prompt/response contain answer text and are audit-only; GTO Wizard terms must be verified",
    }


def build_audit_row(normalized: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(normalized)
    row.update(
        {
            "raw_prompt": record.get("prompt"),
            "raw_instruction": record.get("instruction"),
            "raw_response": record.get("response"),
            "raw_output": record.get("output"),
            "raw_record_json": json.dumps(dict(record), ensure_ascii=False, sort_keys=True),
        }
    )
    return row


def parse_position(text: str) -> str | None:
    import re

    match = re.search(r"your position is\s*([A-Z]{2,3})", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None

