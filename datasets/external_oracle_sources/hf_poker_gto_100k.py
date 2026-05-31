"""Adapter for jevonmao/poker-gto-100k."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    export_dataset,
    first_text,
    load_hf_viewer_rows,
    metadata_dict,
    parse_action_history,
    parse_legal_actions,
    parse_number,
    parse_street,
)
from .label_mapping import extract_bet_size, normalize_action_3intent, normalize_action_4class


SOURCE_DATASET = "jevonmao/poker-gto-100k"
SOURCE_URL = "https://huggingface.co/datasets/jevonmao/poker-gto-100k"
SOURCE_LICENSE = "agpl-3.0 per Hugging Face listing; verify current dataset card before redistribution or commercial use."
DEFAULT_CONFIG = "action_sft"
DEFAULT_SPLITS = ("train",)


def export_poker_gto_100k(
    *,
    output_dir: str | Path,
    sample_size: int | None = None,
    no_download: bool = False,
    force: bool = False,
    config: str = DEFAULT_CONFIG,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_rows = list(records) if records is not None else load_hf_viewer_rows(
        dataset=SOURCE_DATASET,
        config=config,
        splits=DEFAULT_SPLITS,
        sample_size=sample_size,
        no_download=no_download,
    )
    normalized, audit = normalize_records(raw_rows, config=config)
    return export_dataset(
        normalized_rows=normalized,
        audit_rows=audit,
        output_dir=output_dir,
        source_dataset=SOURCE_DATASET,
        source_url=SOURCE_URL,
        source_license_or_terms=SOURCE_LICENSE,
        recommended_next_step="Compare normalized field coverage against PokerBench before merging sources.",
        force=force,
    )


def normalize_records(records: Sequence[Mapping[str, Any]], *, config: str = DEFAULT_CONFIG) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    audit = []
    for index, record in enumerate(records):
        row = normalize_record(record, fallback_index=index, config=config)
        normalized.append(row)
        audit.append(build_audit_row(row, record))
    return normalized, audit


def normalize_record(record: Mapping[str, Any], *, fallback_index: int, config: str) -> dict[str, Any]:
    metadata = metadata_dict(record)
    prompt = first_text(record, ("prompt", "instruction"))
    raw_action = first_text(record, ("response", "output", "chosen", "action"))
    four_class = normalize_action_4class(raw_action)
    source_split = str(record.get("_source_split") or record.get("source_split") or "train")
    source_row_idx = record.get("_source_row_idx", record.get("source_row_id", fallback_index))
    street = str(metadata.get("street") or record.get("street") or parse_street(prompt) or "").upper() or None
    hero_position = metadata.get("to_act") or record.get("position") or parse_position(prompt)
    return {
        "source_dataset": SOURCE_DATASET,
        "source_row_id": f"{config}:{source_split}:{source_row_idx}",
        "source_split": source_split,
        "hand_id": record.get("hand_id"),
        "street": street,
        "hero_position": hero_position,
        "villain_position": infer_hu_villain_position(hero_position),
        "hero_cards": metadata.get("hero_hand") or record.get("hole_cards"),
        "board_cards": metadata.get("board") or record.get("board"),
        "pot_size": parse_number(r"current pot size is\s*(\d+(?:\.\d+)?)", prompt),
        "effective_stack": parse_number(r"current stack:\s*(\d+(?:\.\d+)?)", prompt),
        "action_history": parse_action_history(prompt),
        "legal_actions": parse_legal_actions(prompt),
        "raw_action": raw_action,
        "normalized_action_4class": four_class,
        "label_3intent": normalize_action_3intent(raw_action),
        "bet_size": extract_bet_size(raw_action),
        "ev": parse_number(r"Your EV:\s*(-?\d+(?:\.\d+)?)", prompt),
        "equity": parse_number(r"Your equity:\s*(\d+(?:\.\d+)?)\s*%", prompt),
        "source_license": SOURCE_LICENSE,
        "source_url": SOURCE_URL,
        "leakage_risk_notes": "prompt/response/chosen/rejected contain answer text and are audit-only",
    }


def build_audit_row(normalized: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(normalized)
    row.update(
        {
            "raw_prompt": record.get("prompt"),
            "raw_response": record.get("response"),
            "raw_chosen": record.get("chosen"),
            "raw_rejected": record.get("rejected"),
            "raw_metadata_json": json.dumps(record.get("metadata"), ensure_ascii=False, sort_keys=True) if record.get("metadata") is not None else None,
            "raw_record_json": json.dumps(dict(record), ensure_ascii=False, sort_keys=True),
        }
    )
    return row


def parse_position(text: str) -> str | None:
    import re

    match = re.search(r"your position is\s*([A-Z]{2,3})", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def infer_hu_villain_position(hero_position: Any) -> str | None:
    normalized = str(hero_position or "").upper()
    if normalized == "SB":
        return "BB"
    if normalized == "BB":
        return "SB"
    return None

