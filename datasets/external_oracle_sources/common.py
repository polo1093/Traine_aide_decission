"""Common export helpers for external poker oracle source adapters."""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "external_poker_oracle_dataset_v1"
NORMALIZED_COLUMNS = (
    "source_dataset",
    "source_row_id",
    "source_split",
    "hand_id",
    "street",
    "hero_position",
    "villain_position",
    "hero_cards",
    "board_cards",
    "pot_size",
    "effective_stack",
    "action_history",
    "legal_actions",
    "raw_action",
    "normalized_action_4class",
    "label_3intent",
    "bet_size",
    "ev",
    "equity",
    "source_license",
    "source_url",
    "leakage_risk_notes",
)
AUDIT_RAW_COLUMNS = (
    "raw_prompt",
    "raw_instruction",
    "raw_response",
    "raw_output",
    "raw_chosen",
    "raw_rejected",
    "raw_metadata_json",
    "raw_record_json",
)
LEAKAGE_COLUMNS = set(AUDIT_RAW_COLUMNS)
POKERBENCH_FEATURE_COLUMNS = (
    "features.pot_bb",
    "features.to_call_bb",
    "features.hero_stack_bb",
    "features.effective_stack_bb",
    "features.stack_to_pot_ratio",
    "features.to_call_pot_ratio",
    "features.has_check",
    "features.has_fold",
    "features.has_call",
    "features.has_raise",
    "features.num_players",
    "features.num_bets",
    "features.board_card_count",
    "features.action_count",
    "features.prior_check_count",
    "features.prior_call_count",
    "features.prior_bet_raise_count",
    "features.prior_fold_count",
    "metadata.street",
    "features.hero_position",
)


def empty_normalized_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NORMALIZED_COLUMNS))


def normalized_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    normalized = []
    for row in rows:
        normalized.append({column: row.get(column) for column in NORMALIZED_COLUMNS})
    return pd.DataFrame(normalized, columns=list(NORMALIZED_COLUMNS))


def validate_normalized_schema(df: pd.DataFrame) -> None:
    missing = [column for column in NORMALIZED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing_normalized_column:{missing[0]}")


def load_hf_viewer_rows(
    *,
    dataset: str,
    config: str,
    splits: Sequence[str],
    sample_size: int | None,
    no_download: bool,
) -> list[dict[str, Any]]:
    if no_download:
        raise RuntimeError(
            "download_disabled_for_huggingface_dataset: remove --no-download or pass mocked records in tests"
        )
    if sample_size is None:
        parquet_rows = load_hf_parquet_rows(dataset=dataset, config=config, splits=splits)
        if parquet_rows:
            return parquet_rows
    target = sample_size if sample_size is not None else None
    rows: list[dict[str, Any]] = []
    for split in splits:
        offset = 0
        while target is None or len(rows) < target:
            length = 100 if target is None else max(1, min(100, target - len(rows)))
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": length,
                }
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            page = payload.get("rows") or []
            if not page:
                break
            for item in page:
                row = dict(item.get("row") or {})
                row["_source_row_idx"] = item.get("row_idx")
                row["_source_split"] = split
                row["_source_config"] = config
                rows.append(row)
                if target is not None and len(rows) >= target:
                    break
            offset += len(page)
            if len(page) < length:
                break
    return rows


def load_hf_parquet_rows(*, dataset: str, config: str, splits: Sequence[str]) -> list[dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/parquet?" + urllib.parse.urlencode({"dataset": dataset})
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in payload.get("parquet_files") or []:
        if item.get("config") != config or item.get("split") not in set(splits):
            continue
        try:
            frame = pd.read_parquet(item["url"])
        except Exception:
            return []
        split = str(item.get("split") or "")
        for index, record in frame.reset_index(drop=True).iterrows():
            row = record.to_dict()
            row["_source_row_idx"] = index
            row["_source_split"] = split
            row["_source_config"] = config
            rows.append(row)
    return rows


def export_dataset(
    *,
    normalized_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    source_dataset: str,
    source_url: str,
    source_license_or_terms: str,
    recommended_next_step: str,
    force: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(f"output_dir_exists_use_force:{output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    df = normalized_dataframe(normalized_rows)
    validate_normalized_schema(df)
    usable_df = df[df["label_3intent"].notna() & (df["label_3intent"].astype(str).str.len() > 0)].copy()

    model_input_path = output_path / "model_input.csv"
    compatibility_path = output_path / "model_features_20.csv"
    audit_path = output_path / "audit_candidates.csv"
    report_path = output_path / "dataset_report.json"
    card_path = output_path / "dataset_card.md"

    write_dataframe_csv(usable_df, model_input_path)
    write_dataframe_csv(build_pokerbench_compatibility_frame(usable_df), compatibility_path)
    write_audit_csv(audit_rows, audit_path)
    report = build_report(
        df=df,
        usable_df=usable_df,
        source_dataset=source_dataset,
        source_url=source_url,
        source_license_or_terms=source_license_or_terms,
        recommended_next_step=recommended_next_step,
    )
    report["pokerbench_feature_compatibility_file"] = "model_features_20.csv"
    report["pokerbench_feature_compatibility_notes"] = [
        "Derived best-effort from normalized external fields.",
        "Missing to_call is encoded as 0.0 unless legal-actions text exposes an amount.",
        "No raw prompt/instruction/answer text is used in the compatibility features.",
    ]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    card_path.write_text(render_dataset_card(report), encoding="utf-8")
    return report


def write_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_audit_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(NORMALIZED_COLUMNS) + [column for column in AUDIT_RAW_COLUMNS if any(column in row for row in rows)]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_pokerbench_compatibility_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        pot = float_or_zero(row.get("pot_size"))
        stack = float_or_zero(row.get("effective_stack"))
        to_call = infer_to_call(row.get("legal_actions"))
        action_counts = count_action_history(row.get("action_history"))
        legal = str(row.get("legal_actions") or "").lower()
        board = str(row.get("board_cards") or "")
        feature_row = {
            "features.pot_bb": pot / 100.0 if pot else 0.0,
            "features.to_call_bb": to_call / 100.0 if to_call else 0.0,
            "features.hero_stack_bb": stack / 100.0 if stack else 0.0,
            "features.effective_stack_bb": stack / 100.0 if stack else 0.0,
            "features.stack_to_pot_ratio": stack / pot if pot > 0 else 0.0,
            "features.to_call_pot_ratio": to_call / pot if pot > 0 else 0.0,
            "features.has_check": 1.0 if "check" in legal else 0.0,
            "features.has_fold": 1.0 if "fold" in legal else 0.0,
            "features.has_call": 1.0 if "call" in legal else 0.0,
            "features.has_raise": 1.0 if any(token in legal for token in ("bet", "raise", "all-in", "allin")) else 0.0,
            "features.num_players": 2.0,
            "features.num_bets": float(action_counts["bet_raise"]),
            "features.board_card_count": float(len(re.findall(r"[2-9TJQKA][cdhs]", board, flags=re.IGNORECASE))),
            "features.action_count": float(action_counts["action_count"]),
            "features.prior_check_count": float(action_counts["check"]),
            "features.prior_call_count": float(action_counts["call"]),
            "features.prior_bet_raise_count": float(action_counts["bet_raise"]),
            "features.prior_fold_count": float(action_counts["fold"]),
            "metadata.street": str(row.get("street") or "UNKNOWN").upper(),
            "features.hero_position": str(row.get("hero_position") or "UNKNOWN").upper(),
            "label_3intent": row.get("label_3intent"),
            "source_dataset": row.get("source_dataset"),
            "source_row_id": row.get("source_row_id"),
        }
        rows.append(feature_row)
    return pd.DataFrame(rows, columns=[*POKERBENCH_FEATURE_COLUMNS, "label_3intent", "source_dataset", "source_row_id"])


def infer_to_call(legal_actions: Any) -> float:
    text = str(legal_actions or "").lower()
    match = re.search(r"call\s+(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    return float_or_zero(match.group(1))


def count_action_history(action_history: Any) -> dict[str, int]:
    text = str(action_history or "").lower()
    check = len(re.findall(r"\bcheck(?:ed)?\b", text))
    call = len(re.findall(r"\bcall(?:ed)?\b", text))
    fold = len(re.findall(r"\bfold(?:ed)?\b", text))
    bet_raise = len(re.findall(r"\b(?:bet|bets|raised?|all-in|allin)\b", text))
    return {
        "action_count": check + call + fold + bet_raise,
        "check": check,
        "call": call,
        "fold": fold,
        "bet_raise": bet_raise,
    }


def float_or_zero(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_report(
    *,
    df: pd.DataFrame,
    usable_df: pd.DataFrame,
    source_dataset: str,
    source_url: str,
    source_license_or_terms: str,
    recommended_next_step: str,
) -> dict[str, Any]:
    label_counts = Counter(str(value) for value in usable_df["label_3intent"].dropna())
    raw_counts = Counter(str(value) for value in df["raw_action"].dropna())
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": source_dataset,
        "source_url": source_url,
        "source_license_or_terms": source_license_or_terms,
        "rows_loaded": int(len(df)),
        "rows_usable": int(len(usable_df)),
        "rows_dropped": int(len(df) - len(usable_df)),
        "label_distribution_3intent": dict(sorted(label_counts.items())),
        "raw_action_distribution": dict(sorted(raw_counts.items())),
        "missing_field_rates": missing_field_rates(df),
        "leakage_warnings": [
            "raw prompt/instruction/output/chosen/rejected fields are written only to audit_candidates.csv",
            "model_input.csv excludes raw text answer fields by default",
        ],
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "recommended_next_step": recommended_next_step,
    }


def missing_field_rates(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {column: 1.0 for column in NORMALIZED_COLUMNS}
    rates = {}
    for column in NORMALIZED_COLUMNS:
        missing = df[column].isna() | (df[column].astype(str).str.strip() == "")
        rates[column] = round(float(missing.mean()), 6)
    return rates


def render_dataset_card(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# {report['source_dataset']} External Oracle Dataset",
            "",
            f"Source: `{report['source_url']}`",
            "",
            "## Intended Use",
            "",
            "Offline dataset inspection and future feature extraction for the 3-intent poker decision classifier.",
            "This export does not train a model and is not connected to live play.",
            "",
            "## Label Mapping",
            "",
            "- CHECK/FOLD -> NO_INVEST",
            "- CALL -> CALL",
            "- BET/RAISE/ALL-IN sizing -> RAISE",
            "",
            "## Fields Exported",
            "",
            "`model_input.csv` contains normalized structured columns and labels. `audit_candidates.csv` also keeps raw source text for inspection.",
            "",
            "## Fields Excluded From Model Input",
            "",
            "Raw prompt, instruction, output, response, chosen, rejected, metadata JSON, and full raw record JSON.",
            "",
            "## Leakage Notes",
            "",
            "\n".join(f"- {warning}" for warning in report["leakage_warnings"]),
            "",
            "## License / Terms Notes",
            "",
            str(report["source_license_or_terms"]),
            "",
            "## Limitations",
            "",
            "- Heads-up datasets may not transfer directly to 6-max play.",
            "- 200BB stack depth may be over-represented where applicable.",
            "- Solver/GTO-style labels and expert/imitation histories should not be mixed without explicit source tracking.",
            "- Verify redistribution and commercial-use terms before publishing derived datasets.",
            "",
        ]
    )


def first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def metadata_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def parse_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_legal_actions(text: str) -> str | None:
    match = re.search(r"Legal actions:\s*([^\n.]+(?:\.[^\n]*)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def parse_action_history(text: str) -> str | None:
    match = re.search(r"Here is a game summary:\s*(.*?)\n\s*Now it is your turn", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def parse_street(text: str) -> str | None:
    lowered = text.lower()
    if "before the flop" in lowered or "preflop" in lowered:
        return "PREFLOP"
    for street in ("flop", "turn", "river"):
        if re.search(rf"\bon the {street}\b|\b{street}\b", lowered):
            return street.upper()
    return None
