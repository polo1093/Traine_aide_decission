"""PHH / ACPC hand-history adapters.

The PHH sources are behavioral hand histories, not solver-oracle labels. This
module exports observed actions into the same normalized audit schema used by
the other external adapters so notebooks can inspect coverage, missing cards,
and label distributions without mixing them into supervised oracle training.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .common import SCHEMA_VERSION, empty_normalized_dataframe, export_dataset, normalized_dataframe


SOURCE_URL = "https://github.com/uoftcprg/phh-dataset"
ZENODO_URL = "https://zenodo.org/records/13997158"
ACPC_URL = "https://www.computerpokercompetition.org/"

PHH_SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "phh_handhq_nlhe": {
        "source_dataset": "PHH HandHQ NLHE",
        "source_url": f"{ZENODO_URL} / {SOURCE_URL}",
        "source_license_or_terms": "CC-BY-4.0 for the Zenodo distribution; condensed PHH repository code/data listing is MIT. Verify redistribution before publishing derived exports.",
        "expected_volume": "21,605,687 uncorrupted no-limit hold'em hands; archive is about 1.9 GB.",
        "source_type": "behavioral_hand_history",
        "recommended_next_step": "Use as behavioral pretraining, weak labels, and coverage data. Keep source tags and unknown cards; do not treat observed actions as solver truth.",
    },
    "phh_pluribus": {
        "source_dataset": "PHH Pluribus",
        "source_url": f"{ZENODO_URL} / {SOURCE_URL}",
        "source_license_or_terms": "CC-BY-4.0 via the PHH Zenodo distribution; verify current source terms before redistribution.",
        "expected_volume": "10,000 hands played by Pluribus.",
        "source_type": "expert_demonstration_hand_history",
        "recommended_next_step": "Use as expert demonstrations and error-analysis cases, tagged source=pluribus. Do not treat it as a full mixed strategy.",
    },
    "phh_wsop_tv": {
        "source_dataset": "PHH WSOP TV",
        "source_url": f"{ZENODO_URL} / {SOURCE_URL}",
        "source_license_or_terms": "CC-BY-4.0 via the PHH Zenodo distribution; verify current source terms before redistribution.",
        "expected_volume": "83 televised hands from the 2023 WSOP Event #43 final table.",
        "source_type": "case_study_hand_history",
        "recommended_next_step": "Use as a notebook case study and qualitative sanity set, not as a training-volume source.",
    },
    "acpc_public_logs": {
        "source_dataset": "ACPC public historical logs",
        "source_url": f"{ACPC_URL} / {SOURCE_URL}",
        "source_license_or_terms": "License not specified in the user-provided source table; review source-specific terms before redistribution.",
        "expected_volume": "Historical public competition logs; PHH README mentions 278,842,225 HUNL hands with duplicates/EV variants in some years.",
        "source_type": "competition_log_archive",
        "recommended_next_step": "Use only for a dedicated heads-up module or action-history encoder pretraining after archive-specific parsing and deduplication.",
    },
    "phh_acpc_scaffold": {
        "source_dataset": "PHH / Zenodo ACPC HUNL",
        "source_url": SOURCE_URL,
        "source_license_or_terms": "PHH repository is MIT licensed; ACPC/Zenodo hand-history contents are not direct solver labels and require source-specific redistribution review.",
        "expected_volume": "Scaffold alias retained for backward compatibility.",
        "source_type": "scaffold",
        "recommended_next_step": "Prefer phh_handhq_nlhe, phh_pluribus, phh_wsop_tv, or acpc_public_logs for source-specific exports.",
    },
}

PLAYER_ACTION_RE = re.compile(r"^p(?P<player>\d+)\s+(?P<code>.+)$")
DEALT_HOLE_RE = re.compile(r"^d\s+dh\s+p(?P<player>\d+)\s+(?P<cards>\S+)")
DEALT_BOARD_RE = re.compile(r"^d\s+db\s+(?P<cards>\S+)")
POSITION_BY_PLAYER_COUNT = {
    2: ("SB", "BB"),
    3: ("SB", "BB", "BTN"),
    4: ("SB", "BB", "UTG", "BTN"),
    5: ("SB", "BB", "UTG", "CO", "BTN"),
    6: ("SB", "BB", "UTG", "HJ", "CO", "BTN"),
}


def parse_phh_file(path: str | Path, *, source_key: str = "phh_acpc_scaffold", sample_size: int | None = None) -> pd.DataFrame:
    """Parse one local PHH/PHHS file into observed decision rows.

    PHH files are TOML. A `.phh` usually contains a single hand at the root; a
    `.phhs` file usually contains many numbered TOML tables.
    """
    rows = parse_phh_files([Path(path)], source_key=source_key, sample_size=sample_size)
    return normalized_dataframe(rows) if rows else empty_normalized_dataframe()


def parse_phh_files(
    paths: Iterable[str | Path],
    *,
    source_key: str,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    profile = source_profile(source_key)
    rows: list[dict[str, Any]] = []
    for path in paths:
        for hand_key, hand in load_phh_documents(Path(path)):
            rows.extend(parse_hand(hand, hand_key=hand_key, path=Path(path), profile=profile))
            if sample_size is not None and len(rows) >= sample_size:
                return rows[:sample_size]
    return rows


def load_phh_documents(path: Path) -> list[tuple[str, Mapping[str, Any]]]:
    if not path.exists():
        return []
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if "actions" in payload:
        return [(str(payload.get("hand") or path.stem), payload)]
    docs: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in payload.items():
        if isinstance(value, Mapping) and "actions" in value:
            docs.append((str(value.get("hand") or key), value))
    return docs


def parse_hand(
    hand: Mapping[str, Any],
    *,
    hand_key: str,
    path: Path,
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions = [str(action) for action in hand.get("actions") or []]
    player_count = len(hand.get("starting_stacks") or hand.get("players") or [])
    if player_count <= 0:
        return []

    blinds = numeric_list(hand.get("blinds_or_straddles"), player_count)
    antes = numeric_list(hand.get("antes"), player_count)
    starting_stacks = numeric_list(hand.get("starting_stacks"), player_count)
    street_contrib = [max(0.0, value) for value in blinds]
    current_bet = max(street_contrib or [0.0])
    pot = sum(max(0.0, value) for value in blinds) + sum(max(0.0, value) for value in antes)
    hole_cards: dict[int, str] = {}
    board_cards: list[str] = []
    street = "PREFLOP"
    rows: list[dict[str, Any]] = []
    history: list[str] = []

    for action_index, action in enumerate(actions):
        hole_match = DEALT_HOLE_RE.match(action)
        if hole_match:
            hole_cards[int(hole_match.group("player"))] = hole_match.group("cards")
            history.append(action)
            continue

        board_match = DEALT_BOARD_RE.match(action)
        if board_match:
            board_cards.extend(split_cards(board_match.group("cards")))
            street = street_from_board_count(len([card for card in board_cards if "?" not in card]))
            street_contrib = [0.0 for _ in range(player_count)]
            current_bet = 0.0
            history.append(action)
            continue

        player_match = PLAYER_ACTION_RE.match(action)
        if not player_match:
            history.append(action)
            continue

        player_number = int(player_match.group("player"))
        raw_action = player_match.group("code").strip()
        if not is_betting_action(raw_action):
            history.append(action)
            continue

        player_index = player_number - 1
        to_call = max(0.0, current_bet - value_at(street_contrib, player_index))
        normalized_action = normalize_phh_betting_action(raw_action, to_call=to_call)
        row = build_decision_row(
            profile=profile,
            path=path,
            hand=hand,
            hand_key=hand_key,
            action_index=action_index,
            player_number=player_number,
            player_count=player_count,
            raw_action=raw_action,
            normalized_action=normalized_action,
            street=street,
            board_cards=board_cards,
            hole_cards=hole_cards,
            starting_stacks=starting_stacks,
            pot=pot,
            to_call=to_call,
            history=history,
        )
        rows.append(row)

        pot, current_bet = apply_betting_action(
            raw_action=raw_action,
            player_index=player_index,
            street_contrib=street_contrib,
            current_bet=current_bet,
            pot=pot,
        )
        history.append(action)
    return rows


def build_decision_row(
    *,
    profile: Mapping[str, Any],
    path: Path,
    hand: Mapping[str, Any],
    hand_key: str,
    action_index: int,
    player_number: int,
    player_count: int,
    raw_action: str,
    normalized_action: str,
    street: str,
    board_cards: list[str],
    hole_cards: Mapping[int, str],
    starting_stacks: list[float],
    pot: float,
    to_call: float,
    history: list[str],
) -> dict[str, Any]:
    player_index = player_number - 1
    label = {"CHECK": "NO_INVEST", "FOLD": "NO_INVEST", "CALL": "CALL", "RAISE": "RAISE"}[normalized_action]
    legal = f"fold, call {to_call:g}, raise" if to_call > 0 else "check, bet"
    hand_id = str(hand.get("hand") or hand_key)
    return {
        "source_dataset": profile["source_dataset"],
        "source_row_id": f"{path.as_posix()}:{hand_id}:{action_index}",
        "source_split": "local",
        "hand_id": hand_id,
        "street": street,
        "hero_position": player_position(player_number, player_count),
        "villain_position": None,
        "hero_cards": normalize_unknown_cards(hole_cards.get(player_number)),
        "board_cards": " ".join(board_cards),
        "pot_size": pot,
        "effective_stack": value_at(starting_stacks, player_index),
        "action_history": " | ".join(history),
        "legal_actions": legal,
        "raw_action": raw_action,
        "normalized_action_4class": normalized_action,
        "label_3intent": label,
        "bet_size": extract_phh_amount(raw_action),
        "ev": None,
        "equity": None,
        "source_license": profile["source_license_or_terms"],
        "source_url": profile["source_url"],
        "leakage_risk_notes": "Observed behavioral action, not a solver-oracle best action.",
        "raw_record_json": json.dumps(
            {
                "file": str(path),
                "hand": hand_id,
                "action_index": action_index,
                "action": raw_action,
                "source_type": profile["source_type"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def export_phh_behavioral_dataset(
    *,
    source: str,
    output_dir: str | Path,
    input_path: str | Path | None = None,
    sample_size: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    profile = source_profile(source)
    paths = discover_phh_paths(Path(input_path)) if input_path else []
    rows = parse_phh_files(paths, source_key=source, sample_size=sample_size) if paths else []
    report = export_dataset(
        normalized_rows=rows,
        audit_rows=rows,
        output_dir=output_dir,
        source_dataset=profile["source_dataset"],
        source_url=profile["source_url"],
        source_license_or_terms=profile["source_license_or_terms"],
        recommended_next_step=profile["recommended_next_step"],
        force=force,
    )
    report.update(
        {
            "source_type": profile["source_type"],
            "expected_volume": profile["expected_volume"],
            "input_path": str(input_path) if input_path else None,
            "input_files_found": len(paths),
            "behavioral_not_solver_oracle": True,
            "generation_timestamp": datetime.now(UTC).isoformat(),
        }
    )
    report["leakage_warnings"] = [
        "PHH/ACPC histories are observed action traces, not direct solver-oracle labels.",
        "Use source_dataset/source_type for routing; exclude these rows from supervised oracle training unless relabeled.",
        *report.get("leakage_warnings", []),
    ]
    output_path = Path(output_dir)
    (output_path / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "dataset_card.md").write_text(render_phh_card(report), encoding="utf-8")
    return report


def write_phh_scaffold(*, output_dir: str | Path, force: bool = False) -> dict[str, Any]:
    return export_phh_behavioral_dataset(
        source="phh_acpc_scaffold",
        output_dir=output_dir,
        input_path=None,
        sample_size=None,
        force=force,
    )


def discover_phh_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in {".phh", ".phhs"}:
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in {".phh", ".phhs"})
    return []


def source_profile(source_key: str) -> dict[str, Any]:
    if source_key not in PHH_SOURCE_PROFILES:
        raise ValueError(f"unsupported_phh_source:{source_key}")
    return PHH_SOURCE_PROFILES[source_key]


def numeric_list(value: Any, size: int) -> list[float]:
    items = list(value) if isinstance(value, list) else []
    out = []
    for index in range(size):
        out.append(float_or_zero(items[index]) if index < len(items) else 0.0)
    return out


def split_cards(value: str) -> list[str]:
    text = str(value or "")
    if not text:
        return []
    return [text[index : index + 2] for index in range(0, len(text), 2)]


def street_from_board_count(count: int) -> str:
    if count >= 5:
        return "RIVER"
    if count == 4:
        return "TURN"
    if count >= 3:
        return "FLOP"
    return "PREFLOP"


def is_betting_action(raw_action: str) -> bool:
    return raw_action.split()[0] in {"f", "cc", "cbr"}


def normalize_phh_betting_action(raw_action: str, *, to_call: float) -> str:
    code = raw_action.split()[0]
    if code == "f":
        return "FOLD"
    if code == "cbr":
        return "RAISE"
    if code == "cc":
        return "CALL" if to_call > 0 else "CHECK"
    raise ValueError(f"unsupported_phh_action:{raw_action}")


def apply_betting_action(
    *,
    raw_action: str,
    player_index: int,
    street_contrib: list[float],
    current_bet: float,
    pot: float,
) -> tuple[float, float]:
    code = raw_action.split()[0]
    if code == "f":
        return pot, current_bet
    if code == "cc":
        add = max(0.0, current_bet - value_at(street_contrib, player_index))
        street_contrib[player_index] = max(current_bet, value_at(street_contrib, player_index))
        return pot + add, current_bet
    if code == "cbr":
        target = extract_phh_amount(raw_action) or current_bet
        add = max(0.0, target - value_at(street_contrib, player_index))
        street_contrib[player_index] = max(target, value_at(street_contrib, player_index))
        return pot + add, max(current_bet, target)
    return pot, current_bet


def extract_phh_amount(raw_action: str) -> float | None:
    parts = raw_action.split()
    if len(parts) < 2:
        return None
    return float_or_none(parts[1])


def normalize_unknown_cards(cards: str | None) -> str | None:
    if cards is None:
        return None
    return cards if "?" in cards else " ".join(split_cards(cards))


def player_position(player_number: int, player_count: int) -> str:
    positions = POSITION_BY_PLAYER_COUNT.get(player_count)
    if positions and 1 <= player_number <= len(positions):
        return positions[player_number - 1]
    return f"P{player_number}"


def value_at(values: list[float], index: int) -> float:
    return values[index] if 0 <= index < len(values) else 0.0


def float_or_zero(value: Any) -> float:
    parsed = float_or_none(value)
    return parsed if parsed is not None else 0.0


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def render_phh_card(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# {report['source_dataset']} Behavioral Export",
            "",
            f"Source: `{report['source_url']}`",
            "",
            "## Type",
            "",
            "Observed hand-history actions. This is not a direct solver-label dataset.",
            "",
            "## Files",
            "",
            "- `model_input.csv`: normalized observed decision rows.",
            "- `model_features_20.csv`: best-effort PokerBench-compatible features for inspection.",
            "- `audit_candidates.csv`: includes raw PHH action context for debugging.",
            "- `dataset_report.json`: coverage, labels, and source metadata.",
            "",
            "## Intended Use",
            "",
            "- Behavioral cloning experiments separated from solver-oracle models.",
            "- Weak-label pretraining and action-history encoder coverage.",
            "- Qualitative case studies and error analysis.",
            "",
            "## Exclusions",
            "",
            "- Do not merge these rows into supervised oracle training unless they are relabeled by a solver or routed as a separate objective.",
            "- Keep unknown cards as `?`/`????` instead of inventing complete hidden information.",
            "",
            "## License / Terms Notes",
            "",
            str(report["source_license_or_terms"]),
            "",
        ]
    )
