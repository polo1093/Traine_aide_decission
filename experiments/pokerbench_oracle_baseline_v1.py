"""Train a four-class oracle baseline from PokerBench solver-labeled CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("data/pokerbench")
DEFAULT_OUTPUT_DIR = Path("outputs/readiness/pokerbench_oracle_baseline_v1")
LIVE_BB_MODEL = Path("outputs/readiness/live_bb_baseline_v1/model.joblib")
LIVE_BB_REPORT = Path("outputs/readiness/live_bb_baseline_v1/training_report.json")
HF_BASE_URL = "https://huggingface.co/datasets/RZ412/PokerBench/resolve/main/"
DEFAULT_CSV_FILES = (
    "preflop_60k_train_set_game_scenario_information.csv",
    "postflop_10k_test_set_game_scenario_information.csv",
)
DEFAULT_JSON_FILES = (
    "preflop_1k_test_set_prompt_and_label.json",
    "postflop_10k_test_set_prompt_and_label.json",
)
ALLOWED_LABELS = ("CHECK", "FOLD", "CALL", "RAISE")
NUMERIC_FEATURES = (
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
)
CATEGORICAL_FEATURES = ("metadata.street", "features.hero_position")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
AUDIT_ONLY_COLUMNS = (
    "pokerbench.source_file",
    "pokerbench.row_id",
    "pokerbench.correct_decision_raw",
    "pokerbench.available_moves_raw",
    "pokerbench.preflop_action_raw",
    "pokerbench.postflop_action_raw",
    "pokerbench.hero_holding",
    "pokerbench.board_flop",
    "pokerbench.board_turn",
    "pokerbench.board_river",
)
LEAKAGE_PREFIXES = ("labels.", "debug.", "audit.", "pokerbench.correct_decision")


def run_pokerbench_oracle_baseline_v1(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    download: bool = True,
    max_rows: int | None = None,
    random_seed: int = 17,
    fast_mode: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    live_hash_before = file_sha256(LIVE_BB_MODEL)

    source_paths = prepare_pokerbench_sources(data_path, download=download)
    loaded_rows = load_pokerbench_rows(source_paths, max_rows=max_rows)
    candidates, parse_report = build_candidates(loaded_rows)
    if len(candidates) < 20:
        raise ValueError(f"not_enough_usable_pokerbench_rows:{len(candidates)}")

    candidates_csv = output_path / "candidates.csv"
    write_candidates_csv(candidates, candidates_csv)
    training = train_model(candidates, output_path=output_path, random_seed=random_seed)
    prediction = predict_first_row(output_path, candidates[0])
    comparison = build_comparison_with_live_bb(training, rows_used=len(candidates))
    (output_path / "comparison_with_live_bb_baseline_v1.md").write_text(comparison, encoding="utf-8")
    graphical = write_graphical_study(candidates, output_path=output_path, training=training, random_seed=random_seed, fast_mode=fast_mode)
    live_hash_after = file_sha256(LIVE_BB_MODEL)

    report = {
        **training,
        "schema": "pokerbench_oracle_baseline_v1",
        "dataset_source": "RZ412/PokerBench",
        "input_files": [str(path) for path in source_paths],
        "rows_loaded": len(loaded_rows),
        "rows_usable": len(candidates),
        "candidates_csv": str(candidates_csv),
        "label_source": "pokerbench_solver_oracle",
        "label_distribution": dict(sorted(Counter(row["bootstrap_label"] for row in candidates).items())),
        "street_distribution": dict(sorted(Counter(row["metadata.street"] for row in candidates).items())),
        "preflop_postflop_distribution": preflop_postflop_distribution(candidates),
        "unmapped_outputs": parse_report["unmapped_outputs"],
        "unmapped_count": parse_report["unmapped_count"],
        "offline_prediction": prediction,
        "comparison_with_live_bb_baseline_v1": "comparison_with_live_bb_baseline_v1.md",
        "graphical_study": graphical,
        "live_bb_baseline_v1_overwritten": live_hash_before != live_hash_after,
        "live_bb_baseline_v1_hash_before": live_hash_before,
        "live_bb_baseline_v1_hash_after": live_hash_after,
        "known_limits": [
            "PokerBench structured CSVs do not expose full solver tree metadata in this first pipeline.",
            "Raw prompt text is audit-only and not used as a model feature.",
            "to_call_bb is inferred from available moves and recent action amounts when exact call size is absent.",
            "Large postflop 500k CSV is not downloaded by default to keep the first run lightweight.",
        ],
        "bot_live_connection": "not_modified",
        "not_for_production": True,
    }
    write_json(report, output_path / "training_report.json")
    return report


def prepare_pokerbench_sources(data_dir: Path, *, download: bool) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [data_dir / name for name in DEFAULT_CSV_FILES]
    if all(path.exists() and path.stat().st_size > 0 for path in csv_paths):
        return csv_paths

    local_csvs = sorted(data_dir.glob("*game_scenario_information.csv"))
    if local_csvs:
        return local_csvs

    if download:
        for path in csv_paths:
            if not path.exists() or path.stat().st_size == 0:
                urllib.request.urlretrieve(HF_BASE_URL + path.name, path)
        return csv_paths

    json_paths = [data_dir / name for name in DEFAULT_JSON_FILES if (data_dir / name).exists()]
    if json_paths:
        return json_paths
    raise FileNotFoundError(f"missing_pokerbench_sources:{data_dir}")


def download_json_fallbacks(data_dir: Path) -> list[Path]:
    paths = [data_dir / name for name in DEFAULT_JSON_FILES]
    for path in paths:
        urllib.request.urlretrieve(HF_BASE_URL + path.name, path)
    return paths


def load_pokerbench_rows(paths: Sequence[Path], *, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_file_limit = None if max_rows is None else max(1, max_rows // len(paths))
    for path in paths:
        if path.suffix.lower() == ".csv":
            rows.extend(load_csv_rows(path, per_file_limit=per_file_limit))
        elif path.suffix.lower() == ".json":
            rows.extend(load_json_rows(path, per_file_limit=per_file_limit))
    return rows[:max_rows] if max_rows is not None else rows


def load_csv_rows(path: Path, *, per_file_limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            normalized = {key or "row_index": value for key, value in row.items()}
            normalized["source_file"] = path.name
            rows.append(normalized)
            if per_file_limit is not None and index + 1 >= per_file_limit:
                break
    return rows


def load_json_rows(path: Path, *, per_file_limit: int | None) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("data") or raw.get("rows") or raw.get("examples") or []
    else:
        items = raw
    rows = []
    for index, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        rows.append(normalize_json_row(item, source_file=path.name, index=index))
        if per_file_limit is not None and len(rows) >= per_file_limit:
            break
    return rows


def normalize_json_row(item: Mapping[str, Any], *, source_file: str, index: int) -> dict[str, Any]:
    row = dict(item)
    row.setdefault("row_index", index)
    row["source_file"] = source_file
    row.setdefault("correct_decision", item.get("label") or item.get("answer") or item.get("output") or item.get("correct_action"))
    row.setdefault("available_moves", item.get("available_moves") or item.get("legal_actions") or item.get("actions"))
    row.setdefault("pot_size", item.get("pot_size") or item.get("pot") or 0)
    row.setdefault("hero_pos", item.get("hero_pos") or item.get("hero_position"))
    row.setdefault("prev_line", item.get("prev_line") or item.get("action_history") or item.get("history") or "")
    row.setdefault("postflop_action", item.get("postflop_action") or item.get("action_history") or "")
    row.setdefault("evaluation_at", item.get("evaluation_at") or item.get("street") or ("PREFLOP" if source_file.startswith("preflop") else "UNKNOWN"))
    return row


def build_candidates(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    unmapped = Counter()
    for row in rows:
        label = normalize_label(row.get("correct_decision"))
        if label not in ALLOWED_LABELS:
            unmapped[str(row.get("correct_decision") or "")] += 1
            continue
        candidates.append(flatten_row(row, label))
    return candidates, {
        "unmapped_outputs": dict(sorted(unmapped.items())),
        "unmapped_count": int(sum(unmapped.values())),
    }


def flatten_row(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    source = str(row.get("source_file") or "")
    is_preflop = source.startswith("preflop")
    street = "PREFLOP" if is_preflop else str(row.get("evaluation_at") or "UNKNOWN").upper()
    available_moves = parse_available_moves(row.get("available_moves"))
    history = str(row.get("prev_line") or row.get("postflop_action") or "")
    pot = number_or_zero(row.get("pot_size"))
    to_call = infer_to_call(available_moves=available_moves, history=history)
    stack = 100.0
    board_cards = board_cards_for(row)
    action_counts = count_history_actions(history)
    candidate = {
        "snapshot_id": f"pokerbench:{source}:{row.get('row_index')}",
        "bootstrap_label": label,
        "metadata.street": street,
        "metadata.label_source": "pokerbench_solver_oracle",
        "features.pot_bb": pot,
        "features.to_call_bb": to_call,
        "features.hero_stack_bb": stack,
        "features.effective_stack_bb": stack,
        "features.stack_to_pot_ratio": stack / pot if pot > 0 else 0.0,
        "features.to_call_pot_ratio": to_call / pot if pot > 0 else 0.0,
        "features.has_check": 1.0 if has_action(available_moves, "CHECK") else 0.0,
        "features.has_fold": 1.0 if has_action(available_moves, "FOLD") else 0.0,
        "features.has_call": 1.0 if has_action(available_moves, "CALL") else 0.0,
        "features.has_raise": 1.0 if has_raise_action(available_moves) else 0.0,
        "features.num_players": number_or_zero(row.get("num_players")) or 6.0,
        "features.num_bets": number_or_zero(row.get("num_bets")),
        "features.board_card_count": float(len(board_cards)),
        "features.action_count": float(action_counts["action_count"]),
        "features.prior_check_count": float(action_counts["check"]),
        "features.prior_call_count": float(action_counts["call"]),
        "features.prior_bet_raise_count": float(action_counts["bet_raise"]),
        "features.prior_fold_count": float(action_counts["fold"]),
        "features.hero_position": str(row.get("hero_pos") or row.get("hero_position") or "UNKNOWN").upper(),
        "pokerbench.source_file": source,
        "pokerbench.row_id": row.get("row_index"),
        "pokerbench.correct_decision_raw": row.get("correct_decision"),
        "pokerbench.available_moves_raw": row.get("available_moves"),
        "pokerbench.preflop_action_raw": row.get("prev_line") or row.get("preflop_action"),
        "pokerbench.postflop_action_raw": row.get("postflop_action"),
        "pokerbench.hero_holding": row.get("hero_holding") or row.get("holding"),
        "pokerbench.board_flop": row.get("board_flop"),
        "pokerbench.board_turn": row.get("board_turn"),
        "pokerbench.board_river": row.get("board_river"),
    }
    return candidate


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    head = text.split()[0].replace("_", " ")
    if head == "CHECK":
        return "CHECK"
    if head == "FOLD":
        return "FOLD"
    if head == "CALL":
        return "CALL"
    if head in {"BET", "RAISE"}:
        return "RAISE"
    if text.startswith("RAISE") or text.startswith("BET"):
        return "RAISE"
    if re.fullmatch(r"\d+(?:\.\d+)?\s*BB", text) or re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "RAISE"
    return text


def parse_available_moves(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def infer_to_call(*, available_moves: Sequence[str], history: str) -> float:
    if not has_action(available_moves, "CALL"):
        return 0.0
    amounts = [float(match) for match in re.findall(r"(?:BET|RAISE|/)(\d+(?:\.\d+)?)", history.upper())]
    if amounts:
        return max(0.0, amounts[-1])
    return 1.0


def has_action(moves: Sequence[str], action: str) -> bool:
    target = action.upper()
    return any(str(move).strip().upper().startswith(target) for move in moves)


def has_raise_action(moves: Sequence[str]) -> bool:
    return any(str(move).strip().upper().startswith(("BET", "RAISE", "ALLIN", "ALL-IN")) for move in moves)


def board_cards_for(row: Mapping[str, Any]) -> list[str]:
    cards = []
    flop = str(row.get("board_flop") or "")
    if flop and flop.lower() != "nan":
        cards.extend(split_cards(flop))
    for key in ("board_turn", "board_river"):
        value = str(row.get(key) or "")
        if value and value.lower() != "nan":
            cards.extend(split_cards(value))
    return cards


def split_cards(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    return [text[index : index + 2] for index in range(0, len(text), 2) if len(text[index : index + 2]) == 2]


def count_history_actions(history: str) -> dict[str, int]:
    upper = str(history or "").upper()
    check = upper.count("CHECK")
    call = upper.count("CALL")
    fold = upper.count("FOLD")
    bet_raise = upper.count("BET") + upper.count("RAISE") + upper.count("ALLIN")
    return {
        "action_count": check + call + fold + bet_raise,
        "check": check,
        "call": call,
        "fold": fold,
        "bet_raise": bet_raise,
    }


def train_model(rows: Sequence[Mapping[str, Any]], *, output_path: Path, random_seed: int) -> dict[str, Any]:
    y = [str(row["bootstrap_label"]) for row in rows]
    train_rows, test_rows = train_test_split(list(rows), test_size=0.20, random_state=random_seed, stratify=y)
    labels = list(ALLOWED_LABELS)
    comparisons = {
        name: fit_and_score(name, train_rows, test_rows, labels)
        for name in ("logistic_regression", "extra_trees")
    }
    best_name = max(comparisons.items(), key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]))[0]
    best = comparisons[best_name]
    model = best.pop("model")
    joblib.dump(model, output_path / "model.joblib")
    write_json(build_feature_contract(), output_path / "feature_contract.json")
    write_json({"feature_order": list(FEATURE_COLUMNS), "labels": labels}, output_path / "preprocessing_schema.json")
    return {
        "status": "ok",
        "model_type": best_name,
        "model_feature_columns": list(FEATURE_COLUMNS),
        "allowed_predictions": labels,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
        "performance_by_street": best["performance_by_street"],
        "model_comparison": {name: without_model(value) for name, value in comparisons.items()},
    }


def fit_and_score(name: str, train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    model = make_model(name)
    x_train = [feature_payload(row) for row in train_rows]
    y_train = [str(row["bootstrap_label"]) for row in train_rows]
    x_test = [feature_payload(row) for row in test_rows]
    y_test = [str(row["bootstrap_label"]) for row in test_rows]
    model.fit(x_train, y_train)
    predictions = [str(value) for value in model.predict(x_test)]
    per_street = performance_by_street(test_rows, y_test, predictions, labels)
    return {
        "model": model,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, predictions, labels=list(labels), average="weighted", zero_division=0)), 6),
        "confusion_matrix": matrix_as_dict(y_test, predictions, labels),
        "classification_report": classification_report(y_test, predictions, labels=list(labels), output_dict=True, zero_division=0),
        "performance_by_street": per_street,
    }


def performance_by_street(
    rows: Sequence[Mapping[str, Any]],
    truth: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    streets = sorted({str(row.get("metadata.street") or "UNKNOWN") for row in rows})
    for street in streets:
        indexes = [index for index, row in enumerate(rows) if str(row.get("metadata.street") or "UNKNOWN") == street]
        if not indexes:
            continue
        y_true = [truth[index] for index in indexes]
        y_pred = [predictions[index] for index in indexes]
        result[street] = {
            "rows": len(indexes),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)), 6),
        }
    return result


def make_model(name: str) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=17)
    elif name == "extra_trees":
        classifier = ExtraTreesClassifier(n_estimators=160, random_state=17, class_weight="balanced", n_jobs=-1)
    else:
        raise ValueError(f"unsupported_model:{name}")
    return Pipeline([("vectorizer", DictVectorizer(sparse=False)), ("classifier", classifier)])


def feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for feature in NUMERIC_FEATURES:
        payload[feature] = number_or_zero(row.get(feature))
    for feature in CATEGORICAL_FEATURES:
        payload[feature] = str(row.get(feature) or "UNKNOWN")
    return payload


def predict_first_row(model_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    model = joblib.load(model_dir / "model.joblib")
    features = feature_payload(row)
    prediction = str(model.predict([features])[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = {
            str(label): round(float(probability), 6)
            for label, probability in zip(model.classes_, model.predict_proba([features])[0])
        }
    return {"status": "ok", "prediction": prediction, "probabilities": probabilities}


def write_graphical_study(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    training: Mapping[str, Any],
    random_seed: int,
    fast_mode: bool,
) -> dict[str, str]:
    label_svg = output_path / "eda_label_distribution.svg"
    street_svg = output_path / "eda_street_distribution.svg"
    label_by_street_svg = output_path / "eda_label_by_street.svg"
    confusion_svg = output_path / "confusion_matrix.svg"
    importance_svg = output_path / "feature_importance.svg"
    correlation_svg = output_path / "feature_correlation.svg"
    learning_svg = output_path / "learning_curve.svg"
    study_md = output_path / "graphical_study.md"

    label_counts = Counter(row["bootstrap_label"] for row in rows)
    street_counts = Counter(row["metadata.street"] for row in rows)
    write_bar_chart(label_counts, label_svg, title="PokerBench label distribution")
    write_bar_chart(street_counts, street_svg, title="PokerBench street distribution")
    write_label_by_street(rows, label_by_street_svg)
    write_confusion_matrix_svg(training["confusion_matrix"], confusion_svg)
    write_feature_importance_svg(output_path / "model.joblib", importance_svg)
    write_feature_correlation_svg(rows, correlation_svg, sample_size=12000 if fast_mode else 40000, random_seed=random_seed)
    learning_report = build_learning_curve(rows, random_seed=random_seed, fast_mode=fast_mode)
    write_learning_curve_svg(learning_report, learning_svg)
    study_md.write_text(render_graphical_study(training, rows, learning_report), encoding="utf-8")
    return {
        "eda_label_distribution": str(label_svg),
        "eda_street_distribution": str(street_svg),
        "eda_label_by_street": str(label_by_street_svg),
        "confusion_matrix": str(confusion_svg),
        "feature_importance": str(importance_svg),
        "feature_correlation": str(correlation_svg),
        "learning_curve": str(learning_svg),
        "graphical_study_md": str(study_md),
    }


def write_bar_chart(counts: Mapping[str, int], output_path: Path, *, title: str) -> None:
    items = sorted(counts.items())
    width = 760
    height = 360
    left = 80
    top = 45
    plot_w = 620
    plot_h = 245
    max_value = max([int(value) for _, value in items] or [1])
    bar_w = plot_w / max(1, len(items)) * 0.68
    lines = [svg_header(width, height), svg_text(20, 28, title, size=18)]
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d6d6d6"/>')
    for index, (name, value) in enumerate(items):
        x = left + index * plot_w / max(1, len(items)) + (plot_w / max(1, len(items)) - bar_w) / 2
        h = (float(value) / max_value) * (plot_h - 20)
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#3769a8"/>')
        lines.append(svg_text(x + bar_w / 2, top + plot_h + 22, str(name), size=11, anchor="middle"))
        lines.append(svg_text(x + bar_w / 2, y - 5, str(value), size=10, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_label_by_street(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    streets = sorted({str(row["metadata.street"]) for row in rows})
    matrix = {
        street: Counter(row["bootstrap_label"] for row in rows if row["metadata.street"] == street)
        for street in streets
    }
    cell = 46
    left = 120
    top = 80
    width = left + cell * len(ALLOWED_LABELS) + 60
    height = top + cell * len(streets) + 50
    max_value = max([matrix[street][label] for street in streets for label in ALLOWED_LABELS] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, "Labels by street", size=18)]
    for col, label in enumerate(ALLOWED_LABELS):
        lines.append(svg_text(left + col * cell + cell / 2, top - 14, label, size=10, anchor="middle"))
    for row_index, street in enumerate(streets):
        lines.append(svg_text(left - 12, top + row_index * cell + cell / 2 + 4, street, size=10, anchor="end"))
        for col, label in enumerate(ALLOWED_LABELS):
            value = matrix[street][label]
            fill = blue_scale(value / max_value if max_value else 0)
            x = left + col * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), size=9, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_matrix_svg(matrix: Mapping[str, Mapping[str, int]], output_path: Path) -> None:
    cell = 62
    left = 130
    top = 80
    width = left + cell * len(ALLOWED_LABELS) + 80
    height = top + cell * len(ALLOWED_LABELS) + 70
    max_value = max([int(matrix.get(row, {}).get(col, 0)) for row in ALLOWED_LABELS for col in ALLOWED_LABELS] or [1])
    lines = [svg_header(width, height), svg_text(20, 30, "Confusion matrix", size=18)]
    for col, label in enumerate(ALLOWED_LABELS):
        lines.append(svg_text(left + col * cell + cell / 2, top - 14, f"pred {label}", size=10, anchor="middle"))
    for row_index, label in enumerate(ALLOWED_LABELS):
        lines.append(svg_text(left - 12, top + row_index * cell + cell / 2 + 4, f"true {label}", size=10, anchor="end"))
        for col, pred in enumerate(ALLOWED_LABELS):
            value = int(matrix.get(label, {}).get(pred, 0))
            x = left + col * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{blue_scale(value / max_value if max_value else 0)}" stroke="#ffffff"/>')
            lines.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), size=10, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_importance_svg(model_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    if hasattr(classifier, "feature_importances_"):
        values = [float(value) for value in classifier.feature_importances_]
    elif hasattr(classifier, "coef_"):
        values = [float(value) for value in np.mean(np.abs(classifier.coef_), axis=0)]
    else:
        values = [0.0 for _ in names]
    ranking = sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:18]
    width = 920
    height = 36 + 28 * len(ranking) + 30
    left = 285
    max_value = max([value for _, value in ranking] or [1.0])
    lines = [svg_header(width, height), svg_text(20, 26, "Feature importance", size=18)]
    for index, (name, value) in enumerate(ranking):
        y = 55 + index * 28
        w = 560 * (value / max_value if max_value else 0)
        lines.append(svg_text(left - 10, y + 5, short_label(name, 38), size=10, anchor="end"))
        lines.append(f'<rect x="{left}" y="{y - 10}" width="{w:.2f}" height="16" fill="#3769a8"/>')
        lines.append(svg_text(left + w + 6, y + 4, f"{value:.4f}", size=10))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_correlation_svg(rows: Sequence[Mapping[str, Any]], output_path: Path, *, sample_size: int, random_seed: int) -> None:
    sampled = deterministic_sample(rows, sample_size=sample_size, random_seed=random_seed)
    matrix = np.array([[number_or_zero(row.get(feature)) for feature in NUMERIC_FEATURES] for row in sampled], dtype=float)
    if matrix.shape[0] < 2:
        corr = np.eye(len(NUMERIC_FEATURES))
    else:
        corr = safe_corrcoef(matrix)
    cell = 28
    left = 265
    top = 180
    width = left + cell * len(NUMERIC_FEATURES) + 40
    height = top + cell * len(NUMERIC_FEATURES) + 40
    lines = [svg_header(width, height), svg_text(20, 30, "Numeric feature correlation", size=18), svg_text(20, 52, f"sample rows: {len(sampled)}", size=11)]
    for index, name in enumerate(NUMERIC_FEATURES):
        x = left + index * cell + cell / 2
        y = top + index * cell + cell / 2
        lines.append(svg_text(x, top - 8, short_label(name.replace("features.", ""), 24), size=8, anchor="start", transform=f"rotate(-55 {x} {top - 8})"))
        lines.append(svg_text(left - 8, y + 3, short_label(name.replace("features.", ""), 28), size=8, anchor="end"))
    for row_index, row in enumerate(corr.tolist()):
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{diverging_color(float(value))}" stroke="#ffffff" stroke-width="0.5"/>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_curve(rows: Sequence[Mapping[str, Any]], *, random_seed: int, fast_mode: bool) -> dict[str, Any]:
    sample_size = 9000 if fast_mode else 24000
    sampled = deterministic_sample(rows, sample_size=sample_size, random_seed=random_seed)
    y = [str(row["bootstrap_label"]) for row in sampled]
    train_rows, test_rows = train_test_split(list(sampled), test_size=0.25, random_state=random_seed, stratify=y)
    points = []
    for fraction in (0.15, 0.3, 0.5, 0.75, 1.0):
        subset = stratified_fraction(train_rows, fraction=fraction, random_seed=random_seed)
        model = make_model("extra_trees")
        x_train = [feature_payload(row) for row in subset]
        y_train = [row["bootstrap_label"] for row in subset]
        x_val = [feature_payload(row) for row in test_rows]
        y_val = [row["bootstrap_label"] for row in test_rows]
        model.fit(x_train, y_train)
        train_pred = [str(value) for value in model.predict(x_train)]
        val_pred = [str(value) for value in model.predict(x_val)]
        train_macro = f1_score(y_train, train_pred, labels=list(ALLOWED_LABELS), average="macro", zero_division=0)
        val_macro = f1_score(y_val, val_pred, labels=list(ALLOWED_LABELS), average="macro", zero_division=0)
        points.append(
            {
                "train_size": len(subset),
                "train_macro_f1": round(float(train_macro), 6),
                "validation_macro_f1": round(float(val_macro), 6),
                "gap": round(float(train_macro - val_macro), 6),
            }
        )
    return {"points": points, "diagnostic": learning_diagnostic(points), "sample_rows": len(sampled)}


def write_learning_curve_svg(report: Mapping[str, Any], output_path: Path) -> None:
    points = list(report.get("points") or [])
    width = 760
    height = 360
    left = 70
    top = 45
    plot_w = 610
    plot_h = 245
    lines = [svg_header(width, height), svg_text(20, 28, "Learning curve", size=18), svg_text(20, 348, f"Diagnostic: {report.get('diagnostic')}", size=12)]
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#cccccc"/>')
    for tick in range(6):
        y = top + plot_h - tick * plot_h / 5
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#eeeeee"/>')
        lines.append(svg_text(left - 10, y + 4, f"{tick / 5:.1f}", size=10, anchor="end"))
    if points:
        min_x = min(point["train_size"] for point in points)
        max_x = max(point["train_size"] for point in points)
        add_curve(lines, points, "train_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#2957a4")
        add_curve(lines, points, "validation_macro_f1", min_x, max_x, left, top, plot_w, plot_h, "#c44733")
    lines.append('<circle cx="700" cy="62" r="5" fill="#2957a4"/><text x="711" y="66" font-size="11" font-family="Arial">train</text>')
    lines.append('<circle cx="700" cy="82" r="5" fill="#c44733"/><text x="711" y="86" font-size="11" font-family="Arial">validation</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_graphical_study(training: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], learning: Mapping[str, Any]) -> str:
    label_counts = Counter(row["bootstrap_label"] for row in rows)
    street_counts = Counter(row["metadata.street"] for row in rows)
    live = read_json(LIVE_BB_REPORT)
    return "\n".join(
        [
            "# Graphical Study",
            "",
            "PokerBench is now the primary oracle baseline. The graphs show a large solver-labeled corpus with four native actions.",
            "",
            "## What The Graphs Show",
            "",
            f"- Labels: `{dict(sorted(label_counts.items()))}`.",
            f"- Streets: `{dict(sorted(street_counts.items()))}`.",
            f"- Global accuracy: `{training.get('accuracy')}`, macro F1: `{training.get('macro_f1')}`.",
            f"- Learning diagnostic: `{learning.get('diagnostic')}`.",
            "",
            "## Strengths",
            "",
            "- Solver-oracle labels are separated from legacy/live labels.",
            "- CALL is a native class.",
            "- Raw prompts/cards/history text are audit-only in this first version.",
            "",
            "## Limits",
            "",
            "- CHECK is under-represented compared with FOLD and CALL.",
            "- The current lightweight source is dominated by preflop rows.",
            "- Postflop coverage uses the 10k structured set; the large 500k postflop file is not downloaded by default.",
            "- Some numeric context, especially exact call price and stack, is inferred when the CSV does not expose it directly.",
            "",
            "## Comparison With Live BB Baseline",
            "",
            f"- PokerBench oracle: rows `{len(rows)}`, accuracy `{training.get('accuracy')}`, macro F1 `{training.get('macro_f1')}`.",
            f"- live_bb_baseline_v1: rows `{live.get('rows_used')}`, accuracy `{live.get('accuracy')}`, macro F1 `{live.get('macro_f1')}`.",
            "- The live baseline score is not directly comparable because its labels are legacy/live-derived and include heavy resampling.",
            "",
            "## Recommended Next Action",
            "",
            "Add richer PokerBench feature extraction from action history and bring in more structured postflop rows before considering live integration.",
            "",
        ]
    )


def build_feature_contract() -> dict[str, Any]:
    return {
        "schema": "pokerbench_oracle_baseline_v1",
        "features_model_used": list(FEATURE_COLUMNS),
        "features_audit_only": list(AUDIT_ONLY_COLUMNS),
        "features_leakage_excluded": ["pokerbench.correct_decision_raw", "bootstrap_label", "labels.*", "debug.*", "audit.*"],
        "leakage_columns_used_by_model": [feature for feature in FEATURE_COLUMNS if is_leakage_column(feature)],
        "label_source": "pokerbench_solver_oracle",
        "allowed_predictions": list(ALLOWED_LABELS),
        "call_is_native_class": True,
        "raw_text_direct_features": False,
        "bot_live_connection": "not_modified",
    }


def build_comparison_with_live_bb(training: Mapping[str, Any], *, rows_used: int) -> str:
    live = read_json(LIVE_BB_REPORT)
    lines = [
        "# pokerbench_oracle_baseline_v1 vs live_bb_baseline_v1",
        "",
        "| model | label source | rows | classes | accuracy | macro F1 |",
        "|---|---|---:|---|---:|---:|",
        f"| pokerbench_oracle_baseline_v1 | solver oracle | {rows_used} | CHECK/FOLD/CALL/RAISE | {training['accuracy']} | {training['macro_f1']} |",
        f"| live_bb_baseline_v1 | legacy/live + limited solver | {live.get('rows_used')} | CHECK/FOLD/CALL/RAISE | {live.get('accuracy')} | {live.get('macro_f1')} |",
        "",
        "- PokerBench labels are kept separate from legacy labels.",
        "- Neither model is connected to the live bot.",
        "",
    ]
    return "\n".join(lines)


def preflop_postflop_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter("preflop" if row.get("metadata.street") == "PREFLOP" else "postflop" for row in rows)
    return dict(sorted(counts.items()))


def matrix_as_dict(truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predictions, labels=list(labels))
    return {
        expected: {predicted: int(matrix[row_index][col_index]) for col_index, predicted in enumerate(labels)}
        for row_index, expected in enumerate(labels)
    }


def deterministic_sample(rows: Sequence[Mapping[str, Any]], *, sample_size: int, random_seed: int) -> list[Mapping[str, Any]]:
    if len(rows) <= sample_size:
        return list(rows)
    rng = np.random.default_rng(random_seed)
    indexes = rng.choice(len(rows), size=sample_size, replace=False)
    return [rows[int(index)] for index in indexes]


def stratified_fraction(rows: Sequence[Mapping[str, Any]], *, fraction: float, random_seed: int) -> list[Mapping[str, Any]]:
    if fraction >= 1.0:
        return list(rows)
    rng = np.random.default_rng(random_seed + int(fraction * 1000))
    selected = []
    for label in ALLOWED_LABELS:
        label_rows = [row for row in rows if row["bootstrap_label"] == label]
        if not label_rows:
            continue
        count = max(1, int(round(len(label_rows) * fraction)))
        indexes = rng.choice(len(label_rows), size=min(count, len(label_rows)), replace=False)
        selected.extend(label_rows[int(index)] for index in indexes)
    return selected


def safe_corrcoef(matrix: np.ndarray) -> np.ndarray:
    feature_count = matrix.shape[1]
    corr = np.eye(feature_count)
    std = matrix.std(axis=0)
    variable_indexes = np.where(std > 0)[0]
    if len(variable_indexes) < 2:
        return corr
    variable_corr = np.corrcoef(matrix[:, variable_indexes], rowvar=False)
    variable_corr = np.atleast_2d(np.nan_to_num(variable_corr, nan=0.0, posinf=0.0, neginf=0.0))
    for row_pos, row_index in enumerate(variable_indexes):
        for col_pos, col_index in enumerate(variable_indexes):
            corr[int(row_index), int(col_index)] = float(variable_corr[row_pos, col_pos])
    return corr


def learning_diagnostic(points: Sequence[Mapping[str, Any]]) -> str:
    if not points:
        return "insufficient_data"
    last = points[-1]
    train = float(last.get("train_macro_f1") or 0.0)
    validation = float(last.get("validation_macro_f1") or 0.0)
    gap = train - validation
    if train > 0.9 and gap > 0.12:
        return "overfit_probable"
    if train < 0.6 and validation < 0.6:
        return "underfit_probable"
    if validation < 0.75:
        return "dataset_difficult_or_features_underpowered"
    return "correct"


def add_curve(
    lines: list[str],
    points: Sequence[Mapping[str, Any]],
    key: str,
    min_x: int,
    max_x: int,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    color: str,
) -> None:
    coords = []
    for point in points:
        x = left + (0.5 * plot_w if max_x == min_x else (point["train_size"] - min_x) / (max_x - min_x) * plot_w)
        y = top + plot_h - max(0.0, min(1.0, float(point[key]))) * plot_h
        coords.append(f"{x:.2f},{y:.2f}")
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", transform: str | None = None) -> str:
    transform_attr = "" if transform is None else f' transform="{transform}"'
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-family="Arial"{transform_attr}>{escape_xml(text)}</text>'


def escape_xml(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short_label(value: str, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def blue_scale(value: float) -> str:
    ratio = max(0.0, min(1.0, value))
    red = int(238 - 160 * ratio)
    green = int(244 - 110 * ratio)
    blue = int(250 - 40 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def diverging_color(value: float) -> str:
    safe = max(-1.0, min(1.0, value))
    if safe >= 0:
        red = int(245 - 35 * safe)
        green = int(247 - 130 * safe)
        blue = int(250 - 175 * safe)
    else:
        ratio = abs(safe)
        red = int(245 - 170 * ratio)
        green = int(247 - 95 * ratio)
        blue = int(250 - 20 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_candidates_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "snapshot_id",
        *FEATURE_COLUMNS,
        *AUDIT_ONLY_COLUMNS,
        "metadata.label_source",
        "bootstrap_label",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_leakage_column(column: str) -> bool:
    return column.startswith(LEAKAGE_PREFIXES) or column in {"bootstrap_label", "metadata.label_source"}


def without_model(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "model"}


def number_or_zero(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--full-diagnostics", action="store_true", help="Use larger samples for graphical diagnostics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pokerbench_oracle_baseline_v1(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        download=not args.no_download,
        max_rows=args.max_rows,
        random_seed=args.random_seed,
        fast_mode=not args.full_diagnostics,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
