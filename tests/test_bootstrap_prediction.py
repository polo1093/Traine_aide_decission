from __future__ import annotations

import csv
from pathlib import Path

from models.predict_bootstrap_model import predict_bootstrap_model
from models.train_bootstrap_model import train_bootstrap_model


FIELDNAMES = [
    "source_id",
    "street",
    "hero_cards",
    "villain_hand",
    "board_cards",
    "pot",
    "to_call",
    "stack",
    "spr",
    "dominant_action_frequency",
    "iterations",
    "exploitability_last",
    "candidate_confidence",
    "position_model",
    "decision_context_type",
    "label_source",
    "label_quality",
    "board_card_count",
    "is_river",
    "is_turn",
    "is_check_or_bet_context",
    "is_facing_bet_context",
    "to_call_ratio",
    "stack_to_pot_ratio",
    "bootstrap_label",
    "excluded",
]


def candidate_row(index: int, label: str) -> dict:
    is_raise = label == "RAISE"
    is_fold = label == "FOLD"
    return {
        "source_id": f"row_{index}",
        "street": "RIVER",
        "hero_cards": '["Ah","As"]' if is_raise else "null",
        "villain_hand": "null",
        "board_cards": '["Ac","Kd","7s","2h","2c"]' if is_raise else "null",
        "pot": 250 + index,
        "to_call": 150 if is_fold else 0,
        "stack": 1200,
        "spr": 4.8,
        "dominant_action_frequency": 0.9,
        "iterations": 25,
        "exploitability_last": 0.5,
        "candidate_confidence": "high",
        "position_model": "IP" if is_fold else "OOP",
        "decision_context_type": "hero_facing_bet" if is_fold else "hero_check_or_bet",
        "label_source": "weak_rule_bootstrap" if is_raise else "solver_candidate",
        "label_quality": "bootstrap_weak_rule_untrusted" if is_raise else "bootstrap_solver_untrusted",
        "board_card_count": 5 if is_raise else 0,
        "is_river": True,
        "is_turn": False,
        "is_check_or_bet_context": not is_fold,
        "is_facing_bet_context": is_fold,
        "to_call_ratio": 0.5 if is_fold else 0,
        "stack_to_pot_ratio": 4.8,
        "bootstrap_label": label,
        "excluded": False,
    }


def write_dataset(path: Path) -> None:
    rows = [candidate_row(i, "CHECK") for i in range(12)]
    rows += [candidate_row(i + 12, "FOLD") for i in range(12)]
    rows += [candidate_row(i + 24, "RAISE") for i in range(12)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def trained_model_dir(tmp_path: Path) -> Path:
    input_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "model"
    write_dataset(input_path)
    train_bootstrap_model(input_path=input_path, output_dir=output_dir, model_type="auto", min_rows=10)
    return output_dir


def valid_payload() -> dict:
    return {
        "pot": 100,
        "to_call": 0,
        "stack": 1000,
        "spr": 10,
        "dominant_action_frequency": 0.8,
        "iterations": 25,
        "exploitability_last": 0.5,
        "board_card_count": 5,
        "is_river": True,
        "is_turn": False,
        "is_check_or_bet_context": True,
        "is_facing_bet_context": False,
        "to_call_ratio": 0,
        "stack_to_pot_ratio": 10,
        "street": "RIVER",
        "position_model": "OOP",
        "decision_context_type": "hero_check_or_bet",
        "label_source": "weak_rule_bootstrap",
        "label_quality": "bootstrap_weak_rule_untrusted",
        "candidate_confidence": "high",
    }


def test_prediction_loads_model_and_returns_known_class(tmp_path: Path) -> None:
    model_dir = trained_model_dir(tmp_path)

    result = predict_bootstrap_model(model_dir=model_dir, input_payload=valid_payload())

    assert result["status"] == "ok"
    assert result["prediction"] in {"CHECK", "FOLD", "RAISE"}
    assert "pipeline_smoke_only_not_for_production" in result["warnings"]


def test_prediction_returns_probabilities_when_available(tmp_path: Path) -> None:
    model_dir = trained_model_dir(tmp_path)

    result = predict_bootstrap_model(model_dir=model_dir, input_payload=valid_payload())

    assert result["probabilities"]
    assert set(result["probabilities"]).issubset({"CHECK", "FOLD", "RAISE"})


def test_missing_field_returns_clean_error(tmp_path: Path) -> None:
    model_dir = trained_model_dir(tmp_path)
    payload = valid_payload()
    payload.pop("pot")

    result = predict_bootstrap_model(model_dir=model_dir, input_payload=payload)

    assert result["status"] == "failed"
    assert result["error"] == "missing_field:pot"


def test_unknown_category_is_warned_but_handled(tmp_path: Path) -> None:
    model_dir = trained_model_dir(tmp_path)
    payload = valid_payload()
    payload["position_model"] = "BTN"

    result = predict_bootstrap_model(model_dir=model_dir, input_payload=payload)

    assert result["status"] == "ok"
    assert any(warning.startswith("unknown_category:position_model:BTN") for warning in result["warnings"])


def test_missing_model_returns_clean_error(tmp_path: Path) -> None:
    result = predict_bootstrap_model(model_dir=tmp_path / "missing", input_payload=valid_payload())

    assert result["status"] == "failed"
    assert result["error"] == "model_missing"
