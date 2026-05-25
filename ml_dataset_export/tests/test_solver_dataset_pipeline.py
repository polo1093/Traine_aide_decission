from __future__ import annotations

import json
from pathlib import Path

import pytest

from solver_adapter import MockSolverAdapter, SolverDecision, SolverSpot, normalize_solver_action
from solver_dataset_validator import validate_jsonl
from solver_dataset_writer import DatasetRowRejected, build_dataset_row, write_solver_dataset
from solver_spot_generator import generate_synthetic_spots
import solver_dataset_writer


def test_example_dataset_is_readable() -> None:
    rows = [json.loads(line) for line in Path("example_training_dataset.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert rows
    assert rows[0]["type"] == "ml_decision_snapshot"


def test_generated_schema_matches_example(tmp_path: Path) -> None:
    output = tmp_path / "solver.jsonl"
    spots = generate_synthetic_spots(5, seed=42)
    result = write_solver_dataset(spots, MockSolverAdapter(), output, n_rows=5)
    assert result["written"] == 5
    validation = validate_jsonl(output)
    assert validation.ok, validation.errors


def test_bet_is_normalized_to_raise() -> None:
    assert normalize_solver_action("BET") == "RAISE"


def test_wait_is_rejected() -> None:
    spot = generate_synthetic_spots(1, seed=1)[0]
    with pytest.raises(DatasetRowRejected):
        build_dataset_row(spot, SolverDecision("WAIT"), index=1)


def test_invalid_cards_are_rejected() -> None:
    spot = SolverSpot(
        hero_cards=["AS", "XX"],
        board_cards=[],
        street="PREFLOP",
        hero_position="BB",
        player_active=1,
        player_start=1,
        pot=100.0,
        to_call=0.0,
        buttons=[{"index": 0, "enabled": True, "state": "check", "value": 0.0, "text": "check", "confidence": None}],
        buttons_active=["check"],
    )
    with pytest.raises(DatasetRowRejected):
        build_dataset_row(spot, SolverDecision("CHECK"), index=1)


def test_impossible_action_is_rejected() -> None:
    spot = generate_synthetic_spots(1, seed=2)[0]
    spot = SolverSpot(**{**spot.__dict__, "buttons_active": ["check"], "buttons": [{"index": 0, "enabled": True, "state": "check", "value": 0.0, "text": "check", "confidence": None}]})
    with pytest.raises(DatasetRowRejected):
        build_dataset_row(spot, SolverDecision("CALL"), index=1)


def test_existing_equity_wrapper_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}
    original = solver_dataset_writer.feature_rebuilder.rebuild_derived_features

    def wrapper(*args, **kwargs):
        called["value"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_dataset_writer.feature_rebuilder, "rebuild_derived_features", wrapper)
    spot = generate_synthetic_spots(1, seed=3)[0]
    build_dataset_row(spot, MockSolverAdapter().solve_spot(spot), index=1)
    assert called["value"] is True


def test_debug_and_labels_are_not_features() -> None:
    spot = generate_synthetic_spots(1, seed=4)[0]
    row = build_dataset_row(spot, MockSolverAdapter().solve_spot(spot), index=1)
    assert "debug" not in row["features"]
    assert "labels" not in row["features"]
    assert "solver_action" not in row["features"]


def test_writer_outputs_valid_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "solver.jsonl"
    write_solver_dataset(generate_synthetic_spots(3, seed=5), MockSolverAdapter(), output, n_rows=3)
    for line in output.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["labels"]["legacy_action"] in {"FOLD", "CHECK", "CALL", "RAISE"}
