from __future__ import annotations

from pathlib import Path

import pytest

from solvers import poker_solver_adapter as adapter


REQUIRED_KEYS = {"status", "solver_name", "input", "output", "error", "duration_ms"}


def assert_stable_result(result: dict) -> None:
    assert set(result) == REQUIRED_KEYS
    assert result["status"] in {"ok", "failed"}
    assert result["solver_name"] == "PokerSolver"
    assert isinstance(result["input"], dict)
    assert result["output"] is None or isinstance(result["output"], dict)
    assert result["error"] is None or isinstance(result["error"], str)
    assert isinstance(result["duration_ms"], float)


def test_solver_absent_returns_failed(tmp_path: Path) -> None:
    result = adapter.check_solver_available(solver_path=tmp_path)
    assert_stable_result(result)
    assert result["status"] == "failed"
    assert "poker_solver_package_not_found" in result["error"]


def test_invalid_path_returns_failed(tmp_path: Path) -> None:
    result = adapter.check_solver_available(solver_path=tmp_path / "missing")
    assert_stable_result(result)
    assert result["status"] == "failed"
    assert "solver_path_not_found" in result["error"]


def test_missing_psutil_returns_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "poker_solver"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'fake'\n", encoding="utf-8")

    original_find_spec = adapter.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "psutil":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib.util, "find_spec", fake_find_spec)
    result = adapter.check_solver_available(solver_path=tmp_path)
    assert_stable_result(result)
    assert result["status"] == "failed"
    assert result["error"] == "missing_dependency:psutil"


def test_invalid_equity_input_returns_failed_without_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSolver:
        __version__ = "fake"

        @staticmethod
        def parse_hand(value):
            raise ValueError(f"invalid hand:{value}")

        @staticmethod
        def parse_board(value):
            return []

        @staticmethod
        def equity(*args, **kwargs):
            return []

    monkeypatch.setattr(adapter, "_load_poker_solver", lambda solver_path=None: (FakeSolver(), None, None))
    result = adapter.compute_equity_hand_vs_hand("bad", "QdQc", iterations=10)
    assert_stable_result(result)
    assert result["status"] == "failed"
    assert "ValueError:invalid hand" in result["error"]


def test_all_public_functions_have_stable_format_when_solver_path_is_invalid(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    results = [
        adapter.check_solver_available(solver_path=missing),
        adapter.compute_equity_hand_vs_hand("AhKh", "QdQc", solver_path=missing),
        adapter.compute_equity_hand_vs_range("AhKh", "AA,KK", solver_path=missing),
        adapter.solve_simple_postflop_spot("AhKh", "QdQc", board="2h7h9d", pot=10, solver_path=missing),
        adapter.solve_tiny_postflop_spot("AhKh", "QdQc", board="2h7h9d", pot=10, solver_path=missing),
    ]
    for result in results:
        assert_stable_result(result)
        assert result["status"] == "failed"


def test_real_import_if_dependencies_are_available() -> None:
    result = adapter.check_solver_available()
    assert_stable_result(result)
    if result["status"] == "failed":
        pytest.skip(f"PokerSolver is not importable in this environment: {result['error']}")
    assert result["output"]["available"] is True
    assert result["output"]["functions"]["equity"] is True


def test_real_rust_backend_available_if_solver_imports() -> None:
    result = adapter.check_solver_available()
    assert_stable_result(result)
    if result["status"] == "failed":
        pytest.skip(f"PokerSolver is not importable in this environment: {result['error']}")

    assert result["output"]["available"] is True
    assert result["output"]["rust_backend_available"] is True, result["output"]["rust_backend_error"]


def test_real_equity_hand_vs_hand_if_solver_imports() -> None:
    availability = adapter.check_solver_available()
    if availability["status"] == "failed":
        pytest.skip(f"PokerSolver is not importable in this environment: {availability['error']}")

    result = adapter.compute_equity_hand_vs_hand(
        "AhAd",
        "KsKc",
        iterations=500,
        seed=7,
    )
    assert_stable_result(result)
    assert result["status"] == "ok", result["error"]
    assert 0.0 <= result["output"]["hero_equity"] <= 1.0
    assert 0.0 <= result["output"]["villain_equity"] <= 1.0


def test_rust_backend_unavailable_returns_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSolver:
        @staticmethod
        def parse_hand(value):
            return value

        @staticmethod
        def parse_board(value):
            return value

        class HUNLConfig:
            pass

        class HUNLPoker:
            pass

        class Street:
            FLOP = "flop"
            TURN = "turn"
            RIVER = "river"

        @staticmethod
        def solve(*args, **kwargs):
            raise AssertionError("solve should not run when rust backend is unavailable")

    monkeypatch.setattr(adapter, "_load_poker_solver", lambda solver_path=None: (FakeSolver(), None, None))
    monkeypatch.setattr(adapter, "_rust_backend_status", lambda: (False, "ModuleNotFoundError:boom"))

    result = adapter.solve_simple_postflop_spot(
        "AhKh",
        "QdQc",
        board="As 7c 2d Kh 5s",
        pot=10,
        iterations=2,
        backend="rust",
    )

    assert_stable_result(result)
    assert result["status"] == "failed"
    assert result["error"] == "rust_backend_unavailable:ModuleNotFoundError:boom"


def test_tiny_rust_backend_unavailable_returns_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "_rust_backend_status", lambda: (False, "ModuleNotFoundError:boom"))

    result = adapter.solve_tiny_postflop_spot(
        "AhKh",
        "QdQc",
        board="As 7c 2d Kh 5s",
        pot=10,
        iterations=10,
        backend="rust",
        timeout_s=1,
    )

    assert_stable_result(result)
    assert result["status"] == "failed"
    assert "rust_backend_unavailable:ModuleNotFoundError:boom" in result["error"]


def test_tiny_invalid_input_returns_failed() -> None:
    result = adapter.solve_tiny_postflop_spot(
        "AhKh",
        "QdQc",
        board="As 7c",
        pot=10,
        iterations=10,
        backend="rust",
        timeout_s=1,
    )

    assert_stable_result(result)
    assert result["status"] == "failed"
    assert result["error"] is not None


def test_real_simple_postflop_rust_solve_if_backend_imports() -> None:
    availability = adapter.check_solver_available()
    if availability["status"] == "failed":
        pytest.skip(f"PokerSolver is not importable in this environment: {availability['error']}")
    if not availability["output"]["rust_backend_available"]:
        pytest.skip(f"PokerSolver Rust backend is unavailable: {availability['output']['rust_backend_error']}")

    result = adapter.solve_simple_postflop_spot(
        "AhKh",
        "QdQc",
        board="As 7c 2d Kh 5s",
        pot=10,
        stack=100,
        iterations=2,
        backend="rust",
    )

    assert_stable_result(result)
    assert result["status"] == "ok", result["error"]
    assert result["output"]["backend"] == "rust"
    assert result["output"]["iterations"] == 2
    assert isinstance(result["output"]["strategy_entry_count"], int)


def test_real_tiny_postflop_rust_solve_if_backend_imports() -> None:
    availability = adapter.check_solver_available()
    if availability["status"] == "failed":
        pytest.skip(f"PokerSolver is not importable in this environment: {availability['error']}")
    if not availability["output"]["rust_backend_available"]:
        pytest.skip(f"PokerSolver Rust backend is unavailable: {availability['output']['rust_backend_error']}")

    result = adapter.solve_tiny_postflop_spot(
        "AhKh",
        "QdQc",
        board="As 7c 2d Kh 5s",
        pot=10,
        stack=100,
        bet_sizes=(0.33,),
        iterations=10,
        backend="rust",
        timeout_s=5,
    )

    assert_stable_result(result)
    assert result["status"] == "ok", result["error"]
    assert result["output"]["backend"] == "rust"
    assert result["output"]["iterations"] == 10
    assert isinstance(result["output"]["game_value"], float)
    assert isinstance(result["output"]["strategy_entry_count"], int)
