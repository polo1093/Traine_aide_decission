"""Defensive adapter around the local PokerSolver checkout.

The adapter keeps PokerSolver imports behind one stable boundary. It resolves
the local checkout path, temporarily adds it to ``sys.path`` only while
importing, and always returns a stable dict instead of leaking raw exceptions.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SOLVER_NAME = "PokerSolver"
DEFAULT_SOLVER_RELATIVE_PATH = Path("projet importer") / "poker_solver-main"
REQUIRED_RESULT_KEYS = ("status", "solver_name", "input", "output", "error", "duration_ms")
DEFAULT_TINY_ITERATIONS = 10
MAX_TINY_ITERATIONS = 100
DEFAULT_TINY_TIMEOUT_S = 5.0


def check_solver_available(solver_path: str | Path | None = None) -> dict[str, Any]:
    """Return availability details for the local PokerSolver package."""

    started = time.perf_counter()
    input_payload = {"solver_path": _input_path_value(solver_path)}
    try:
        solver, error, resolved_path = _load_poker_solver(solver_path)
        input_payload["resolved_path"] = str(resolved_path) if resolved_path else None
        if error is not None:
            return _result(started, input_payload, None, error)

        rust_available, rust_error = _rust_backend_status()
        output = {
            "available": True,
            "version": getattr(solver, "__version__", None),
            "module_file": getattr(solver, "__file__", None),
            "solver_path": str(resolved_path) if resolved_path else None,
            "rust_backend_available": rust_available,
            "rust_backend_error": rust_error,
            "functions": _available_functions(solver),
        }
        return _result(started, input_payload, output, None)
    except Exception as exc:  # noqa: BLE001 - public boundary must not leak
        return _result(started, input_payload, None, _format_error(exc))


def compute_equity_hand_vs_hand(
    hero_hand: Any,
    villain_hand: Any,
    *,
    board: Any = None,
    iterations: int = 25_000,
    seed: int | None = None,
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute equity for one concrete Hold'em hand against another."""

    started = time.perf_counter()
    input_payload = {
        "hero_hand": hero_hand,
        "villain_hand": villain_hand,
        "board": board,
        "iterations": iterations,
        "seed": seed,
        "solver_path": _input_path_value(solver_path),
    }
    try:
        solver, error, _ = _load_poker_solver(solver_path)
        if error is not None:
            return _result(started, input_payload, None, error)
        missing = _missing_functions(solver, ("parse_hand", "parse_board", "equity"))
        if missing:
            return _result(started, input_payload, None, f"missing_solver_function:{','.join(missing)}")

        hero = solver.parse_hand(_cards_to_solver_string(hero_hand))
        villain = solver.parse_hand(_cards_to_solver_string(villain_hand))
        parsed_board = solver.parse_board(_cards_to_solver_string(board, allow_empty=True))
        rng = random.Random(seed) if seed is not None else None
        results = solver.equity([hero, villain], board=parsed_board, iterations=int(iterations), rng=rng)
        return _result(
            started,
            input_payload,
            {
                "hero": _equity_result_to_dict(results[0]),
                "villain": _equity_result_to_dict(results[1]),
                "hero_equity": float(results[0].equity),
                "villain_equity": float(results[1].equity),
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(started, input_payload, None, _format_error(exc))


def compute_equity_hand_vs_range(
    hero_hand: Any,
    villain_range: str,
    *,
    board: Any = None,
    iterations: int = 25_000,
    seed: int | None = None,
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute equity for one concrete hand against a PokerSolver range spec."""

    started = time.perf_counter()
    input_payload = {
        "hero_hand": hero_hand,
        "villain_range": villain_range,
        "board": board,
        "iterations": iterations,
        "seed": seed,
        "solver_path": _input_path_value(solver_path),
    }
    try:
        solver, error, _ = _load_poker_solver(solver_path)
        if error is not None:
            return _result(started, input_payload, None, error)
        missing = _missing_functions(solver, ("parse_hand", "parse_board", "parse_range", "equity"))
        if missing:
            return _result(started, input_payload, None, f"missing_solver_function:{','.join(missing)}")

        hero = solver.parse_hand(_cards_to_solver_string(hero_hand))
        villain = solver.parse_range(str(villain_range))
        parsed_board = solver.parse_board(_cards_to_solver_string(board, allow_empty=True))
        rng = random.Random(seed) if seed is not None else None
        results = solver.equity([hero, villain], board=parsed_board, iterations=int(iterations), rng=rng)
        return _result(
            started,
            input_payload,
            {
                "hero": _equity_result_to_dict(results[0]),
                "villain_range": _equity_result_to_dict(results[1]),
                "hero_equity": float(results[0].equity),
                "villain_range_equity": float(results[1].equity),
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(started, input_payload, None, _format_error(exc))


def solve_simple_postflop_spot(
    hero_hand: Any,
    villain_hand: Any,
    *,
    board: Any,
    pot: float,
    to_call: float = 0.0,
    stack: float = 100.0,
    iterations: int = 100,
    backend: str = "python",
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Best-effort tiny concrete HUNL postflop solve.

    This is intentionally conservative. It is useful only as a smoke test for
    the solver boundary, not as a production-grade poker decision.
    """

    started = time.perf_counter()
    input_payload = {
        "hero_hand": hero_hand,
        "villain_hand": villain_hand,
        "board": board,
        "pot": pot,
        "to_call": to_call,
        "stack": stack,
        "iterations": iterations,
        "backend": backend,
        "solver_path": _input_path_value(solver_path),
    }
    try:
        solver, error, _ = _load_poker_solver(solver_path)
        if error is not None:
            return _result(started, input_payload, None, error)
        missing = _missing_functions(
            solver,
            ("parse_hand", "parse_board", "HUNLConfig", "HUNLPoker", "Street", "solve"),
        )
        if missing:
            return _result(started, input_payload, None, f"missing_solver_function:{','.join(missing)}")
        if backend == "rust":
            rust_available, rust_error = _rust_backend_status()
            if not rust_available:
                return _result(started, input_payload, None, f"rust_backend_unavailable:{rust_error}")
        elif backend != "python":
            return _result(started, input_payload, None, f"unsupported_backend:{backend}")

        hero = tuple(solver.parse_hand(_cards_to_solver_string(hero_hand)))
        villain = tuple(solver.parse_hand(_cards_to_solver_string(villain_hand)))
        parsed_board = tuple(solver.parse_board(_cards_to_solver_string(board)))
        street = _street_from_board_length(solver, len(parsed_board))
        if street is None:
            return _result(started, input_payload, None, "unsupported_board_length_for_postflop")

        pot_cents = _bb_to_cents(pot)
        to_call_cents = _bb_to_cents(to_call)
        stack_cents = max(_bb_to_cents(stack), pot_cents, 100)
        contributions = _initial_contributions(pot_cents, to_call_cents)
        config = solver.HUNLConfig(
            starting_stack=stack_cents,
            starting_street=street,
            initial_board=parsed_board,
            initial_pot=pot_cents,
            initial_contributions=contributions,
            initial_hole_cards=(hero, villain),
        )
        solved = solver.solve(solver.HUNLPoker(config), iterations=int(iterations), backend=backend)
        output = {
            "backend": getattr(solved, "backend", backend),
            "iterations": getattr(solved, "iterations", None),
            "game_value": getattr(solved, "game_value", None),
            "exploitability_history": list(getattr(solved, "exploitability_history", []) or []),
            "strategy_entry_count": len(getattr(solved, "average_strategy", {}) or {}),
        }
        return _result(started, input_payload, output, None)
    except Exception as exc:  # noqa: BLE001
        return _result(started, input_payload, None, _format_error(exc))


def solve_tiny_postflop_spot(
    hero_hand: Any,
    villain_hand: Any | None = None,
    *,
    villain_range: str | None = None,
    board: Any,
    pot: float,
    stack: float = 100.0,
    bet_sizes: Any = (0.33,),
    iterations: int = DEFAULT_TINY_ITERATIONS,
    backend: str = "rust",
    timeout_s: float | None = DEFAULT_TINY_TIMEOUT_S,
    solver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a bounded concrete postflop smoke solve through PokerSolver.

    The function validates that the heavy solver can be called and return a
    stable payload. It is not a label-generation API and the low default
    iteration count is intentionally not strategy-quality.
    """

    started = time.perf_counter()
    input_payload = {
        "hero_hand": hero_hand,
        "villain_hand": villain_hand,
        "villain_range": villain_range,
        "board": board,
        "pot": pot,
        "stack": stack,
        "bet_sizes": bet_sizes,
        "iterations": iterations,
        "backend": backend,
        "timeout_s": timeout_s,
        "solver_path": _input_path_value(solver_path),
    }

    try:
        timeout_value = _validate_timeout(timeout_s)
        if timeout_value is None:
            output = _solve_tiny_postflop_spot_output(
                hero_hand=hero_hand,
                villain_hand=villain_hand,
                villain_range=villain_range,
                board=board,
                pot=pot,
                stack=stack,
                bet_sizes=bet_sizes,
                iterations=iterations,
                backend=backend,
                solver_path=solver_path,
            )
            return _result(started, input_payload, output, None)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            _solve_tiny_postflop_spot_output,
            hero_hand=hero_hand,
            villain_hand=villain_hand,
            villain_range=villain_range,
            board=board,
            pot=pot,
            stack=stack,
            bet_sizes=bet_sizes,
            iterations=iterations,
            backend=backend,
            solver_path=solver_path,
        )
        try:
            output = future.result(timeout=timeout_value)
            return _result(started, input_payload, output, None)
        except TimeoutError:
            future.cancel()
            return _result(started, input_payload, None, f"solver_timeout:{timeout_value}s")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:  # noqa: BLE001
        return _result(started, input_payload, None, _format_error(exc))


def _solve_tiny_postflop_spot_output(
    *,
    hero_hand: Any,
    villain_hand: Any | None,
    villain_range: str | None,
    board: Any,
    pot: float,
    stack: float,
    bet_sizes: Any,
    iterations: int,
    backend: str,
    solver_path: str | Path | None,
) -> dict[str, Any]:
    if villain_hand is None:
        if villain_range is not None:
            raise ValueError("villain_range_not_supported_for_tiny_postflop_smoke")
        raise ValueError("villain_hand is required")
    if villain_range is not None:
        raise ValueError("villain_range_not_supported_for_tiny_postflop_smoke")

    iterations_int = _validate_tiny_iterations(iterations)
    bet_size_fractions = _validate_bet_sizes(bet_sizes)
    if backend == "rust":
        rust_available, rust_error = _rust_backend_status()
        if not rust_available:
            raise RuntimeError(f"rust_backend_unavailable:{rust_error}")
    elif backend != "python":
        raise ValueError(f"unsupported_backend:{backend}")

    solver, error, _ = _load_poker_solver(solver_path)
    if error is not None:
        raise RuntimeError(error)
    missing = _missing_functions(
        solver,
        ("parse_hand", "parse_board", "HUNLConfig", "HUNLPoker", "Street", "solve"),
    )
    if missing:
        raise RuntimeError(f"missing_solver_function:{','.join(missing)}")

    hero = tuple(solver.parse_hand(_cards_to_solver_string(hero_hand)))
    villain = tuple(solver.parse_hand(_cards_to_solver_string(villain_hand)))
    parsed_board = tuple(solver.parse_board(_cards_to_solver_string(board)))
    street = _street_from_board_length(solver, len(parsed_board))
    if street is None:
        raise ValueError("unsupported_board_length_for_postflop")

    pot_cents = _bb_to_cents(pot)
    stack_cents = max(_bb_to_cents(stack), pot_cents, 100)
    config = solver.HUNLConfig(
        starting_stack=stack_cents,
        starting_street=street,
        initial_board=parsed_board,
        initial_pot=pot_cents,
        initial_contributions=_initial_contributions(pot_cents, 0),
        initial_hole_cards=(hero, villain),
        bet_size_fractions=bet_size_fractions,
    )
    game = solver.HUNLPoker(config)
    solved = solver.solve(game, iterations=iterations_int, backend=backend)
    root_strategy_raw, root_strategy_error = _extract_root_strategy_raw(
        game=game,
        solved=solved,
        bet_size_fractions=bet_size_fractions,
    )
    return {
        "backend": getattr(solved, "backend", backend),
        "iterations": getattr(solved, "iterations", None),
        "game_value": getattr(solved, "game_value", None),
        "exploitability_history": list(getattr(solved, "exploitability_history", []) or []),
        "strategy_entry_count": len(getattr(solved, "average_strategy", {}) or {}),
        "root_strategy_raw": root_strategy_raw,
        "root_strategy_error": root_strategy_error,
    }


def _load_poker_solver(solver_path: str | Path | None = None) -> tuple[Any | None, str | None, Path | None]:
    resolved_path = _resolve_solver_path(solver_path)
    if not resolved_path.exists():
        return None, f"solver_path_not_found:{resolved_path}", resolved_path
    if not (resolved_path / "poker_solver" / "__init__.py").exists():
        return None, f"poker_solver_package_not_found:{resolved_path}", resolved_path

    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is None:
        return None, "missing_dependency:psutil", resolved_path

    try:
        with _temporary_sys_path(resolved_path):
            module = importlib.import_module("poker_solver")
        return module, None, resolved_path
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name == "psutil":
            return None, "missing_dependency:psutil", resolved_path
        if missing_name == "poker_solver":
            return None, "poker_solver_not_importable", resolved_path
        return None, f"missing_dependency:{missing_name}", resolved_path
    except ImportError as exc:
        return None, _format_error(exc), resolved_path
    except Exception as exc:  # noqa: BLE001
        return None, _format_error(exc), resolved_path


def _resolve_solver_path(solver_path: str | Path | None) -> Path:
    if solver_path is not None:
        return Path(solver_path).expanduser().resolve()
    env_path = os.environ.get("POKER_SOLVER_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / DEFAULT_SOLVER_RELATIVE_PATH).resolve()


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_text = str(path)
    inserted = False
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            with contextlib_suppress_value_error():
                sys.path.remove(path_text)


@contextmanager
def contextlib_suppress_value_error() -> Iterator[None]:
    try:
        yield
    except ValueError:
        pass


def _rust_backend_status() -> tuple[bool, str | None]:
    try:
        importlib.import_module("poker_solver._rust")
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, _format_error(exc)


def _available_functions(solver: Any) -> dict[str, bool]:
    names = (
        "equity",
        "parse_hand",
        "parse_board",
        "parse_range",
        "HUNLConfig",
        "HUNLPoker",
        "Street",
        "solve",
        "solve_hunl_postflop",
        "solve_range_vs_range",
    )
    return {name: hasattr(solver, name) for name in names}


def _missing_functions(solver: Any, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not hasattr(solver, name)]


def _street_from_board_length(solver: Any, board_count: int) -> Any | None:
    if board_count == 3:
        return solver.Street.FLOP
    if board_count == 4:
        return solver.Street.TURN
    if board_count == 5:
        return solver.Street.RIVER
    return None


def _initial_contributions(pot_cents: int, to_call_cents: int) -> tuple[int, int]:
    if pot_cents <= 0:
        return (0, 0)
    if to_call_cents <= 0:
        left = pot_cents // 2
        return (left, pot_cents - left)
    lower = max(0, (pot_cents - to_call_cents) // 2)
    return (lower, pot_cents - lower)


def _bb_to_cents(value: Any) -> int:
    return max(0, int(round(float(value) * 100)))


def _validate_tiny_iterations(value: Any) -> int:
    iterations = int(value)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if iterations > MAX_TINY_ITERATIONS:
        raise ValueError(f"iterations_exceed_tiny_limit:{MAX_TINY_ITERATIONS}")
    return iterations


def _validate_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    timeout_value = float(value)
    if timeout_value <= 0:
        raise ValueError("timeout_s must be positive")
    return timeout_value


def _validate_bet_sizes(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        raw_parts = [value]
    bet_sizes = tuple(float(part) for part in raw_parts if str(part).strip())
    if not bet_sizes:
        raise ValueError("bet_sizes must not be empty")
    if len(bet_sizes) > 5:
        raise ValueError("bet_sizes_exceed_tiny_limit:5")
    if any(size <= 0 for size in bet_sizes):
        raise ValueError("bet_sizes must be positive")
    return bet_sizes


def _extract_root_strategy_raw(
    *,
    game: Any,
    solved: Any,
    bet_size_fractions: tuple[float, ...],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        average_strategy = getattr(solved, "average_strategy", None)
        if not isinstance(average_strategy, dict) or not average_strategy:
            return None, "root_strategy_not_available:average_strategy_missing"

        state = game.initial_state()
        player = game.current_player(state)
        if player is None or int(player) < 0:
            return None, "root_strategy_not_available:no_current_player"
        root_key = game.infoset_key(state, player)
        actions = list(game.legal_actions(state))
        probs = average_strategy.get(root_key)
        if probs is None:
            return None, "root_strategy_not_available:root_infoset_missing"
        frequencies = _numeric_frequencies(probs)
        if frequencies is None:
            return None, "root_strategy_not_available:invalid_frequency_value"
        if len(actions) != len(frequencies):
            return None, "root_strategy_not_available:action_frequency_length_mismatch"
        total = sum(frequencies)
        if total < 0.98 or total > 1.02:
            return None, "root_strategy_not_available:invalid_frequency_sum"

        return (
            {
                "infoset_key": str(root_key),
                "player": int(player),
                "root_player": int(player),
                "root_player_role": "unknown",
                "action_ids": [int(action) for action in actions],
                "action_labels": [_action_label(action, bet_size_fractions) for action in actions],
                "frequencies": [round(float(value), 12) for value in frequencies],
                "source": "average_strategy",
                "bet_size_fractions": [float(value) for value in bet_size_fractions],
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001 - strategy inspection is best-effort
        return None, f"root_strategy_not_available:{_format_error(exc)}"


def _numeric_frequencies(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    frequencies: list[float] = []
    for value in values:
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            return None
        if frequency < 0.0 or frequency > 1.0:
            return None
        frequencies.append(frequency)
    return frequencies


def _action_label(action: Any, bet_size_fractions: tuple[float, ...]) -> str:
    action_id = int(action)
    if action_id == 0:
        return "FOLD"
    if action_id == 1:
        return "CHECK"
    if action_id == 2:
        return "CALL"
    if 3 <= action_id <= 7:
        return f"BET_{_fraction_label(bet_size_fractions[action_id - 3])}"
    if 8 <= action_id <= 12:
        return f"RAISE_{_fraction_label(bet_size_fractions[action_id - 8])}"
    if action_id == 13:
        return "ALL_IN"
    return f"ACTION_{action_id}"


def _fraction_label(value: float) -> str:
    percent = float(value) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return str(int(round(percent)))
    text = f"{percent:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _cards_to_solver_string(value: Any, *, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise ValueError("cards value is required")
    if isinstance(value, str):
        text = value.strip()
        if allow_empty and not text:
            return ""
        if "," in text or " " in text:
            return " ".join(_normalize_card_token(part) for part in text.replace(",", " ").split())
        if len(text) in (0, 2, 4, 6, 8, 10):
            return text
        return _normalize_card_token(text)
    if isinstance(value, (list, tuple)):
        parts = [_normalize_card_token(part) for part in value if part is not None and str(part).strip()]
        if allow_empty and not parts:
            return ""
        return " ".join(parts)
    raise TypeError(f"unsupported cards value type:{type(value).__name__}")


def _normalize_card_token(value: Any) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("empty card token")
    token = token.replace("10", "T")
    suit_map = {
        "\u2660": "s",
        "\u2665": "h",
        "\u2666": "d",
        "\u2663": "c",
        "\u00e2\u2122\u00a0": "s",
        "\u00e2\u2122\u00a5": "h",
        "\u00e2\u2122\u00a6": "d",
        "\u00e2\u2122\u00a3": "c",
    }
    for raw, normalized in suit_map.items():
        token = token.replace(raw, normalized)
    return token


def _equity_result_to_dict(value: Any) -> dict[str, Any]:
    return {
        "equity": float(getattr(value, "equity", 0.0)),
        "win_pct": float(getattr(value, "win_pct", 0.0)),
        "tie_pct": float(getattr(value, "tie_pct", 0.0)),
        "lose_pct": float(getattr(value, "lose_pct", 0.0)),
        "win": int(getattr(value, "win", 0)),
        "tie": int(getattr(value, "tie", 0)),
        "lose": int(getattr(value, "lose", 0)),
        "iterations": int(getattr(value, "iterations", 0)),
    }


def _input_path_value(solver_path: str | Path | None) -> str | None:
    if solver_path is not None:
        return str(solver_path)
    return os.environ.get("POKER_SOLVER_PATH")


def _result(
    started: float,
    input_payload: dict[str, Any],
    output: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    status = "failed" if error else "ok"
    return {
        "status": status,
        "solver_name": SOLVER_NAME,
        "input": input_payload,
        "output": output,
        "error": error,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}:{message}"
    return type(exc).__name__


__all__ = [
    "check_solver_available",
    "compute_equity_hand_vs_hand",
    "compute_equity_hand_vs_range",
    "solve_simple_postflop_spot",
    "solve_tiny_postflop_spot",
]
