from __future__ import annotations

import pytest

from datasets.external_oracle_sources.label_mapping import (
    extract_bet_size,
    normalize_action_3intent,
    normalize_action_4class,
)


@pytest.mark.parametrize(
    ("raw", "four_class", "intent"),
    [
        ("<action>check</action>", "CHECK", "NO_INVEST"),
        ("check", "CHECK", "NO_INVEST"),
        ("fold", "FOLD", "NO_INVEST"),
        ("call", "CALL", "CALL"),
        ("bet 12.5", "RAISE", "RAISE"),
        ("raise 40", "RAISE", "RAISE"),
        ("raises to 100", "RAISE", "RAISE"),
        ("<ACTION>Bet 375</ACTION>", "RAISE", "RAISE"),
        ("all-in raise 19750", "RAISE", "RAISE"),
        ("b275", "RAISE", "RAISE"),
    ],
)
def test_action_normalization_variants(raw: str, four_class: str, intent: str) -> None:
    assert normalize_action_4class(raw) == four_class
    assert normalize_action_3intent(raw) == intent


def test_extract_bet_size_from_raise_variants() -> None:
    assert extract_bet_size("bet 12.5") == 12.5
    assert extract_bet_size("raise 40") == 40.0
    assert extract_bet_size("raises to 100") == 100.0
    assert extract_bet_size("<action>bet 375</action>") == 375.0

