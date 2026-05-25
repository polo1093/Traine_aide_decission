"""Small card/deck helpers for synthetic poker spots."""

from __future__ import annotations

import random
from collections.abc import Iterable


RANKS = "23456789TJQKA"
SUITS = "hdcs"


def full_deck() -> list[str]:
    """Return a canonical 52-card deck using solver card notation."""

    return [f"{rank}{suit}" for rank in RANKS for suit in SUITS]


def validate_unique_cards(cards: Iterable[str]) -> None:
    """Raise ``ValueError`` when a card list contains duplicates."""

    card_list = list(cards)
    duplicates = sorted({card for card in card_list if card_list.count(card) > 1})
    if duplicates:
        raise ValueError(f"duplicate_cards:{','.join(duplicates)}")


def remaining_deck(excluded: Iterable[str] = ()) -> list[str]:
    """Return the canonical deck without excluded cards."""

    excluded_set = set(excluded)
    deck = [card for card in full_deck() if card not in excluded_set]
    if len(deck) != 52 - len(excluded_set):
        raise ValueError("excluded_cards_not_unique_or_invalid")
    return deck


def draw_cards(rng: random.Random, count: int, excluded: Iterable[str] = ()) -> list[str]:
    """Draw cards without replacement from the canonical deck."""

    if count < 0:
        raise ValueError("draw_count_must_be_nonnegative")
    deck = remaining_deck(excluded)
    if count > len(deck):
        raise ValueError("draw_count_exceeds_remaining_deck")
    rng.shuffle(deck)
    return deck[:count]
