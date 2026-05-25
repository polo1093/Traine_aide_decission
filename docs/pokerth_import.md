# PokerTH Import

The PokerTH importer turns text hand histories into reconstructed hand
summaries, then builds solver-ready `ml_dataset_v1` snapshots only when the hand
is simple enough.

This importer does not train a model, does not generate ML labels, and does not
modify PokerSolver or `aide_decision`.

## Supported Input

The parser expects hands with headers like:

```text
## Game: 4 | Hand: 38 ##
polo posts small blind ($160)
Player 3 posts big blind ($320)
*** FLOP *** [9♣, 5♣, 8♠]
*** TURN *** [K♦]
*** RIVER *** [2h]
polo shows [8♥,9♠]
Player 3 shows [8♣,K♣]
```

Cards are normalized to solver-style strings:

```text
8♥ -> 8h
9♠ -> 9s
8♣ -> 8c
K♦ -> Kd
```

## Two-Step Flow

First parse the complete hand:

```python
from pokerth.history_parser import parse_pokerth_hand

parsed = parse_pokerth_hand(text, hero_name="polo")
hand_summary = parsed["hand_summary"]
```

Then build one requested snapshot only when the context is clear:

```python
from pokerth.snapshot_builder import build_snapshot_from_hand_summary

snapshot_result = build_snapshot_from_hand_summary(
    hand_summary,
    street="FLOP",
    # Pass None when no real decision context is known; the builder will reject
    # instead of inventing to_call=0.
    to_call=0.0,
)
```

The builder does not automatically create flop, turn, and river snapshots. This
prevents clean-looking snapshots from being invented out of multiway or
incomplete histories.

## Snapshot Shape

Successful snapshots are `ml_dataset_v1` records:

```python
{
    "schema_version": "ml_dataset_v1",
    "snapshot_id": "pokerth_game4_hand38_flop",
    "metadata": {
        "source_type": "pokerth_history",
        "source_reliability": "reconstructed_history",
        "game_id": 4,
        "hand_id": 38,
        "street": "FLOP",
    },
    "features": {
        "hero_cards": ["8h", "9s"],
        "villain_hand": ["8c", "Kc"],
        "board_cards": ["9c", "5c", "8s"],
        "pot": 1600.0,
        "to_call": 0.0,
        "to_call_is_estimated": False,
        "decision_context_known": True,
        "active_opponents": 1,
        "units": "chips",
        "pot_is_estimated": True,
        "pot_reconstruction_method": "sum_posted_bet_call_raise_amounts",
    },
    "quality_flags": {
        "usable_for_training": False,
        "usable_for_solver": True,
    },
    "labels": {
        "label_source": "pokerth_history",
        "label_quality": "history_reconstructed",
        "training_label": None,
    },
}
```

The winner of a hand is never converted into a strategic label.

## Rejection Rules

Hands or requested snapshots are rejected with stable reasons:

- `multiway_not_supported`;
- `multiway_context_not_supported`;
- `villain_hand_missing`;
- `hero_hand_missing`;
- `side_pot_not_supported`;
- `invalid_board`;
- `to_call_unknown`;
- `pot_reconstruction_failed`;
- `showdown_missing`.

If more than two players saw the flop, a flop snapshot is rejected. A later
turn/river snapshot may be built only when that specific street is requested,
the active players are exactly hero plus one villain, the board is complete, and
both hands are visible.

## Pot And To Call

Pot reconstruction is fragile, so the snapshot explicitly marks:

```python
"pot_is_estimated": True
"pot_reconstruction_method": "sum_posted_bet_call_raise_amounts"
```

`to_call` is not invented. If the caller does not know a clear decision context,
the builder should receive `to_call=None`, which rejects the snapshot with
`to_call_unknown`. Accepted snapshots include:

```python
"to_call_is_estimated": False
"decision_context_known": True
```

The mapper rejects snapshots with `decision_context_known: False` or missing
`to_call`.

## Tests

Run PokerTH tests:

```powershell
python -m pytest tests/test_pokerth_history_parser.py tests/test_pokerth_snapshot_builder.py
```

Run all project tests:

```powershell
python -m pytest
```

## Limits

- The parser is conservative and supports only common English PokerTH text
  patterns.
- Side pots are excluded.
- Multiway contexts are excluded unless a later requested street is clearly
  heads-up.
- Pot is estimated, not authoritative.
- Snapshots are solver-ready validation inputs, not ML training labels.
