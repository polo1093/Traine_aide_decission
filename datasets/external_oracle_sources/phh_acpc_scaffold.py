"""Scaffold adapter for PHH / Zenodo ACPC HUNL hand histories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import SCHEMA_VERSION, empty_normalized_dataframe


SOURCE_DATASET = "PHH / Zenodo ACPC HUNL"
SOURCE_URL = "https://github.com/uoftcprg/phh-dataset"
ZENODO_URL = "https://doi.org/10.5281/zenodo.10796885"
SOURCE_TERMS = "PHH repository is MIT licensed; ACPC/Zenodo hand-history contents are not direct solver labels and require source-specific redistribution review."


def parse_phh_file(path: str | Path):
    """Return an empty normalized dataframe placeholder for future PHH parsing."""
    _ = Path(path)
    return empty_normalized_dataframe()


def write_phh_scaffold(*, output_dir: str | Path, force: bool = False) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(f"output_dir_exists_use_force:{output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "zenodo_url": ZENODO_URL,
        "source_license_or_terms": SOURCE_TERMS,
        "rows_loaded": 0,
        "rows_usable": 0,
        "rows_dropped": 0,
        "label_distribution_3intent": {},
        "raw_action_distribution": {},
        "missing_field_rates": {},
        "leakage_warnings": [
            "PHH/ACPC histories are action traces, not direct solver-oracle labels.",
            "No heavy archives are downloaded by this scaffold.",
        ],
        "generation_timestamp": None,
        "recommended_next_step": "Implement a bounded PHH parser and decide whether labels come from imitation, offline RL, or solver relabeling.",
    }
    (output_path / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "dataset_card.md").write_text(render_phh_card(), encoding="utf-8")
    return report


def render_phh_card() -> str:
    return "\n".join(
        [
            "# PHH / Zenodo ACPC HUNL Scaffold",
            "",
            f"Source: `{SOURCE_URL}`",
            f"Zenodo: `{ZENODO_URL}`",
            "",
            "## Type",
            "",
            "Expert benchmark and hand-history corpus. It is not a direct solver-label dataset.",
            "",
            "## Intended Future Uses",
            "",
            "- Imitation learning from observed actions.",
            "- Offline RL experiments.",
            "- Solver relabeling of selected decision points.",
            "- Parser compatibility tests against real hand-history structure.",
            "",
            "## Current Status",
            "",
            "Scaffold only. No archives are downloaded by default, and no model is trained.",
            "",
            "## Limitations",
            "",
            "- Labels are not theoretical best actions without relabeling.",
            "- ACPC HUNL integration requires substantial parsing and game-state reconstruction.",
            "- Redistribution and source terms must be reviewed before publishing derived exports.",
            "",
        ]
    )
