"""Synthetic solver job generation helpers."""

from synthetic.spot_generator import (
    DEFAULT_MAX_COUNT,
    SUPPORTED_PROFILES,
    generate_solver_jobs,
    is_drawy_board,
    is_paired_board,
)
from synthetic.hero_oriented_spot_generator import (
    SUPPORTED_HERO_CONTEXTS,
    generate_hero_oriented_solver_jobs,
)

__all__ = [
    "DEFAULT_MAX_COUNT",
    "SUPPORTED_HERO_CONTEXTS",
    "SUPPORTED_PROFILES",
    "generate_hero_oriented_solver_jobs",
    "generate_solver_jobs",
    "is_drawy_board",
    "is_paired_board",
]
