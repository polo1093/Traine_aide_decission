"""Bounded solver job helpers for offline PokerSolver validation."""

from solver_jobs.batch_runner import run_solver_batch, write_solver_batch_jsonl
from solver_jobs.action_candidate import build_solver_action_candidate
from solver_jobs.candidate_quality import evaluate_candidate_quality
from solver_jobs.eligibility import evaluate_solver_eligibility
from solver_jobs.hero_oriented_builder import (
    build_hero_oriented_solver_job,
    validate_hero_root_alignment,
)
from solver_jobs.job_builder import build_solver_job, manual_fixture_spots
from solver_jobs.job_file_runner import run_solver_job_file
from solver_jobs.job_runner import run_solver_job
from solver_jobs.job_schema import SCHEMA_VERSION, validate_solver_job
from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job
from solver_jobs.subprocess_runner import run_solver_job_subprocess

__all__ = [
    "SCHEMA_VERSION",
    "build_hero_oriented_solver_job",
    "build_solver_action_candidate",
    "build_solver_job",
    "evaluate_candidate_quality",
    "evaluate_solver_eligibility",
    "map_snapshot_to_solver_job",
    "manual_fixture_spots",
    "run_solver_batch",
    "run_solver_job_file",
    "run_solver_job_subprocess",
    "run_solver_job",
    "validate_hero_root_alignment",
    "validate_solver_job",
    "write_solver_batch_jsonl",
]
