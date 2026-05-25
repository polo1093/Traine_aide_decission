"""Bounded solver job helpers for offline PokerSolver validation."""

from solver_jobs.batch_runner import run_solver_batch, write_solver_batch_jsonl
from solver_jobs.eligibility import evaluate_solver_eligibility
from solver_jobs.job_builder import build_solver_job, manual_fixture_spots
from solver_jobs.job_file_runner import run_solver_job_file
from solver_jobs.job_runner import run_solver_job
from solver_jobs.job_schema import SCHEMA_VERSION, validate_solver_job
from solver_jobs.snapshot_mapper import map_snapshot_to_solver_job
from solver_jobs.subprocess_runner import run_solver_job_subprocess

__all__ = [
    "SCHEMA_VERSION",
    "build_solver_job",
    "evaluate_solver_eligibility",
    "map_snapshot_to_solver_job",
    "manual_fixture_spots",
    "run_solver_batch",
    "run_solver_job_file",
    "run_solver_job_subprocess",
    "run_solver_job",
    "validate_solver_job",
    "write_solver_batch_jsonl",
]
