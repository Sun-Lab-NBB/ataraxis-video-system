"""Provides the orchestration layer: the job identity and output layout, the archive-derived sizing model, the
manifest-derived job resolution, the single-job runner, the shared-pool batch engine, and the sequential pipeline.
"""

from .jobs import (
    CAMERA_EXTRACTION_JOB_NAME,
    JobSizing,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    resolve_timestamps_path,
)
from .worker import execute_job
from .pipeline import run_log_processing_pipeline
from .discovery import JobSet, JobSource, JobUniverse, size_job, prepare_jobs, resolve_jobs
from .execution import (
    ActiveJob,
    JobExecutionState,
    get_execution_state,
    group_jobs_by_tracker,
    start_execution_session,
)
from .allocation import (
    CAMERA_EXTRACTION_JOB_CORES,
    ArchiveFootprint,
    size_archive_job,
    resolve_pool_size,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_memory_budget_mb,
)

__all__ = [
    "CAMERA_EXTRACTION_JOB_CORES",
    "CAMERA_EXTRACTION_JOB_NAME",
    "ActiveJob",
    "ArchiveFootprint",
    "JobDescriptor",
    "JobExecutionState",
    "JobSet",
    "JobSizing",
    "JobSource",
    "JobUniverse",
    "OutputLayout",
    "estimate_job_memory_mb",
    "execute_job",
    "generate_job_ids",
    "get_execution_state",
    "group_jobs_by_tracker",
    "prepare_jobs",
    "resolve_core_budget",
    "resolve_job_workers",
    "resolve_jobs",
    "resolve_memory_budget_mb",
    "resolve_pool_size",
    "resolve_timestamps_path",
    "run_log_processing_pipeline",
    "size_archive_job",
    "size_job",
    "start_execution_session",
]
