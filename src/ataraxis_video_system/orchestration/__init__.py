"""Provides the orchestration layer: the job descriptors and manifest-derived job discovery, the declared core
allocation and archive-derived footprint model, the local batch execution engine, and the pipeline entry point.
"""

from .jobs import (
    TRACKER_FILENAME,
    FRAME_TIME_COLUMN,
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_PREFIX,
    CAMERA_TIMESTAMPS_SUFFIX,
    CAMERA_TIMESTAMPS_DIRECTORY,
    PendingJob,
    generate_job_ids,
    discover_camera_jobs,
    resolve_camera_timestamps_path,
)
from .pipeline import execute_job, run_log_processing_pipeline
from .execution import (
    JobExecutionState,
    size_pending_job,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
)
from .allocation import (
    TIMESTAMP_JOB_CORES,
    ArchiveFootprint,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
)

__all__ = [
    "CAMERA_TIMESTAMPS_DIRECTORY",
    "CAMERA_TIMESTAMPS_PREFIX",
    "CAMERA_TIMESTAMPS_SUFFIX",
    "FRAME_TIME_COLUMN",
    "TIMESTAMP_JOB_CORES",
    "TIMESTAMP_JOB_NAME",
    "TRACKER_FILENAME",
    "ArchiveFootprint",
    "JobExecutionState",
    "PendingJob",
    "discover_camera_jobs",
    "estimate_job_memory_mb",
    "execute_job",
    "generate_job_ids",
    "get_execution_state",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "resolve_archive_footprint",
    "resolve_camera_timestamps_path",
    "resolve_core_budget",
    "resolve_job_workers",
    "resolve_memory_budget_mb",
    "run_log_processing_pipeline",
    "set_execution_state",
    "size_pending_job",
]
