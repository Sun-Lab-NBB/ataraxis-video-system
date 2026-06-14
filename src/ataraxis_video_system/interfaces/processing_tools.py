"""Provides MCP tools for preparing, executing, monitoring, canceling, and resetting batch log processing jobs, as well
as analyzing and cleaning processed frame timestamp output.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from threading import Thread

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, TimestampFormats, TimestampPrecisions, convert_time, get_timestamp
from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

from ..video import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    prepare_tracker,
    generate_job_ids,
)
from .mcp_instance import mcp
from .mcp_execution import (
    PendingJob,
    JobExecutionState,
    get_execution_state,
    set_execution_state,
    job_execution_manager,
)

_RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations. The worker budget is computed as available cores minus this
value, with a minimum of 1."""


@mcp.tool()
def prepare_log_processing_batch_tool(
    log_directories: list[str],
    source_ids: list[str],
    output_directories: list[str],
) -> dict[str, Any]:
    """Prepares an execution manifest for batch log processing without starting execution.

    Accepts log directories, source IDs, and output directories from the caller and initializes a
    ProcessingTracker with one timestamp-extraction job per source ID for each log directory. Idempotent: if a
    tracker already exists for a log directory, returns the existing manifest with current job statuses instead
    of reinitializing. Requires prior discovery — the caller must provide confirmed source IDs rather than
    relying on implicit archive or manifest discovery.

    Important:
        The AI agent calling this tool MUST run discover_camera_data_tool first to obtain log directory paths
        and confirmed source IDs. The agent MUST ask the user for the output directory paths before calling
        this tool. Do not assume or guess directory paths or source IDs.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_camera_data_tool.
        source_ids: The list of confirmed source IDs to process. Accepts IDs from the 'source_id' field of
            entries in the 'sources' list returned by discover_camera_data_tool. Applied uniformly: each log
            directory creates jobs for every source ID in this list that has a matching archive on disk.
        output_directories: The list of absolute paths for per-log-directory output. Must match the length of
            log_directories. Each output directory receives a ``camera_timestamps/`` subdirectory containing
            the processing tracker and feather output files.

    Returns:
        A dictionary containing per-log-directory manifests in 'log_directories' with tracker paths and job
        lists, total counts, and any invalid paths.
    """
    if len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    source_id_set = set(source_ids)
    result_log_dirs: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_dir_str in enumerate(log_directories):
        log_dir_path = Path(log_dir_str)

        if not log_dir_path.exists() or not log_dir_path.is_dir():
            invalid_paths.append(log_dir_str)
            continue

        # Filters the requested source IDs to those that have a matching archive in this log directory.
        # Discovery already confirmed these archives exist, but the check guards against stale data.
        filtered_ids = sorted(
            source_id for source_id in source_id_set if (log_dir_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}").exists()
        )

        if not filtered_ids:
            result_log_dirs[log_dir_str] = {"source_ids": [], "jobs": [], "tracker_path": None, "summary": {}}
            continue

        # Resolves the output directory for this log directory.
        output_path = Path(output_directories[entry_index])

        # Creates the camera_timestamps subdirectory under the output path for tracker and feather files.
        timestamps_path = output_path / CAMERA_TIMESTAMPS_DIRECTORY
        timestamps_path.mkdir(parents=True, exist_ok=True)
        tracker_path = timestamps_path / TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state.
            try:
                tracker_status = _read_tracker_status(tracker_path=tracker_path)
            except Exception:
                tracker_status = {"jobs": [], "summary": {}}

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(timestamps_path),
                "source_ids": filtered_ids,
                **tracker_status,
            }
            total_jobs += len(tracker_status.get("jobs", []))
        else:
            # Initializes a new tracker with jobs for the filtered source IDs. Uses prepare_tracker so the MCP
            # batch-preparation path inherits the same regeneration logic as run_log_processing_pipeline.
            tracker = ProcessingTracker(file_path=tracker_path)
            tracker_jobs: list[tuple[str, str]] = [(TIMESTAMP_JOB_NAME, source_id) for source_id in filtered_ids]
            prepare_tracker(tracker=tracker, jobs=tracker_jobs, universe=tracker_jobs)
            job_ids = generate_job_ids(source_ids=filtered_ids)

            jobs: list[dict[str, str]] = [
                {
                    "job_id": job_ids[source_id],
                    "source_id": source_id,
                    "status": "SCHEDULED",
                    "log_directory": log_dir_str,
                    "output_directory": str(timestamps_path),
                    "tracker_path": str(tracker_path),
                }
                for source_id in filtered_ids
            ]

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(timestamps_path),
                "source_ids": filtered_ids,
                "jobs": jobs,
                "summary": {
                    "total": len(jobs),
                    "succeeded": 0,
                    "failed": 0,
                    "running": 0,
                    "scheduled": len(jobs),
                },
            }
            total_jobs += len(jobs)

    result: dict[str, Any] = {
        "success": True,
        "log_directories": result_log_dirs,
        "total_log_directories": len(result_log_dirs),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    worker_budget: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution with budget-based worker allocation.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager that allocates CPU cores to each job based on its archive size. The worker budget directly
    controls memory footprint since each worker spawns a separate process. Large archives (>= 2000 messages) receive
    more workers, while small archives receive 1 worker since they process sequentially regardless. The manager fills
    available budget greedily, dispatching smaller jobs alongside large ones when cores are available.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors, each a dictionary with 'log_directory', 'output_directory',
            'tracker_path', 'job_id', and 'source_id' keys.
        worker_budget: The total number of CPU cores available for the execution session. Directly controls memory
            footprint. Set to -1 for automatic resolution via resolve_worker_count.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved worker budget, per-job message counts, and
        any invalid jobs.
    """
    # Enforces single-session constraint.
    existing_state = get_execution_state()
    if (
        existing_state is not None
        and existing_state.manager_thread is not None
        and existing_state.manager_thread.is_alive()
    ):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"log_directory", "output_directory", "tracker_path", "job_id", "source_id"}
    pending: list[PendingJob] = []
    all_jobs: dict[tuple[str, str], PendingJob] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_dict in jobs:
        if not required_keys.issubset(job_dict.keys()):
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {required_keys - job_dict.keys()}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        pending_job = PendingJob(
            log_directory=Path(job_dict["log_directory"]),
            output_directory=Path(job_dict["output_directory"]),
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            source_id=job_dict["source_id"],
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the total worker budget.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=_RESERVED_CORES)

    # Probes archive message counts for all pending jobs. This reads only the zip directory of each .npz file,
    # which is fast and does not load message data into memory.
    job_message_counts: dict[tuple[str, str], int] = {}
    for job in pending:
        job_message_counts[job.dispatch_key] = _probe_archive_message_count(job=job)

    # Creates execution state and starts the manager thread.
    state = JobExecutionState(
        all_jobs=all_jobs,
        pending_queue=pending,
        job_message_counts=job_message_counts,
        worker_budget=resolved_budget,
    )
    set_execution_state(state)

    manager = Thread(target=job_execution_manager, daemon=True)
    manager.start()
    state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "job_message_counts": job_message_counts,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()
def get_log_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of the active log processing execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. If no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts
        for pending, running, succeeded, and failed jobs.
    """
    state = get_execution_state()
    if state is None:
        return {"active": False, "message": "No execution session exists."}

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    # Reads status from tracker files for each job.
    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:
            job_details.extend(
                {"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"} for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id in tracker.jobs:
                job_state = tracker.jobs[job.job_id]
                status = job_state.status

                if status == ProcessingStatus.SUCCEEDED:
                    succeeded_count += 1
                elif status == ProcessingStatus.FAILED:
                    failed_count += 1
                elif status == ProcessingStatus.RUNNING:
                    running_count += 1
                else:
                    scheduled_count += 1

                entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id, "status": status.name}
                if job_state.error_message is not None:
                    entry["error_message"] = job_state.error_message
                if job_state.executor_id is not None:
                    entry["executor_id"] = job_state.executor_id
                job_details.append(entry)
            else:
                job_details.append({"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"})

    return {
        "active": manager_alive,
        "canceled": state.canceled,
        "jobs": job_details,
        "summary": {
            "total": len(state.all_jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


@mcp.tool()
def get_log_processing_timing_tool() -> dict[str, Any]:
    """Returns timing information for all jobs in the active execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps from ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with
        total elapsed seconds and throughput.
    """
    state = get_execution_state()
    if state is None:
        return {"active": False, "message": "No execution session exists."}

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()
    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    job_timing: list[dict[str, Any]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:  # noqa: S112
            continue

        for job in path_jobs:
            if job.job_id not in tracker.jobs:
                continue

            job_info = tracker.jobs[job.job_id]
            entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id}

            if job_info.executor_id is not None:
                entry["executor_id"] = job_info.executor_id

            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(duration_seconds, 2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level statistics.
    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            2,
        )

    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": sum(1 for entry in job_timing if "elapsed_seconds" in entry),
        "pending_count": len(state.all_jobs)
        - completed_count
        - failed_count
        - sum(1 for entry in job_timing if "elapsed_seconds" in entry),
    }

    if completed_count > 0 and earliest_start is not None:
        elapsed_hours = convert_time(
            time=current_us - earliest_start,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.HOUR,
            as_float=True,
        )
        if elapsed_hours > 0:
            session["throughput_jobs_per_hour"] = round(completed_count / elapsed_hours, 2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()
def cancel_log_processing_tool() -> dict[str, Any]:
    """Cancels the active log processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    state = get_execution_state()
    if state is None:
        return {"canceled": False, "message": "No execution session is active."}

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_groups)

    # Counts final job statuses from tracker files.
    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
            for job_state in tracker.jobs.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1
        except Exception:  # noqa: S110
            pass

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} group(s) still completing.",
        "final_state": {
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "active_jobs_at_cancel": active_count,
        },
    }


@mcp.tool()
def reset_log_processing_jobs_tool(
    tracker_path: str,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the ProcessingTracker YAML file.
        source_ids: An optional list of source IDs whose jobs should be reset. If not provided, all jobs are reset.

    Returns:
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Identifies which job IDs to reset based on source_ids filter.
    target_ids: set[str] = set()
    if source_ids is not None:
        source_id_set = set(source_ids)
        for job_id, job_state in tracker.jobs.items():
            if job_state.specifier in source_id_set:
                target_ids.add(job_id)
    else:
        target_ids = set(tracker.jobs.keys())

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Collects (job_name, specifier) tuples for the jobs to reset.
    reset_jobs: list[tuple[str, str]] = [
        (tracker.jobs[job_id].job_name, tracker.jobs[job_id].specifier) for job_id in target_ids
    ]

    # Removes target jobs and re-initializes them.
    for job_id in target_ids:
        del tracker.jobs[job_id]
    tracker.to_yaml(file_path=path)

    # Re-initializes the reset jobs.
    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    # Reads back the updated state for the response.
    try:
        updated_status = _read_tracker_status(tracker_path=path)
    except Exception:
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:
    """Discovers and summarizes processing status for all log directories under a root directory.

    Recursively searches for camera_processing_tracker.yaml files and aggregates their status. Each tracker
    corresponds to a single DataLogger output directory.

    Args:
        root_directory: The absolute path to the root directory to search for tracker files.

    Returns:
        A dictionary containing per-log-directory status summaries and aggregate counts.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    log_directory_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for tracker_path in sorted(root_path.rglob(TRACKER_FILENAME)):
        log_directory = str(tracker_path.parent)
        try:
            status = _read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            directory_status = _derive_tracker_status(summary=summary)

            log_directory_statuses.append(
                {
                    "log_directory": log_directory,
                    "tracker_path": str(tracker_path),
                    "status": directory_status,
                    **status,
                }
            )
        except Exception:
            log_directory_statuses.append(
                {
                    "log_directory": log_directory,
                    "tracker_path": str(tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    return {
        "log_directories": log_directory_statuses,
        "total_log_directories": len(log_directory_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()
def analyze_camera_frame_statistics_tool(
    feather_files: list[str],
    drop_threshold_us: int = 0,
    max_drop_locations: int = 50,
) -> dict[str, Any]:
    """Reads one or more processed camera timestamp feather files and computes frame acquisition statistics.

    For each file, computes basic recording statistics (total frames, duration, estimated frame rate), inter-frame
    timing distribution (mean, median, standard deviation, min, max), and frame drop analysis (gap detection,
    estimated drop count, drop locations). Frame drops are identified as inter-frame intervals exceeding a threshold,
    which defaults to 2x the median inter-frame interval when not specified. Accepts the 'timestamps_file' paths
    returned by discover_camera_data_tool.

    Args:
        feather_files: The list of absolute paths to camera timestamp feather files produced by the log processing
            pipeline. Expected filename pattern: ``camera_{source_id}_timestamps.feather``. Accepts paths from the
            'timestamps_file' field returned by discover_camera_data_tool.
        drop_threshold_us: The inter-frame interval threshold in microseconds above which a gap is classified as a
            frame drop. When 0, the threshold is automatically computed as 2x the median inter-frame interval.
            Applied uniformly to all files.
        max_drop_locations: The maximum number of frame drop locations to include per file. Caps the
            'drop_locations' list to prevent oversized responses.

    Returns:
        A dictionary containing a 'results' list with per-file statistics (each with 'file', 'basic_stats',
        'inter_frame_timing', and 'frame_drop_analysis' keys) and a 'total_files' count. Files that cannot be
        read produce an entry with 'file' and 'error' keys instead of statistics.
    """
    results = [
        _analyze_single_feather(
            feather_file=feather_file, drop_threshold_us=drop_threshold_us, max_drop_locations=max_drop_locations
        )
        for feather_file in feather_files
    ]

    return {"results": results, "total_files": len(results)}


@mcp.tool()
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]:
    """Deletes the camera_timestamps subdirectory under one or more output directories.

    Removes each ``camera_timestamps/`` subdirectory and all of its contents, including processed feather files
    and the processing tracker. Uses ``delete_directory`` from ataraxis-data-structures for parallel file deletion
    with platform-safe retry logic. After cleanup, the output directories can be passed to
    prepare_log_processing_batch_tool to reinitialize from scratch. Accepts the 'log_directories' list returned
    by discover_camera_data_tool.

    Args:
        output_directories: The list of absolute paths to output directories containing ``camera_timestamps/``
            subdirectories to delete.

    Returns:
        A dictionary containing a 'results' list with per-directory outcomes (each with 'output_directory',
        'cleaned' flag, and either 'timestamps_path' or 'error') and a 'total_cleaned' count.
    """
    results = [_clean_single_output(output_directory=directory) for directory in output_directories]
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_directories": len(results)}


def _read_tracker_status(tracker_path: Path) -> dict[str, Any]:
    """Reads a log processing tracker file and returns structured per-job status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.

    Returns:
        A dictionary containing per-job status details and summary counts.
    """
    tracker = ProcessingTracker.from_yaml(file_path=tracker_path)

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in tracker.jobs.items():
        source_id = job_state.specifier or job_id[:8]
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {"job_id": job_id, "source_id": source_id, "status": status.name}
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        job_details.append(entry)

    return {
        "jobs": job_details,
        "summary": {
            "total": len(tracker.jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def _derive_tracker_status(summary: dict[str, Any]) -> str:
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if any
    are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise.

    Args:
        summary: A dictionary containing 'total', 'succeeded', 'failed', 'running', and 'scheduled' counts.

    Returns:
        A status string: one of 'failed', 'completed', 'processing', 'not_started', or 'in_progress'.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0) > 0:
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


def _group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[PendingJob]]:
    """Groups all jobs in an execution state by their tracker file path.

    Minimizes redundant file reads by batching jobs that share the same tracker, so each tracker YAML file is
    deserialized only once when iterating over the groups.

    Args:
        state: The active job execution state containing the job registry.

    Returns:
        A dictionary mapping each tracker path to its list of pending job descriptors.
    """
    tracker_jobs: dict[Path, list[PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _analyze_single_feather(
    feather_file: str,
    drop_threshold_us: int,
    max_drop_locations: int,
) -> dict[str, Any]:
    """Reads a single camera timestamp feather file and computes frame acquisition statistics.

    Args:
        feather_file: The absolute path to the feather file.
        drop_threshold_us: The inter-frame interval threshold in microseconds. When 0, auto-detected as 2x median.
        max_drop_locations: The maximum number of frame drop locations to include.

    Returns:
        A dictionary containing 'file', 'basic_stats', 'inter_frame_timing', and 'frame_drop_analysis' keys,
        or 'file' and 'error' keys if the file cannot be read.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return {"file": feather_file, "error": f"File does not exist: {feather_file}"}

    if not file_path.is_file():
        return {"file": feather_file, "error": f"Path is not a file: {feather_file}"}

    # Reads the feather file and validates the expected schema.
    try:
        dataframe = pl.read_ipc(source=file_path)
    except Exception as error:
        return {"file": feather_file, "error": f"Unable to read feather file: {error}"}

    if "frame_time_us" not in dataframe.columns:
        return {"file": feather_file, "error": f"Missing required 'frame_time_us' column. Found: {dataframe.columns}"}

    timestamps = dataframe["frame_time_us"].to_numpy()
    total_frames = len(timestamps)

    # Handles edge cases for empty or single-frame recordings.
    if total_frames == 0:
        return {
            "file": feather_file,
            "basic_stats": {"total_frames": 0},
            "inter_frame_timing": {},
            "frame_drop_analysis": {},
        }

    if total_frames == 1:
        return {
            "file": feather_file,
            "basic_stats": {
                "total_frames": 1,
                "first_timestamp_us": int(timestamps[0]),
                "last_timestamp_us": int(timestamps[0]),
                "duration_us": 0,
                "duration_seconds": 0.0,
                "estimated_fps": 0.0,
            },
            "inter_frame_timing": {},
            "frame_drop_analysis": {},
        }

    # Computes basic recording statistics.
    first_timestamp_us = int(timestamps[0])
    last_timestamp_us = int(timestamps[-1])
    duration_us = last_timestamp_us - first_timestamp_us
    duration_seconds = round(
        convert_time(time=duration_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True), 6
    )
    estimated_fps = round((total_frames - 1) / duration_seconds, 3) if duration_seconds > 0 else 0.0

    # Computes inter-frame interval statistics. Casts to int64 to handle potential uint64 underflow.
    intervals_us = np.diff(timestamps).astype(np.int64)
    mean_us = round(float(np.mean(intervals_us)), 2)
    median_us = round(float(np.median(intervals_us)), 2)
    std_us = round(float(np.std(intervals_us)), 2)
    min_us = int(np.min(intervals_us))
    max_us = int(np.max(intervals_us))

    # Performs frame drop analysis using the specified or auto-detected threshold.
    if drop_threshold_us > 0:
        threshold = float(drop_threshold_us)
        threshold_source = "user_specified"
    else:
        threshold = 2.0 * median_us
        threshold_source = "auto_2x_median"

    drop_mask = intervals_us > threshold
    drop_indices = np.where(drop_mask)[0]
    total_gaps_detected = len(drop_indices)

    if total_gaps_detected > 0:
        # Estimates the number of dropped frames per gap using the median interval as expected spacing.
        expected_interval = median_us if median_us > 0 else 1.0
        dropped_per_gap = np.round(intervals_us[drop_mask] / expected_interval).astype(np.int64) - 1
        total_estimated_dropped_frames = int(np.sum(np.maximum(dropped_per_gap, 0)))

        total_expected_frames = total_frames + total_estimated_dropped_frames
        drop_rate_percent = round(total_estimated_dropped_frames / total_expected_frames * 100, 4)

        longest_gap_us = int(np.max(intervals_us[drop_mask]))
        longest_gap_ms = round(
            convert_time(
                time=longest_gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
            ),
            4,
        )

        # Builds the capped drop locations list.
        drop_locations: list[dict[str, Any]] = []
        for index in drop_indices[:max_drop_locations]:
            gap_us = int(intervals_us[index])
            gap_ms = round(
                convert_time(
                    time=gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
                ),
                4,
            )
            estimated_lost = max(round(gap_us / expected_interval) - 1, 0)
            drop_locations.append(
                {
                    "frame_index": int(index),
                    "gap_us": gap_us,
                    "gap_ms": gap_ms,
                    "estimated_frames_lost": estimated_lost,
                }
            )

        frame_drop_analysis: dict[str, Any] = {
            "threshold_us": round(threshold, 2),
            "threshold_source": threshold_source,
            "total_gaps_detected": total_gaps_detected,
            "total_estimated_dropped_frames": total_estimated_dropped_frames,
            "drop_rate_percent": drop_rate_percent,
            "longest_gap_us": longest_gap_us,
            "longest_gap_ms": longest_gap_ms,
            "drop_locations": drop_locations,
            "drop_locations_truncated": total_gaps_detected > max_drop_locations,
        }
    else:
        frame_drop_analysis = {
            "threshold_us": round(threshold, 2),
            "threshold_source": threshold_source,
            "total_gaps_detected": 0,
            "total_estimated_dropped_frames": 0,
            "drop_rate_percent": 0.0,
            "longest_gap_us": 0,
            "longest_gap_ms": 0.0,
            "drop_locations": [],
            "drop_locations_truncated": False,
        }

    # Converts inter-frame interval statistics from microseconds to milliseconds.
    mean_ms, median_ms, std_ms, min_ms, max_ms = (
        round(
            convert_time(time=value, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True),
            4,
        )
        for value in (mean_us, median_us, std_us, min_us, max_us)
    )

    return {
        "file": feather_file,
        "basic_stats": {
            "total_frames": total_frames,
            "first_timestamp_us": first_timestamp_us,
            "last_timestamp_us": last_timestamp_us,
            "duration_us": duration_us,
            "duration_seconds": duration_seconds,
            "estimated_fps": estimated_fps,
        },
        "inter_frame_timing": {
            "mean_us": mean_us,
            "median_us": median_us,
            "std_us": std_us,
            "min_us": min_us,
            "max_us": max_us,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        },
        "frame_drop_analysis": frame_drop_analysis,
    }


def _clean_single_output(output_directory: str) -> dict[str, Any]:
    """Deletes the camera_timestamps subdirectory under a single output directory.

    Args:
        output_directory: The absolute path to the output directory.

    Returns:
        A dictionary containing 'output_directory', 'cleaned' flag, and either 'timestamps_path' or 'error'.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"output_directory": output_directory, "cleaned": False, "error": "Directory does not exist."}

    if not output_path.is_dir():
        return {"output_directory": output_directory, "cleaned": False, "error": "Path is not a directory."}

    timestamps_path = output_path / CAMERA_TIMESTAMPS_DIRECTORY

    if not timestamps_path.exists():
        return {"output_directory": output_directory, "cleaned": True, "message": "Nothing to clean."}

    try:
        delete_directory(directory_path=timestamps_path)
    except Exception as error:
        return {
            "output_directory": output_directory,
            "cleaned": False,
            "timestamps_path": str(timestamps_path),
            "error": f"Unable to delete: {error}",
        }

    return {"output_directory": output_directory, "cleaned": True, "timestamps_path": str(timestamps_path)}


def _probe_archive_message_count(job: PendingJob) -> int:
    """Probes the message count of a job's log archive by reading the .npz zip directory.

    Reconstructs the archive path from the job's log directory and source ID, then reads the file list from the .npz
    archive without loading any message data. The message count is the total entry count minus one (excluding the onset
    message).

    Args:
        job: The pending job descriptor containing the log directory and source ID.

    Returns:
        The number of data messages in the archive, or 0 if the archive cannot be read.
    """
    archive_path = job.log_directory / f"{job.source_id}{LOG_ARCHIVE_SUFFIX}"
    if not archive_path.exists():
        return 0

    try:
        with np.load(file=archive_path, allow_pickle=False) as archive:
            return max(0, len(archive.files) - 1)
    except Exception:
        return 0
