"""Provides MCP tools for preparing, executing, monitoring, canceling, and resetting batch log processing jobs, as well
as analyzing and cleaning processed frame timestamp output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import replace

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, TimestampFormats, TimestampPrecisions, convert_time, get_timestamp
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    delete_directory,
    discover_marker_files,
)

from ..video import ExtractedDataColumns
from .responses import (
    page_fields,
    project_item,
    resolve_page,
    item_breakdown,
    reject_unknown,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from ..orchestration import (
    JobSizing,
    OutputLayout,
    JobDescriptor,
    ArchiveFootprint,
    JobExecutionState,
    size_job,
    prepare_jobs,
    resolve_pool_size,
    get_execution_state,
    resolve_core_budget,
    resolve_job_workers,
    group_jobs_by_tracker,
    estimate_job_memory_mb,
    start_execution_session,
    resolve_memory_budget_mb,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_OVERVIEW_AXES: tuple[str, ...] = ("status",)
"""The directory keys a caller filters the batch overview by, which a bare call reports the counts of."""

_OVERVIEW_SEMI_DETAIL_FIELDS: tuple[str, ...] = ("output_directory", "status", "summary")
"""The fields every listed output directory carries."""

_OVERVIEW_DETAIL_FIELDS: tuple[str, ...] = ("tracker_path", "jobs", "error")
"""The fields a listed output directory carries once detail is requested. One entry per tracked job makes the job list
the term that grows a whole-project overview fastest, so it is withheld until a caller asks for it."""

_AUTO_DROP_THRESHOLD_MULTIPLIER: float = 1.5
"""The multiple of the median inter-frame interval a gap has to pass before it is examined as a frame drop.

Notes:
    Losing a single frame spans two nominal intervals, so a multiplier of two sits on the very gap it is meant to
    catch and resolves on acquisition jitter rather than on loss. A multiplier between one and two separates the two
    outcomes, and the midpoint holds the widest margin against jitter in both directions.
"""


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
    of reinitializing. Requires prior discovery. The caller must provide confirmed source IDs.

    Important:
        The AI agent calling this tool MUST run discover_camera_data_tool first to obtain log directory paths, and
        MUST read the confirmed source IDs from the 'breakdown' that call reports or from the 'sources' list it
        returns under include_items. The agent MUST ask the user for the output directory paths before calling
        this tool. Do not assume or guess directory paths or source IDs.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_camera_data_tool.
        source_ids: The list of confirmed source IDs to process. Accepts the 'source_id' keys of the 'breakdown' a
            bare discover_camera_data_tool call reports, and the 'source_id' field of the entries in the 'sources'
            list it returns once a filter is named or include_items is set. Applied uniformly: each log
            directory creates jobs for every source ID in this list that has a matching archive on disk. Passing an
            empty list prepares every source the log directory's camera_manifest.yaml registers.
        output_directories: The list of absolute paths for per-log-directory output. Must match the length of
            log_directories. Each output directory receives a ``camera_timestamps/`` subdirectory containing
            the processing tracker and feather output files.

    Returns:
        A dictionary containing a 'success' flag and per-log-directory manifests in 'log_directories'. Each manifest
        carries the tracker path, the output directory, the resolved source IDs, a 'summary' of status counts, and the
        sources the sizing skipped with their reasons. Each manifest also carries a 'jobs' list of dispatchable
        descriptors, annotated with their sized cores and memory, their live tracker status, an 'error_message' for a
        job whose tracker recorded a failure, and the archive figures 'message_count', 'archive_bytes', and 'modeled'
        that the execution tool requires. Also reports total counts and
        any invalid paths. Returns an error dictionary when the log directory and output directory lists differ in
        length.
    """
    if len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    result_log_directories: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_directory_string in enumerate(log_directories):
        log_directory_path = Path(log_directory_string)

        if not log_directory_path.exists() or not log_directory_path.is_dir():
            invalid_paths.append(log_directory_string)
            continue

        # Sizes every requested source that the manifest registers and whose archive resolves under this log
        # directory. Lenient sourcing records the sources it cannot size rather than failing the whole batch, since
        # one caller applies one source list across several recordings.
        try:
            job_set = prepare_jobs(
                log_directory=log_directory_path,
                output_directory=Path(output_directories[entry_index]),
                source_ids=source_ids or None,
                strict_sources=False,
            )
            sized_jobs = [size_job(job=job) for job in job_set.jobs]
            sized_jobs.sort(key=lambda entry: entry[1].memory_mb, reverse=True)
        except Exception:
            invalid_paths.append(log_directory_string)
            continue

        # Merges the tracker's live state over the sized set, so a directory prepared twice reports what its jobs
        # have already done rather than presenting every job as freshly scheduled.
        try:
            tracker_status = _read_tracker_status(tracker_path=job_set.tracker_path)
        except Exception:
            tracker_status = {"jobs": [], "summary": {}}

        recorded = {entry["job_id"]: entry for entry in tracker_status.get("jobs", [])}

        jobs: list[dict[str, Any]] = []
        for descriptor, sizing in sized_jobs:
            entry: dict[str, Any] = dict(descriptor.to_mapping())
            entry["memory_mb"] = sizing.memory_mb
            entry["message_count"] = sizing.message_count
            entry["archive_bytes"] = sizing.archive_bytes
            entry["modeled"] = sizing.modeled
            entry["status"] = recorded.get(descriptor.job_id, {}).get("status", "SCHEDULED")
            error_message = recorded.get(descriptor.job_id, {}).get("error_message")
            if error_message is not None:
                entry["error_message"] = error_message
            jobs.append(entry)

        result_log_directories[log_directory_string] = {
            "tracker_path": str(job_set.tracker_path),
            "output_directory": str(job_set.output_directory),
            "source_ids": [descriptor.source_id for descriptor, _ in sized_jobs],
            "jobs": jobs,
            "summary": tracker_status.get("summary", {}),
            "skipped_sources": [
                {"source_id": source_id, "reason": reason} for source_id, reason in job_set.skipped_sources
            ],
        }
        total_jobs += len(jobs)

    result: dict[str, Any] = {
        "success": True,
        "log_directories": result_log_directories,
        "total_log_directories": len(result_log_directories),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, Any]],
    *,
    core_budget: int = -1,
    memory_budget_mb: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution against a core and a memory budget.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager. Each job's cores and memory are resolved from the archive it reads before dispatch, so a long
    recording and a short one are admitted at their own sizes. An archive below the parallel extraction threshold takes
    a single core and every archive above it takes the declared stage width, collapsed onto the core budget when that
    budget is narrower. The
    manager admits a job once the running set has room for both its cores and its memory, and it admits an oversized
    job alone rather than leaving it queued forever. A job whose estimated memory passes the host's total physical
    memory is failed on its tracker instead of being admitted, since it cannot complete wherever it is dispatched.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors, each a dictionary carrying every key the 'jobs' entries of
            prepare_log_processing_batch_tool report.
        core_budget: The total number of CPU cores available for the execution session. Set to -1 to auto-resolve
            to every available core minus the reserved host cores.
        memory_budget_mb: The total memory in megabytes available for the execution session. Set to -1 to
            auto-resolve to a share of the host's physical memory.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', the resolved 'core_budget', 'memory_budget_mb', and
        'pool_size', a 'job_allocations' entry per job carrying its 'job_id', 'source_id', 'cores', 'memory_mb',
        'message_count', and 'modeled', and any invalid jobs. Returns an error dictionary when an execution session
        is already active, and one carrying 'invalid_jobs' when no submitted job is valid.
    """
    # Resolves both budgets before sizing, since the core budget bounds the width any single job receives.
    resolved_cores = resolve_core_budget(requested_budget=core_budget)
    resolved_memory = resolve_memory_budget_mb(requested_budget_mb=memory_budget_mb)

    # Rebuilds every descriptor and its footprint from the mapping the preparation emitted, then resolves this
    # session's own width and memory from those figures. The preparation already read each archive, so re-deriving
    # the width against a different budget costs no filesystem access.
    pending: list[tuple[JobDescriptor, JobSizing]] = []
    all_jobs: dict[tuple[str, str], JobDescriptor] = {}
    invalid_jobs: list[dict[str, Any]] = []
    job_allocations: list[dict[str, Any]] = []

    for job_dict in jobs:
        try:
            descriptor = JobDescriptor.from_mapping(mapping=job_dict)
        except Exception as error:
            invalid_jobs.append({**job_dict, "error": str(error)})
            continue

        if not descriptor.tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {descriptor.tracker_path}"})
            continue

        try:
            footprint = ArchiveFootprint(
                message_count=int(job_dict["message_count"]),
                archive_bytes=int(job_dict["archive_bytes"]),
                modeled=bool(job_dict["modeled"]),
            )
        except (KeyError, TypeError, ValueError):
            invalid_jobs.append({**job_dict, "error": "Missing or unreadable sizing keys from the prepared manifest."})
            continue

        core_weight = min(resolve_job_workers(footprint=footprint), resolved_cores)
        descriptor = replace(descriptor, core_weight=core_weight)
        sizing = JobSizing(
            memory_mb=estimate_job_memory_mb(footprint=footprint, cores=core_weight),
            message_count=footprint.message_count,
            archive_bytes=footprint.archive_bytes,
            modeled=footprint.modeled,
        )

        pending.append((descriptor, sizing))
        all_jobs[descriptor.dispatch_key] = descriptor
        job_allocations.append(
            {
                "job_id": descriptor.job_id,
                "source_id": descriptor.source_id,
                "cores": core_weight,
                "memory_mb": sizing.memory_mb,
                "message_count": sizing.message_count,
                "modeled": sizing.modeled,
            }
        )

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Creates the execution state and reserves the single session slot. The reservation performs the incumbent check,
    # the publication, and the thread start as one atomic step, since two callers splitting those steps would each
    # start a manager and double-commit the host's cores and memory.
    pool_size = resolve_pool_size(job_count=len(pending), core_budget=resolved_cores, memory_budget_mb=resolved_memory)
    state = JobExecutionState(
        all_jobs=all_jobs,
        pending_jobs=pending,
        core_budget=resolved_cores,
        memory_budget_mb=resolved_memory,
        pool_size=pool_size,
    )

    if not start_execution_session(state=state):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "core_budget": resolved_cores,
        "memory_budget_mb": resolved_memory,
        "pool_size": pool_size,
        "job_allocations": job_allocations,
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
        A dictionary containing an 'active' flag, a 'canceled' flag, per-job status entries in 'jobs', each naming the
        log directory the job reads so jobs sharing a source across recordings stay distinguishable, and a
        'summary' carrying 'total', 'succeeded', 'failed', 'running', and 'scheduled' counts.
    """
    state = get_execution_state()
    if state is None:
        return {"active": False, "message": "No execution session exists."}

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
        except Exception:
            job_details.extend(
                {
                    "job_id": job.job_id,
                    "source_id": job.source_id,
                    "log_directory": str(job.log_directory),
                    "status": "UNKNOWN",
                }
                for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id in registry:
                job_state = registry[job.job_id]
                status = job_state.status

                if status == ProcessingStatus.SUCCEEDED:
                    succeeded_count += 1
                elif status == ProcessingStatus.FAILED:
                    failed_count += 1
                elif status == ProcessingStatus.RUNNING:
                    running_count += 1
                else:
                    scheduled_count += 1

                entry: dict[str, Any] = {
                    "job_id": job.job_id,
                    "source_id": job.source_id,
                    "log_directory": str(job.log_directory),
                    "status": status.name,
                }
                if job_state.error_message is not None:
                    entry["error_message"] = job_state.error_message
                if job_state.executor_id is not None:
                    entry["executor_id"] = job_state.executor_id
                job_details.append(entry)
            else:
                job_details.append(
                    {
                        "job_id": job.job_id,
                        "source_id": job.source_id,
                        "log_directory": str(job.log_directory),
                        "status": "UNKNOWN",
                    }
                )

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
        A dictionary containing an 'active' flag, per-job timing in 'jobs', each naming the log directory the job
        reads so jobs sharing a source across recordings stay distinguishable, and a 'session' summary carrying
        'total_elapsed_seconds', 'completed_count', 'failed_count', 'running_count', and 'pending_count', plus
        'throughput_jobs_per_hour' once at least one job has completed. Returns an 'active' flag of False with a
        'message' and neither 'jobs' nor 'session' when no execution session exists.
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

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
        except Exception:  # noqa: S112 - An unreadable tracker contributes no timing, so the rest still report.
            continue

        for job in path_jobs:
            if job.job_id not in registry:
                continue

            job_info = registry[job.job_id]
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "source_id": job.source_id,
                "log_directory": str(job.log_directory),
            }

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
                entry["elapsed_seconds"] = round(elapsed_seconds, ndigits=2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(duration_seconds, ndigits=2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            ndigits=2,
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
            session["throughput_jobs_per_hour"] = round(completed_count / elapsed_hours, ndigits=2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()
def cancel_log_processing_tool() -> dict[str, Any]:
    """Cancels the active log processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation. Returns a 'canceled' flag of False with a 'message' and
        no 'final_state' when no execution session is active.
    """
    state = get_execution_state()
    if state is None:
        return {"canceled": False, "message": "No execution session is active."}

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_jobs)
        state.pending_jobs.clear()
        active_count = len(state.active_jobs)

    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
            for job_state in registry.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1
        except Exception:  # noqa: S110 - An unreadable tracker contributes no counts, since cancellation still holds.
            pass

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} job(s) still completing.",
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
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses. Returns a 'reset'
        flag of False with a 'message' and no job counts when no job matches the requested source IDs. Returns an
        error dictionary when the tracker file is absent or cannot be read.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    tracker = ProcessingTracker(file_path=path)
    try:
        registry = tracker.snapshot()
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    if source_ids is not None:
        source_id_set = set(source_ids)
        target_ids = [job_id for job_id, job_state in registry.items() if job_state.specifier in source_id_set]
    else:
        target_ids = list(registry)

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Resets the targeted jobs back to SCHEDULED under the tracker's lock, leaving every other job untouched.
    tracker.reset_jobs(job_ids=target_ids)

    try:
        updated_status = _read_tracker_status(tracker_path=path)
    except Exception:
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_batch_status_overview_tool(
    root_directory: str,
    statuses: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Summarizes processing status for all camera timestamp output directories under a root, in three widening stages.

    Recursively searches for camera_processing_tracker.yaml files and aggregates their status. Each tracker sits in
    the ``camera_timestamps/`` subdirectory of one output directory, so every entry reports that subdirectory under
    its 'output_directory' key rather than the DataLogger log directory the archives came from.

    A bare call reports the aggregate job counts alongside a ``breakdown`` naming how many directories carry each
    status, which answers what needs attention without listing anything. Naming a status adds a page of directories
    carrying their own counts. Opting into detail adds each directory's tracker path, its per-job entries, and the
    error message of a directory whose tracker cannot be read.

    The aggregate counts and the breakdown span every discovered directory regardless of the filters, so narrowing
    what is listed never distorts what is reported.

    Args:
        root_directory: The absolute path to the root directory to search for tracker files.
        statuses: Restricts the listing to directories carrying these status labels.
        limit: The directories to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero
            lists every match, which is how a caller reading under a tight filter takes the whole result at once.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list directories when no status is named.
        detailed: Determines whether the listed directories report their tracker path, their per-job entries, and the
            error message of a directory whose tracker cannot be read.

    Returns:
        A dictionary carrying 'total_output_directories', an aggregate 'summary' of job counts, and a 'breakdown' of
        directories per status. Carries an 'output_directories' list with 'rows', 'matched_rows', 'start_row', and
        'next_start_row' whenever a status is named or the listing is requested. Returns an error dictionary when the
        root directory does not exist, is not a directory, cannot be searched, or a status names a value no tracker
        holds.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    output_directory_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    try:
        tracker_paths = discover_marker_files(directory=root_path, marker_name=OutputLayout.TRACKER_FILENAME)
    except OSError as error:
        return {"error": f"Unable to search '{root_directory}': {error}"}

    for tracker_path in tracker_paths:
        output_directory = str(tracker_path.parent)
        try:
            status = _read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            directory_status = ProcessingTracker.resolve_status(summary=summary).value

            output_directory_statuses.append(
                {
                    "output_directory": output_directory,
                    "tracker_path": str(tracker_path),
                    "status": directory_status,
                    **status,
                }
            )
        except Exception:
            output_directory_statuses.append(
                {
                    "output_directory": output_directory,
                    "tracker_path": str(tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    response: dict[str, Any] = {
        "total_output_directories": len(output_directory_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
        "breakdown": item_breakdown(items=output_directory_statuses, axes=_OVERVIEW_AXES),
    }

    if statuses is None and not include_items:
        return response

    matched = output_directory_statuses
    if statuses is not None:
        rejection = reject_unknown(
            items=output_directory_statuses, key="status", values=statuses, subject="output directory"
        )
        if rejection is not None:
            return rejection
        matched = [entry for entry in matched if entry["status"] in statuses]

    fields = (*_OVERVIEW_SEMI_DETAIL_FIELDS, *_OVERVIEW_DETAIL_FIELDS) if detailed else _OVERVIEW_SEMI_DETAIL_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["output_directories"] = [project_item(item=entry, fields=fields) for entry in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


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
    which defaults to 1.5x the median inter-frame interval when not specified. Every gap is netted against the interval
    that follows it, so a frame whose timestamp arrives late and is repaid by the next interval reports no loss.

    Args:
        feather_files: The list of absolute paths to camera timestamp feather files produced by the log processing
            pipeline. Expected filename pattern: ``camera_{source_id}_timestamps.feather``. Accepts paths from the
            'timestamps_file' field of the 'sources' entries a detailed discover_camera_data_tool call returns.
        drop_threshold_us: The inter-frame interval threshold in microseconds above which a gap is classified as a
            frame drop. When 0, the threshold is automatically computed as 1.5x the median inter-frame interval.
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
    with platform-safe retry logic. Accepts the same output directory paths that were supplied to
    prepare_log_processing_batch_tool.

    Args:
        output_directories: The list of absolute paths to output directories containing ``camera_timestamps/``
            subdirectories to delete.

    Returns:
        A dictionary containing a 'results' list with per-directory outcomes and the 'total_cleaned' and
        'total_directories' counts. Each outcome carries an 'output_directory' and a 'cleaned' flag. A successful
        delete adds a 'timestamps_path', and a directory with nothing to clean adds a 'message'. An output directory
        that does not exist or is not a directory reports an 'error' alone, while a failed delete reports both a
        'timestamps_path' and an 'error'.
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
    registry = ProcessingTracker(file_path=tracker_path).snapshot()

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in registry.items():
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
            "total": len(registry),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def _analyze_single_feather(
    feather_file: str,
    drop_threshold_us: int,
    max_drop_locations: int,
) -> dict[str, Any]:
    """Reads a single camera timestamp feather file and computes frame acquisition statistics.

    Args:
        feather_file: The absolute path to the feather file.
        drop_threshold_us: The inter-frame interval threshold in microseconds. When 0, the threshold is resolved from
            the median inter-frame interval and the module's automatic drop threshold multiplier.
        max_drop_locations: The maximum number of frame drop locations to include.

    Returns:
        A dictionary containing 'file', 'basic_stats', 'inter_frame_timing', and 'frame_drop_analysis' keys,
        or 'file' and 'error' keys if the file cannot be read.
    """
    timestamps, error_message = _read_feather_timestamps(feather_file=feather_file)

    if timestamps is None:
        return {"file": feather_file, "error": error_message}

    statistics = _compute_frame_statistics(
        timestamps=timestamps, drop_threshold_us=drop_threshold_us, max_drop_locations=max_drop_locations
    )

    return {"file": feather_file, **statistics}


def _read_feather_timestamps(feather_file: str) -> tuple[NDArray[Any] | None, str | None]:
    """Reads the frame acquisition timestamp column of a single camera timestamp feather file.

    Args:
        feather_file: The absolute path to the feather file.

    Returns:
        A two-element tuple. The first element stores the frame acquisition timestamp array, or None when the file
        cannot be read. The second element stores the message explaining the failure, or None when the read succeeds.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return None, f"File does not exist: {feather_file}"

    if not file_path.is_file():
        return None, f"Path is not a file: {feather_file}"

    try:
        dataframe = pl.read_ipc(source=file_path)
    except Exception as error:
        return None, f"Unable to read feather file: {error}"

    if str(ExtractedDataColumns.FRAME_TIME) not in dataframe.columns:
        return None, f"Missing required '{ExtractedDataColumns.FRAME_TIME}' column. Found: {dataframe.columns}"

    return dataframe[str(ExtractedDataColumns.FRAME_TIME)].to_numpy(), None


def _compute_frame_statistics(
    timestamps: NDArray[Any],
    drop_threshold_us: int,
    max_drop_locations: int,
) -> dict[str, Any]:
    """Computes frame acquisition statistics from an array of frame acquisition timestamps.

    Args:
        timestamps: The frame acquisition timestamps, in microseconds elapsed since the UTC epoch onset.
        drop_threshold_us: The inter-frame interval threshold in microseconds. When 0, the threshold is resolved from
            the median inter-frame interval and the module's automatic drop threshold multiplier.
        max_drop_locations: The maximum number of frame drop locations to include.

    Returns:
        A dictionary containing the 'basic_stats', 'inter_frame_timing', and 'frame_drop_analysis' keys. A recording
        holding fewer than two timestamps carries empty 'inter_frame_timing' and 'frame_drop_analysis' dictionaries.
    """
    total_frames = len(timestamps)

    # Handles edge cases for empty or single-frame recordings.
    if total_frames == 0:
        return {
            "basic_stats": {"total_frames": 0},
            "inter_frame_timing": {},
            "frame_drop_analysis": {},
        }

    if total_frames == 1:
        return {
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

    first_timestamp_us = int(timestamps[0])
    last_timestamp_us = int(timestamps[-1])
    duration_us = last_timestamp_us - first_timestamp_us
    duration_seconds = round(
        convert_time(time=duration_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True),
        ndigits=6,
    )
    estimated_fps = round((total_frames - 1) / duration_seconds, ndigits=3) if duration_seconds > 0 else 0.0

    # Computes inter-frame interval statistics. Reinterpreting a uint64 buffer as int64 before differencing keeps a
    # decreasing pair negative and costs no allocation, so it holds the same values the cast produces while dropping
    # the full-length temporary. A column of any other width keeps the cast, which rounds each difference rather than
    # each timestamp.
    intervals_us: NDArray[Any]
    if timestamps.dtype == np.uint64:
        intervals_us = np.diff(timestamps.view(np.int64))
    else:
        intervals_us = np.diff(timestamps).astype(np.int64)
    mean_us = round(float(np.mean(intervals_us)), ndigits=2)
    median_us = round(float(np.median(intervals_us)), ndigits=2)
    std_us = round(float(np.std(intervals_us)), ndigits=2)
    min_us = int(np.min(intervals_us))
    max_us = int(np.max(intervals_us))

    # Performs frame drop analysis using the specified or auto-detected threshold.
    if drop_threshold_us > 0:
        threshold = float(drop_threshold_us)
        threshold_source = "user_specified"
    else:
        threshold = _AUTO_DROP_THRESHOLD_MULTIPLIER * median_us
        threshold_source = f"auto_{_AUTO_DROP_THRESHOLD_MULTIPLIER}x_median"

    drop_mask = intervals_us > threshold
    drop_indices = np.where(drop_mask)[0]
    total_gaps_detected = len(drop_indices)

    if total_gaps_detected > 0:
        expected_interval = median_us if median_us > 0 else 1.0
        dropped_per_gap = _estimate_dropped_frames(
            intervals_us=intervals_us, gap_indices=drop_indices, expected_interval=expected_interval
        )
        total_estimated_dropped_frames = int(np.sum(dropped_per_gap))
        jitter_compensated_gaps = int(np.count_nonzero(dropped_per_gap == 0))

        total_expected_frames = total_frames + total_estimated_dropped_frames
        drop_rate_percent = round(total_estimated_dropped_frames / total_expected_frames * 100, ndigits=4)

        longest_gap_us = int(np.max(intervals_us[drop_mask]))
        longest_gap_ms = round(
            convert_time(
                time=longest_gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
            ),
            ndigits=4,
        )

        drop_locations: list[dict[str, Any]] = []
        for position, index in enumerate(drop_indices[:max_drop_locations]):
            gap_us = int(intervals_us[index])
            gap_ms = round(
                convert_time(
                    time=gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
                ),
                ndigits=4,
            )
            drop_locations.append(
                {
                    "frame_index": int(index),
                    "gap_us": gap_us,
                    "gap_ms": gap_ms,
                    "estimated_frames_lost": int(dropped_per_gap[position]),
                }
            )

        frame_drop_analysis: dict[str, Any] = {
            "threshold_us": round(threshold, ndigits=2),
            "threshold_source": threshold_source,
            "total_gaps_detected": total_gaps_detected,
            "jitter_compensated_gaps": jitter_compensated_gaps,
            "total_estimated_dropped_frames": total_estimated_dropped_frames,
            "drop_rate_percent": drop_rate_percent,
            "longest_gap_us": longest_gap_us,
            "longest_gap_ms": longest_gap_ms,
            "drop_locations": drop_locations,
            "drop_locations_truncated": total_gaps_detected > max_drop_locations,
        }
    else:
        frame_drop_analysis = {
            "threshold_us": round(threshold, ndigits=2),
            "threshold_source": threshold_source,
            "total_gaps_detected": 0,
            "jitter_compensated_gaps": 0,
            "total_estimated_dropped_frames": 0,
            "drop_rate_percent": 0.0,
            "longest_gap_us": 0,
            "longest_gap_ms": 0.0,
            "drop_locations": [],
            "drop_locations_truncated": False,
        }

    mean_ms, median_ms, std_ms, min_ms, max_ms = (
        round(
            convert_time(time=value, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True),
            ndigits=4,
        )
        for value in (mean_us, median_us, std_us, min_us, max_us)
    )

    return {
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


def _estimate_dropped_frames(
    intervals_us: NDArray[Any], gap_indices: NDArray[Any], expected_interval: float
) -> NDArray[np.int64]:
    """Estimates the frames lost at each detected gap, netting every gap against the interval that follows it.

    Notes:
        A frame whose timestamp arrives late stretches its own interval and shortens the following one by the same
        amount, so the pair still spans the frames it carries. Charging the stretched interval on its own reports that
        jitter as a loss. Subtracting whatever the following interval falls short of a full interval leaves only the
        span no frame accounts for.

        The last interval of a recording has no successor to repay it, so it is paired with a full interval and
        carries no compensation.

    Args:
        intervals_us: The inter-frame intervals of the whole recording, in microseconds.
        gap_indices: The indices, into the interval array, of the gaps that passed the drop threshold.
        expected_interval: The interval one frame occupies when none is lost, in microseconds.

    Returns:
        The frames lost at each gap, in the order the gap indices were supplied.
    """
    gaps = intervals_us[gap_indices].astype(np.float64)

    following = np.full(gaps.shape, expected_interval, dtype=np.float64)
    successor_indices = gap_indices + 1
    resolved = successor_indices < len(intervals_us)
    following[resolved] = intervals_us[successor_indices[resolved]]

    shortfall = np.maximum(expected_interval - following, 0.0)
    unaccounted = gaps - shortfall
    dropped: NDArray[np.int64] = np.maximum(np.round(unaccounted / expected_interval).astype(np.int64) - 1, 0)
    return dropped


def _clean_single_output(output_directory: str) -> dict[str, Any]:
    """Deletes the camera_timestamps subdirectory under a single output directory.

    Args:
        output_directory: The absolute path to the output directory.

    Returns:
        A dictionary containing an 'output_directory' and a 'cleaned' flag. A successful delete adds a
        'timestamps_path', and a directory with nothing to clean adds a 'message'. A directory that is absent or is
        not a directory reports an 'error' alone, while a failed delete reports both a 'timestamps_path' and an
        'error'.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"output_directory": output_directory, "cleaned": False, "error": "Directory does not exist."}

    if not output_path.is_dir():
        return {"output_directory": output_directory, "cleaned": False, "error": "Path is not a directory."}

    timestamps_path = output_path / OutputLayout.DIRECTORY_NAME

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
