"""Provides the single-job runner every scheduler dispatches and the local pipeline entry point that runs a whole
DataLogger output directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker, find_log_archive

from .jobs import (
    TRACKER_FILENAME,
    FRAME_TIME_COLUMN,
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    generate_job_ids,
    discover_camera_jobs,
    resolve_camera_timestamps_path,
)
from ..video import extract_logged_camera_timestamps
from .allocation import resolve_core_budget, resolve_job_workers, resolve_archive_footprint

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor


def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None:
    """Executes a single timestamp extraction job for the target log archive.

    Extracts camera frame acquisition timestamps from the log archive, converts them to a Polars DataFrame, and
    writes the result as an IPC (Feather) file.

    Notes:
        Delegates the job's state transitions to the tracker's run_job() context manager, which marks the job as
        running, completes it when the block returns, and marks it as failed with the exception's message before
        re-raising when the block raises an Exception.

        Writes the feather file directly into the output directory, creates no directory, and registers no job on
        the tracker, so a scheduler owning its own tracker and output layout dispatches this function unchanged.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        display_progress: Determines whether to display a progress bar during timestamp extraction.
        executor: When provided, parallel processing reuses this pool instead of creating a new one. The pool is
            passed through to extract_logged_camera_timestamps to avoid spawning a redundant process pool.
    """
    console.echo(message=f"Running '{TIMESTAMP_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")

    with tracker.run_job(job_id=job_id):
        # Extracts frame acquisition timestamps from the log archive as a contiguous numpy array.
        timestamps = extract_logged_camera_timestamps(
            log_path=log_path, workers=workers, display_progress=display_progress, executor=executor
        )

        # Wraps the numpy array in a Polars DataFrame for Feather output. Polars can reference the numpy buffer
        # directly, avoiding a full copy of the timestamp data.
        dataframe = pl.DataFrame({FRAME_TIME_COLUMN: pl.Series(name=FRAME_TIME_COLUMN, values=timestamps)})
        dataframe.write_ipc(file=resolve_camera_timestamps_path(output_directory=output_directory, source_id=source_id))


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None = None,
    log_ids: list[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested VideoSystem log archives from a single DataLogger output directory.

    Supports both local and remote processing modes. In local mode (job_id is None), resolves each requested log
    archive by source ID, aligns a processing tracker in the output directory with the requested jobs, and executes
    them sequentially. In remote mode (job_id is provided), aligns the tracker with the full job universe derived
    from the camera manifest, then resolves and executes only the single archive matching the requested job ID. The
    universe alignment lets independent remote jobs share one tracker without resetting each other's state, which
    supports running every source in parallel under an external scheduler.

    In local mode, all resolved archives must reside in the same directory. If the log_directory contains archives
    from multiple DataLogger instances (in separate subdirectories), each must be processed independently. Use the
    MCP batch processing tools to orchestrate multi-directory workflows.

    Notes:
        Every job's width is resolved from the archive it reads, so a run mixing a long recording with a short one
        gives each the cores its own archive repays rather than one width chosen for the whole run.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``camera_timestamps/`` subdirectory is created
            automatically under this path, and all tracker and feather output files are written there.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        log_ids: A list of source log IDs to process in local mode. Each ID must correspond to exactly one archive
            under the log directory, and all archives must reside in the same parent directory. If not provided,
            reads the camera_manifest.yaml file from the log directory to resolve all registered source IDs. This
            argument is ignored in remote mode, where the executed job is selected solely by job_id.
        workers: The ceiling on the cores any single job receives. Setting this to a value less than 1 resolves the
            ceiling from the host's core count. Setting this to 1 conducts every job sequentially.
        display_progress: Determines whether to display progress bars during timestamp extraction. Defaults to True
            for interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, a requested log ID has no matching archive, or no
            camera manifest is found.
        OSError: If any directory beneath the log_directory cannot be read.
        ValueError: If the provided job_id does not match any job in the manifest universe, if no source IDs can be
            resolved, if a requested log ID matches multiple archives, or if resolved archives span multiple
            directories.
    """
    # Builds the universe of every job the manifest could produce: one timestamp-extraction job per registered
    # camera source ID. The universe is a manifest fingerprint, not an invocation fingerprint, so every invocation
    # (full, subset, or single remote job) aligns the tracker against the same set and never resets sibling jobs.
    universe, _ = discover_camera_jobs(log_directory=log_directory)
    universe_ids = [specifier for _, specifier in universe]

    # Creates the camera_timestamps subdirectory under the output path. All tracker and feather files are written here.
    timestamps_path = output_directory / CAMERA_TIMESTAMPS_DIRECTORY
    timestamps_path.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=timestamps_path / TRACKER_FILENAME)

    # Bounds every job resolved below, so no job is dispatched at a width the host cannot supply.
    ceiling = resolve_core_budget(requested_budget=workers)

    if job_id is not None:
        # Remote mode: selects the job to run solely by ID, validated against the manifest universe. Aligns the
        # tracker with the full universe so start_job finds the requested ID and concurrent remote jobs do not
        # treat each other's entries as foreign. Resolves only the matched archive so a missing or late sibling
        # archive cannot fail this job.
        _, source_id = tracker.resolve_job(job_id=job_id, universe=universe)

        tracker.align_jobs(jobs=universe, universe=universe)

        _execute_sized_job(
            log_path=find_log_archive(log_directory=log_directory, source_id=source_id),
            output_directory=timestamps_path,
            source_id=source_id,
            job_id=job_id,
            ceiling=ceiling,
            tracker=tracker,
            display_progress=display_progress,
        )
    else:
        # Local mode: resolves source IDs from the manifest when none are provided, otherwise validates the
        # requested IDs against the manifest to prevent processing non-video logs.
        if log_ids is None or not log_ids:
            source_ids = universe_ids
            console.echo(message=f"Resolved {len(source_ids)} source ID(s) from manifest: {', '.join(source_ids)}")
        else:
            invalid_ids = [source_id for source_id in log_ids if source_id not in universe_ids]
            if invalid_ids:
                message = (
                    f"Unable to process logs in '{log_directory}'. The following source IDs are not registered "
                    f"in the camera manifest: {', '.join(invalid_ids)}. Registered source IDs: "
                    f"{', '.join(universe_ids)}."
                )
                console.error(message=message, error=ValueError)
            source_ids = sorted(log_ids)

        # Resolves all requested archive paths upfront and validates they belong to the same DataLogger directory.
        archive_paths = {
            source_id: find_log_archive(log_directory=log_directory, source_id=source_id) for source_id in source_ids
        }
        parent_directories = {path.parent for path in archive_paths.values()}
        if len(parent_directories) > 1:
            message = (
                f"Unable to process logs in '{log_directory}'. The requested log archives span multiple "
                f"directories: {sorted(str(parent) for parent in parent_directories)}. Each DataLogger output "
                f"directory must be processed independently."
            )
            console.error(message=message, error=ValueError)

        # Aligns the tracker with the requested subset while detecting foreign entries against the full universe.
        jobs: list[tuple[str, str]] = [(TIMESTAMP_JOB_NAME, source_id) for source_id in source_ids]
        tracker.align_jobs(jobs=jobs, universe=universe)

        job_ids = generate_job_ids(source_ids=source_ids)

        for source_id in source_ids:
            _execute_sized_job(
                log_path=archive_paths[source_id],
                output_directory=timestamps_path,
                source_id=source_id,
                job_id=job_ids[source_id],
                ceiling=ceiling,
                tracker=tracker,
                display_progress=display_progress,
            )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _execute_sized_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    ceiling: int,
    tracker: ProcessingTracker,
    *,
    display_progress: bool,
) -> None:
    """Sizes one job from the archive it reads and executes it at the resolved width.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        ceiling: The cores available to this job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        display_progress: Determines whether to display a progress bar during timestamp extraction.
    """
    job_workers = resolve_job_workers(footprint=resolve_archive_footprint(archive_path=log_path), ceiling=ceiling)

    execute_job(
        log_path=log_path,
        output_directory=output_directory,
        source_id=source_id,
        job_id=job_id,
        workers=job_workers,
        tracker=tracker,
        display_progress=display_progress,
    )
