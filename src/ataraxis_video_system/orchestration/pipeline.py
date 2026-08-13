"""Provides the sequential processing pipeline that runs the camera timestamp extraction jobs of one recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from .worker import execute_job
from .discovery import prepare_jobs
from .allocation import resolve_job_workers, resolve_archive_footprint

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None = None,
    source_ids: Sequence[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested VideoSystem log archives from a single DataLogger output directory.

    Supports both local and external processing modes. In local mode (job_id is None), resolves each requested log
    archive by source ID, aligns a processing tracker in the output directory, and executes the jobs sequentially. In
    external mode (job_id is provided), resolves and executes only the single archive matching the requested job ID.

    Notes:
        The tracker is aligned against the full job universe the camera manifest defines in both modes, which lets
        independent external jobs share one tracker without resetting each other's state.

        Each job runs at the width the caller named, or at the width its own archive resolves to when the caller
        named none. Jobs run one at a time, so this path weighs nothing against a core or a memory budget.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``camera_timestamps/`` subdirectory is created
            automatically under this path, and all tracker and feather output files are written there.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (external mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        source_ids: The camera source IDs to process in local mode. Each ID must be registered in the camera
            manifest and correspond to exactly one archive under the log directory. If not provided, resolves all
            registered source IDs from the manifest. This argument is ignored in external mode.
        workers: The workers every job receives. A positive value is used verbatim. A non-positive value resolves the
            width from each archive, which is one worker below the parallel extraction threshold and the declared
            per-job allocation above it.
        display_progress: Determines whether to display progress bars during timestamp extraction.

    Raises:
        FileNotFoundError: If the log directory does not exist, if a requested source's archive is absent, or if
            the recording resolves no job to run.
        ValueError: If the tree holds more than one camera manifest, if a manifest registers no sources, if a
            requested source or job identifier is not registered, if the resolved archives span several directories,
            or if a resolved log archive carries no onset timestamp message.
        OSError: If any directory beneath the log directory cannot be read.
        YAMLError: If the camera manifest does not hold a well-formed YAML document.
        MissingValueError: If the camera manifest omits a field the CameraManifest class requires.
        TimeoutError: If the tracker's .lock file cannot be acquired within the timeout period.
    """
    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        source_ids=source_ids,
        job_id=job_id,
    )

    # A caller reaching this function asked for work to be carried out, so resolving nothing is a failure here even
    # though the resolution itself reports a recording holding no camera data as an ordinary answer.
    if not job_set.jobs:
        message = (
            f"Unable to process camera log archives in '{log_directory}'. The recording resolved no extraction job. "
            f"Its tree holds no camera manifest, or the manifest registers no source whose log archive resolves to "
            f"exactly one file beneath it."
        )
        console.error(message=message, error=FileNotFoundError)

    console.echo(
        message=(
            f"Resolved {len(job_set.jobs)} job(s) for source ID(s): {', '.join(job.source_id for job in job_set.jobs)}"
        )
    )

    tracker = ProcessingTracker(file_path=job_set.tracker_path)

    for job in job_set.jobs:
        # An unset width is the one choice the caller left open, so it is resolved from each archive in turn.
        job_workers = (
            workers
            if workers > 0
            else resolve_job_workers(footprint=resolve_archive_footprint(archive_path=job.archive_path))
        )
        execute_job(
            log_path=job.archive_path,
            output_directory=job.output_directory,
            source_id=job.source_id,
            job_id=job.job_id,
            workers=job_workers,
            tracker=tracker,
            display_progress=display_progress,
        )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)
