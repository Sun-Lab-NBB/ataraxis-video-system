"""Provides the sequential processing pipeline that runs the camera timestamp extraction jobs of one recording.

Notes:
    This module serves the command-line interface and any external driver that dispatches one job by its identifier.
    It imports no batch engine. Batch orchestration across many recordings belongs to the MCP server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from .errors import OrchestrationErrors
from .worker import execute_job
from .discovery import prepare_jobs

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
        independent external jobs share one tracker without resetting each other's state. That is what supports
        running every source of one recording in parallel under an external scheduler.

        Each job runs at the requested worker ceiling, and the extraction narrows that to what its own archive
        repays. A sequential run commits one job's resources at a time, so it weighs nothing against a budget and
        reads no archive before dispatching it.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``camera_timestamps/`` subdirectory is created
            automatically under this path, and all tracker and feather output files are written there.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (external mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        source_ids: A list of camera source IDs to process in local mode. Each ID must be registered in the camera
            manifest and correspond to exactly one archive under the log directory. If not provided, resolves all
            registered source IDs from the manifest. This argument is ignored in external mode.
        workers: The ceiling on the workers any single job receives. Setting this to a value less than 1 resolves the
            ceiling from the host's core count. Setting this to 1 conducts every job sequentially.
        display_progress: Determines whether to display progress bars during timestamp extraction.

    Raises:
        OrchestrationError: Carrying the kind of the first unsupported condition met, which covers the manifest
            resolution kinds, the unknown job source kind, the unresolved archive kind, the split logger output kind,
            and the unknown job identifier kind.
        OSError: If any directory beneath the log directory cannot be read.
    """
    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        source_ids=source_ids,
        job_id=job_id,
        core_ceiling=workers,
    )

    # A caller reaching this function asked for work to be carried out, so resolving nothing is a failure here even
    # though the resolution itself reports a recording holding no camera data as an ordinary answer.
    if not job_set.jobs:
        message = (
            f"Unable to process camera log archives in '{log_directory}'. The recording resolved no extraction job. "
            f"Its tree holds no camera manifest, or the manifest registers no source whose log archive resolves to "
            f"exactly one file beneath it."
        )
        console.error(message=message, error=OrchestrationErrors.UNRESOLVED_ARCHIVE.as_error())

    console.echo(
        message=(
            f"Resolved {len(job_set.jobs)} job(s) for source ID(s): {', '.join(job.source_id for job in job_set.jobs)}"
        )
    )

    tracker = ProcessingTracker(file_path=job_set.tracker_path)

    for job in job_set.jobs:
        execute_job(
            log_path=job.archive_path,
            output_directory=job.output_directory,
            source_id=job.source_id,
            job_id=job.job_id,
            workers=job.core_weight,
            tracker=tracker,
            display_progress=display_progress,
        )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)
