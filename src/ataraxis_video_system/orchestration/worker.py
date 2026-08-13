"""Provides the single-job runner every scheduler dispatches and the picklable descriptor-addressed entry point a
process pool submits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker

from .jobs import CAMERA_EXTRACTION_JOB_NAME, resolve_timestamps_path
from ..video import ExtractedDataColumns, extract_logged_camera_timestamps

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    from .jobs import JobDescriptor


def run_extraction_job(job: JobDescriptor) -> None:
    """Runs one camera timestamp extraction job described entirely by its descriptor.

    Notes:
        This is the picklable entry point a process pool submits. It takes one flat descriptor, so the only state
        crossing the process boundary is paths, strings, and integers.

        Opens the tracker from the descriptor's own path, because a tracker instance caches an in-memory job registry
        that would arrive in the child already stale.

    Args:
        job: The descriptor of the job to run.
    """
    execute_job(
        log_path=job.archive_path,
        output_directory=job.output_directory,
        source_id=job.source_id,
        job_id=job.job_id,
        workers=job.core_weight,
        tracker=ProcessingTracker(file_path=job.tracker_path),
        display_progress=False,
    )


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
        Delegates the job's state transitions to the tracker's run_job() context manager. The context marks the job
        as running, completes it when the block returns, and marks it as failed with the exception's message before
        re-raising.

        Writes the feather file directly into the output directory, creates no directory, and registers no job on
        the tracker, so a scheduler owning its own tracker and output layout dispatches this function unchanged.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The identifier of the camera source whose archive is processed.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The tracker recording this job's outcome.
        display_progress: Determines whether to display a progress bar during timestamp extraction.
        executor: When provided, parallel processing reuses this pool instead of creating a new one.
    """
    console.echo(message=f"Running '{CAMERA_EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")

    with tracker.run_job(job_id=job_id):
        timestamps = extract_logged_camera_timestamps(
            log_path=log_path, workers=workers, display_progress=display_progress, executor=executor
        )

        # Polars can reference the numpy buffer directly, avoiding a full copy of the timestamp data.
        column = str(ExtractedDataColumns.FRAME_TIME)
        dataframe = pl.DataFrame({column: pl.Series(name=column, values=timestamps)})
        dataframe.write_ipc(file=resolve_timestamps_path(output_directory=output_directory, source_id=source_id))
