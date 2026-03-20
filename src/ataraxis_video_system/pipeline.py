"""Provides the log data processing pipeline for extracting frame timestamps from VideoSystem log archives."""

from enum import StrEnum
from pathlib import Path

import polars as pl
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from .video_system import extract_logged_camera_timestamps

_LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""Naming convention suffix for log archives produced by assemble_log_archives()."""

_TRACKER_FILENAME: str = "camera_processing_tracker.yaml"
"""Filename for the processing tracker file placed in the output directory."""


class VideoJobNames(StrEnum):
    """Defines the job names used by the video data processing pipeline."""

    TIMESTAMPS = "camera_timestamp_extraction"
    """The name for the camera timestamp extraction job."""


def discover_log_archives(log_directory: Path) -> dict[str, Path]:
    """Discovers all VideoSystem log archives in the target directory and maps source IDs to their file paths.

    Globs for files matching the *_log.npz naming convention in the specified directory (non-recursive) and extracts
    the source ID from each filename.

    Args:
        log_directory: The path to the directory containing log archives.

    Returns:
        A dictionary mapping source ID strings to their corresponding log archive paths, sorted by source ID.

    Raises:
        FileNotFoundError: If the log_directory does not exist or is not a directory.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = (
            f"Unable to discover log archives in '{log_directory}'. The path does not exist or is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    archives: dict[str, Path] = {}
    for path in sorted(log_directory.glob(f"*{_LOG_ARCHIVE_SUFFIX}")):
        # Extracts the source ID by stripping the _log.npz suffix from the filename.
        source_id = path.name.removesuffix(_LOG_ARCHIVE_SUFFIX)
        if source_id:
            archives[source_id] = path

    return archives


def process_logs(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None = None,
    log_ids: list[str] | None = None,
    *,
    process_timestamps: bool = False,
    workers: int = -1,
) -> None:
    """Processes the requested VideoSystem log archives from the target directory.

    Supports both local and remote processing modes. In local mode (job_id is None), discovers archives, initializes
    a processing tracker in the output directory, and executes all requested jobs sequentially. In remote mode
    (job_id is provided), generates all possible job IDs for the discovered archives and executes only the matching
    job.

    Args:
        log_directory: The path to the directory containing the .npz log archives to process.
        output_directory: The path to the directory where processed output files and the tracker file are written.
            Created automatically if it does not exist.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        log_ids: An optional list of source log IDs to process. If provided, only archives matching these IDs are
            included. If not provided, all discovered archives are processed.
        process_timestamps: Determines whether to extract camera frame timestamps from the log archives.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.

    Raises:
        ValueError: If the provided job_id does not match any discoverable job, or if none of the requested log IDs
            are found in the log directory.
    """
    # Discovers all available log archives.
    all_archives = discover_log_archives(log_directory=log_directory)

    if not all_archives:
        console.echo(message=f"No log archives found in '{log_directory}'.", level=LogLevel.WARNING)
        return

    # Filters archives by the requested log IDs, if specified.
    if log_ids is not None:
        archives = {source_id: path for source_id, path in all_archives.items() if source_id in log_ids}

        # Warns about requested IDs that were not found among the discovered archives.
        missing_ids = set(log_ids) - set(archives.keys())
        if missing_ids:
            console.echo(
                message=f"Requested log IDs not found in '{log_directory}': {sorted(missing_ids)}.",
                level=LogLevel.WARNING,
            )

        if not archives:
            message = (
                f"Unable to process logs. None of the requested log IDs {log_ids} were found in "
                f"'{log_directory}'. Available IDs: {sorted(all_archives.keys())}."
            )
            console.error(message=message, error=ValueError)
    else:
        archives = all_archives

    source_ids = sorted(archives.keys())

    # If all job flags are False, treats them as all True (processes everything).
    if not process_timestamps:
        process_timestamps = True

    # Creates the output directory if it does not exist.
    output_directory.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=output_directory / _TRACKER_FILENAME)

    if job_id is not None:
        # REMOTE mode: generates all possible job IDs and matches against the provided job_id.
        all_job_ids = _generate_job_ids(source_ids=source_ids)
        id_to_source: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_source:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the discovered log archives. Valid job IDs: "
                f"{list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        source_id = id_to_source[job_id]
        _execute_job(
            log_path=archives[source_id],
            output_directory=output_directory,
            source_id=source_id,
            job_id=job_id,
            workers=workers,
            tracker=tracker,
        )
    else:
        # LOCAL mode: initializes tracker and runs all requested jobs sequentially.
        jobs_to_run: list[str] = source_ids if process_timestamps else []

        if not jobs_to_run:
            console.echo(message="No processing jobs to run.", level=LogLevel.WARNING)
            return

        console.echo(message=f"Initializing processing tracker for {len(jobs_to_run)} job(s)...")
        job_ids = _initialize_processing_tracker(output_directory=output_directory, source_ids=jobs_to_run)

        for source_id in jobs_to_run:
            _execute_job(
                log_path=archives[source_id],
                output_directory=output_directory,
                source_id=source_id,
                job_id=job_ids[source_id],
                workers=workers,
                tracker=tracker,
            )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for each source ID.

    Args:
        source_ids: The list of source ID strings for which to generate job IDs.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    job_ids: dict[str, str] = {}
    for source_id in source_ids:
        job_ids[source_id] = ProcessingTracker.generate_job_id(job_name=VideoJobNames.TIMESTAMPS, specifier=source_id)
    return job_ids


def _initialize_processing_tracker(
    output_directory: Path,
    source_ids: list[str],
) -> dict[str, str]:
    """Initializes the processing tracker file with timestamp extraction jobs for each source ID.

    Notes:
        Used to process data in the 'local' processing mode. During remote data processing, the tracker file is
        pre-generated before submitting the processing jobs to the remote compute server.

    Args:
        output_directory: The path to the output directory where the tracker file is created.
        source_ids: The source ID strings for the log archives to track.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    tracker = ProcessingTracker(file_path=output_directory / _TRACKER_FILENAME)

    # Builds the (job_name, specifier) tuples required by the tracker's initialization interface.
    jobs: list[tuple[str, str]] = [(VideoJobNames.TIMESTAMPS, source_id) for source_id in source_ids]
    tracker.initialize_jobs(jobs=jobs)

    return _generate_job_ids(source_ids=source_ids)


def _execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single timestamp extraction job for the target log archive.

    Extracts camera frame acquisition timestamps from the log archive, converts them to a Polars DataFrame, and writes
    the result as an IPC (Feather) file.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
    """
    console.echo(message=f"Running '{VideoJobNames.TIMESTAMPS}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Extracts frame acquisition timestamps from the log archive.
        timestamps = extract_logged_camera_timestamps(log_path=log_path, n_workers=workers)

        # Converts the timestamps to a Polars DataFrame and writes as an IPC (Feather) file.
        dataframe = pl.DataFrame({"frame_time_us": pl.Series(values=timestamps, dtype=pl.UInt64)})
        output_path = output_directory / f"{source_id}_timestamps.feather"
        dataframe.write_ipc(file=output_path)

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise
