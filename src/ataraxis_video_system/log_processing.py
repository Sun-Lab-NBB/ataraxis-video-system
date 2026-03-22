"""Provides the log data processing pipeline for extracting frame timestamps from VideoSystem log archives."""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""Naming convention suffix for log archives produced by assemble_log_archives()."""

TRACKER_FILENAME: str = "camera_processing_tracker.yaml"
"""Filename for the processing tracker file placed in the output directory."""

PARALLEL_PROCESSING_THRESHOLD: int = 2000
"""The minimum number of messages in a log archive required to enable parallel processing. Archives with fewer messages
are processed sequentially to avoid multiprocessing overhead. Matches LogArchiveReader.PARALLEL_PROCESSING_THRESHOLD.
"""

_TIMESTAMP_JOB_NAME: str = "camera_timestamp_extraction"
"""The job name used by the processing pipeline for camera timestamp extraction."""


def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolves a set of discovered log directories to their recording root directories.

    Recording roots are the meaningful top-level directories that uniquely identify each recording session. Log
    archives and pipeline outputs may be nested at arbitrary depths below the root, but the root itself is essential
    for proper recording identification and display labels. Uses _extract_unique_components to identify the first path
    component (from the end) that uniquely distinguishes each path, then truncates each path at that component to
    strip shared structural subdirectories without assuming a fixed directory hierarchy.

    Args:
        paths: The directories containing discovered log archives. Each path is resolved to its recording root
            by walking up to the ancestor matching its unique component.

    Returns:
        A deduplicated tuple of recording root paths, one per unique recording.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    unique_ids = _extract_unique_components(paths=list(paths))
    roots: list[Path] = []
    for path, unique_id in zip(paths, unique_ids, strict=True):
        # Walks up from the path to the ancestor whose name matches the unique component.
        current = path
        while current.name != unique_id and current != current.parent:
            current = current.parent
        if current not in roots:
            roots.append(current)
    return tuple(roots)


def find_log_archive(log_directory: Path, source_id: str) -> Path:
    """Searches for a single log archive matching the target source ID under the log directory.

    Recursively searches the log_directory and all subdirectories for an archive file matching the
    ``{source_id}_log.npz`` naming convention. Expects exactly one match per source ID within the directory tree.

    Args:
        log_directory: The path to the root directory to search. The directory is searched recursively, so archives
            may be nested at any depth below this path.
        source_id: The source ID string to match. Corresponds to the filename prefix before the ``_log.npz`` suffix.

    Returns:
        The path to the discovered log archive.

    Raises:
        FileNotFoundError: If the log_directory does not exist, is not a directory, or no archive matching the
            source ID is found.
        ValueError: If multiple archives matching the source ID are found under the log directory.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. The path does not exist or "
            f"is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    matches = sorted(log_directory.rglob(f"{source_id}{LOG_ARCHIVE_SUFFIX}"))

    if not matches:
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. No file matching "
            f"'{source_id}{LOG_ARCHIVE_SUFFIX}' was found."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. Found {len(matches)} "
            f"matching archives, but expected exactly one: {[str(p) for p in matches]}."
        )
        console.error(message=message, error=ValueError)

    return matches[0]


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
    archive by source ID, initializes a processing tracker in the output directory, and executes all requested jobs
    sequentially. In remote mode (job_id is provided), generates all possible job IDs for the requested source IDs
    and executes only the matching job.

    All resolved archives must reside in the same directory. If the log_directory contains archives from multiple
    DataLogger instances (in separate subdirectories), each must be processed independently. Use the MCP batch
    processing tools to orchestrate multi-directory workflows.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the directory where processed output files and the tracker file are written.
            Created automatically if it does not exist.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        log_ids: A list of source log IDs to process. Each ID must correspond to exactly one archive under the
            log directory, and all archives must reside in the same parent directory.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.
        display_progress: Determines whether to display progress bars during timestamp extraction. Defaults to True
            for interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist or a requested log ID has no matching archive.
        ValueError: If the provided job_id does not match any discoverable job, if no log IDs are provided, if a
            requested log ID matches multiple archives, or if resolved archives span multiple directories.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = f"Unable to process logs in '{log_directory}'. The path does not exist or is not a directory."
        console.error(message=message, error=FileNotFoundError)

    if log_ids is None or not log_ids:
        message = "Unable to process logs. No log IDs were provided."
        console.error(message=message, error=ValueError)

    source_ids = sorted(log_ids)

    # Resolves all archive paths upfront and validates they belong to the same DataLogger output directory.
    archive_paths = {
        source_id: find_log_archive(log_directory=log_directory, source_id=source_id) for source_id in source_ids
    }
    parent_directories = {path.parent for path in archive_paths.values()}
    if len(parent_directories) > 1:
        message = (
            f"Unable to process logs in '{log_directory}'. The requested log archives span multiple directories: "
            f"{sorted(str(parent) for parent in parent_directories)}. Each DataLogger output directory must be "
            f"processed independently."
        )
        console.error(message=message, error=ValueError)

    # Creates the output directory if it does not exist.
    output_directory.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=output_directory / TRACKER_FILENAME)

    if job_id is not None:
        # Generates all possible job IDs and executes only the one matching the provided job_id (remote mode).
        all_job_ids = _generate_job_ids(source_ids=source_ids)
        id_to_source: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_source:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the provided log IDs. Valid job IDs: "
                f"{list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        source_id = id_to_source[job_id]
        execute_job(
            log_path=archive_paths[source_id],
            output_directory=output_directory,
            source_id=source_id,
            job_id=job_id,
            workers=workers,
            tracker=tracker,
            display_progress=display_progress,
        )
    else:
        # Initializes the tracker and runs all requested jobs sequentially (local mode).
        console.echo(message=f"Initializing processing tracker for {len(source_ids)} job(s)...")
        job_ids = initialize_processing_tracker(output_directory=output_directory, source_ids=source_ids)

        for source_id in source_ids:
            execute_job(
                log_path=archive_paths[source_id],
                output_directory=output_directory,
                source_id=source_id,
                job_id=job_ids[source_id],
                workers=workers,
                tracker=tracker,
                display_progress=display_progress,
            )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def extract_logged_camera_timestamps(
    log_path: Path,
    n_workers: int = -1,
    *,
    display_progress: bool = True,
) -> NDArray[np.uint64]:
    """Extracts the video camera frame acquisition timestamps from the target .npz log file generated by a VideoSystem
    instance during runtime.

    This function reads the '.npz' archive generated by the DataLogger's assemble_log_archives() method for a
    VideoSystem instance and, if the system saved any frames acquired by the managed camera, extracts the array of
    frame timestamps. The order of timestamps in the array is sequential and matches the order in which the frames were
    appended to the .mp4 video file.

    Notes:
        The timestamps are given as microseconds elapsed since the UTC epoch onset.

        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

        Returns a contiguous numpy array instead of a Python tuple to minimize memory footprint. For a 120 fps camera
        recording over 1.5 hours (~648,000 frames), this reduces timestamp storage from ~25 MB (Python objects) to
        ~5 MB (contiguous uint64 buffer).

    Args:
        log_path: The path to the .npz log file that stores the logged data generated by the VideoSystem
            instance during runtime.
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.

    Returns:
        A contiguous numpy array of frame acquisition timestamps. Each timestamp is stored as the number of
        microseconds elapsed since the UTC epoch onset.

    Raises:
        ValueError: If the target .npz archive does not exist.
    """
    # Validates the archive path. LogArchiveReader checks existence, but not the .npz suffix or file type.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to extract camera frame timestamp data from the log file {log_path}, as it does not exist or does "
            f"not point to a valid .npz archive."
        )
        console.error(message=message, error=ValueError)

    # Creates a reader for the target archive. The reader handles onset timestamp discovery and message key management.
    reader = LogArchiveReader(archive_path=log_path)

    # Returns early if the archive contains no data messages.
    if reader.message_count == 0:  # pragma: no cover
        return np.array([], dtype=np.uint64)

    # Processes sequentially for small archives or explicit single-worker requests to avoid multiprocessing overhead.
    if n_workers == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
        return np.array(
            [message.timestamp_us for message in reader.iter_messages() if message.payload.size == 0],
            dtype=np.uint64,
        )

    # Resolves the number of workers and generates batches optimized for parallel processing. The batch_multiplier of 4
    # creates (workers * 4) batches for over-batching, which improves load distribution when processing times vary.
    n_workers = resolve_worker_count(requested_workers=n_workers)
    batches = reader.get_batches(workers=n_workers, batch_multiplier=4)

    if not batches:  # pragma: no cover
        return np.array([], dtype=np.uint64)

    # Passes the pre-discovered onset timestamp to worker processes so each can construct a lightweight reader that
    # skips redundant onset scanning.
    onset_us = reader.onset_timestamp_us

    # Processes batches using ProcessPoolExecutor. Each worker returns a contiguous numpy array of timestamps,
    # avoiding Python object overhead for each individual timestamp value.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(_process_frame_message_batch, log_path=log_path, keys=batch_keys, onset_us=onset_us): index
            for index, batch_keys in enumerate(batches)
        }

        # Collects results while maintaining frame order.
        results: list[NDArray[np.uint64] | None] = [None] * len(batches)

        if display_progress:
            with console.progress(
                total=len(batches), description="Extracting camera frame timestamps", unit="batch"
            ) as pbar:
                for future in as_completed(future_to_index):
                    results[future_to_index[future]] = future.result()
                    pbar.update(1)
        else:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

    # Concatenates batch arrays into a single contiguous array. Filters out None placeholders from batches that
    # yielded no frame messages.
    batch_arrays = [batch for batch in results if batch is not None and batch.size > 0]
    if not batch_arrays:
        return np.array([], dtype=np.uint64)

    return np.concatenate(batch_arrays)


def initialize_processing_tracker(
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
    tracker = ProcessingTracker(file_path=output_directory / TRACKER_FILENAME)

    # Builds the (job_name, specifier) tuples required by the tracker's initialization interface.
    jobs: list[tuple[str, str]] = [(_TIMESTAMP_JOB_NAME, source_id) for source_id in source_ids]
    tracker.initialize_jobs(jobs=jobs)

    return _generate_job_ids(source_ids=source_ids)


def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    *,
    display_progress: bool = True,
) -> None:
    """Executes a single timestamp extraction job for the target log archive.

    Extracts camera frame acquisition timestamps from the log archive, converts them to a Polars DataFrame, and
    writes the result as an IPC (Feather) file.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        display_progress: Determines whether to display a progress bar during timestamp extraction.
    """
    console.echo(message=f"Running '{_TIMESTAMP_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Extracts frame acquisition timestamps from the log archive as a contiguous numpy array.
        timestamps = extract_logged_camera_timestamps(
            log_path=log_path, n_workers=workers, display_progress=display_progress
        )

        # Wraps the numpy array in a Polars DataFrame for Feather output. Polars can reference the numpy buffer
        # directly, avoiding a full copy of the timestamp data.
        dataframe = pl.DataFrame({"frame_time_us": pl.Series(name="frame_time_us", values=timestamps)})
        output_path = output_directory / f"camera_{source_id}_timestamps.feather"
        dataframe.write_ipc(file=output_path)

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Adapts the processing pipeline to directory structures where the unique recording identifier appears at different
    levels of the path hierarchy. For example, given paths like ``/data/day1/recording`` and ``/data/day2/recording``,
    identifies ``day1`` and ``day2`` as the unique components (not ``recording``, which is shared).

    Args:
        paths: The list or tuple of Path objects to extract unique components from.

    Returns:
        A tuple of unique component strings, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    paths_list = list(paths)
    unique_components: list[str] = []

    for index, path in enumerate(paths_list):
        # Iterates components from right to left to find the first one unique to this path.
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks whether this component appears in any other path.
            is_unique = all(
                component not in other_path.parts
                for other_index, other_path in enumerate(paths_list)
                if other_index != index
            )

            if is_unique:
                unique_components.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"Unable to extract a unique component from the given path: {path}."
            console.error(message=message, error=RuntimeError)

    return tuple(unique_components)


def _generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for each source ID.

    Args:
        source_ids: The list of source ID strings for which to generate job IDs.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=_TIMESTAMP_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def _process_frame_message_batch(
    log_path: Path, keys: list[str], onset_us: np.uint64
) -> NDArray[np.uint64]:  # pragma: no cover
    """Processes a batch of messages from a VideoSystem log archive to extract frame timestamps.

    This worker function is designed for parallel execution via ProcessPoolExecutor. Each worker creates its own
    LogArchiveReader instance with the pre-discovered onset timestamp to avoid redundant archive scanning. Returns
    a contiguous numpy array to minimize IPC serialization overhead and avoid Python object-per-timestamp memory cost.

    Args:
        log_path: The path to the .npz log archive file.
        keys: The message keys to process in this batch.
        onset_us: The pre-discovered onset timestamp in microseconds since epoch.

    Returns:
        A contiguous numpy array of absolute frame acquisition timestamps in microseconds since UTC epoch.
    """
    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    return np.array(
        [message.timestamp_us for message in reader.iter_messages(keys=keys) if message.payload.size == 0],
        dtype=np.uint64,
    )
