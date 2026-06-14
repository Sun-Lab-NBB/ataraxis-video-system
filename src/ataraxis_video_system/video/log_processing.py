"""Provides the log data processing pipeline for extracting frame timestamps from VideoSystem log archives."""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

from .manifest import CAMERA_MANIFEST_FILENAME, CameraManifest

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""Naming convention suffix for log archives produced by assemble_log_archives()."""

TRACKER_FILENAME: str = "camera_processing_tracker.yaml"
"""Filename for the processing tracker file placed in the output directory."""

CAMERA_TIMESTAMPS_DIRECTORY: str = "camera_timestamps"
"""Name of the subdirectory created under the output path for camera timestamp processing results. All tracker files
and processed feather outputs are written into this subdirectory."""

PARALLEL_PROCESSING_THRESHOLD: int = 2000
"""The minimum number of messages in a log archive required to enable parallel processing. Archives with fewer messages
are processed sequentially to avoid multiprocessing overhead. Matches the threshold used internally by
``LogArchiveReader`` (2000).
"""

TIMESTAMP_JOB_NAME: str = "camera_timestamp_extraction"
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
    archive by source ID, aligns a processing tracker in the output directory with the requested jobs, and executes
    them sequentially. In remote mode (job_id is provided), aligns the tracker with the full job universe derived
    from the camera manifest, then resolves and executes only the single archive matching the requested job ID. The
    universe alignment lets independent remote jobs share one tracker without resetting each other's state, which
    supports running every source in parallel under an external scheduler.

    In local mode, all resolved archives must reside in the same directory. If the log_directory contains archives
    from multiple DataLogger instances (in separate subdirectories), each must be processed independently. Use the
    MCP batch processing tools to orchestrate multi-directory workflows.

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
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.
        display_progress: Determines whether to display progress bars during timestamp extraction. Defaults to True
            for interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, a requested log ID has no matching archive, or no
            camera manifest is found.
        ValueError: If the provided job_id does not match any job in the manifest universe, if no source IDs can be
            resolved, if a requested log ID matches multiple archives, or if resolved archives span multiple
            directories.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = f"Unable to process logs in '{log_directory}'. The path does not exist or is not a directory."
        console.error(message=message, error=FileNotFoundError)

    # Locates the camera manifest to resolve or validate source IDs. The manifest ensures only
    # axvs-produced log archives are processed, preventing accidental processing of logs from other
    # libraries (e.g., ataraxis-communication-interface, sl-experiment).
    candidates = sorted(log_directory.rglob(CAMERA_MANIFEST_FILENAME))
    if not candidates:
        message = (
            f"Unable to process logs in '{log_directory}'. No {CAMERA_MANIFEST_FILENAME} was found. "
            f"A camera manifest is required to identify which log archives were produced by "
            f"ataraxis-video-system."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest_path = candidates[0]
    manifest = CameraManifest.from_yaml(file_path=manifest_path)
    manifest_ids = {str(source.id) for source in manifest.sources}

    if not manifest_ids:
        message = (
            f"Unable to process logs in '{log_directory}'. The {CAMERA_MANIFEST_FILENAME} at "
            f"'{manifest_path}' contains no source entries."
        )
        console.error(message=message, error=ValueError)

    # Builds the universe of every job the manifest could produce: one timestamp-extraction job per registered
    # camera source ID. The universe is a manifest fingerprint, not an invocation fingerprint, so every invocation
    # (full, subset, or single remote job) aligns the tracker against the same set and never resets sibling jobs.
    universe_ids = sorted(manifest_ids)
    universe: list[tuple[str, str]] = [(TIMESTAMP_JOB_NAME, source_id) for source_id in universe_ids]

    # Creates the camera_timestamps subdirectory under the output path. All tracker and feather files are written here.
    timestamps_path = output_directory / CAMERA_TIMESTAMPS_DIRECTORY
    timestamps_path.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=timestamps_path / TRACKER_FILENAME)

    if job_id is not None:
        # Remote mode: selects the job to run solely by ID, validated against the manifest universe. Aligns the
        # tracker with the full universe so start_job finds the requested ID and concurrent remote jobs do not
        # treat each other's entries as foreign. Resolves only the matched archive so a missing or late sibling
        # archive cannot fail this job.
        all_job_ids = generate_job_ids(source_ids=universe_ids)
        id_to_source: dict[str, str] = {generated_id: source for source, generated_id in all_job_ids.items()}

        if job_id not in id_to_source:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the camera manifest. Valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        prepare_tracker(tracker=tracker, jobs=universe, universe=universe)

        source_id = id_to_source[job_id]
        execute_job(
            log_path=find_log_archive(log_directory=log_directory, source_id=source_id),
            output_directory=timestamps_path,
            source_id=source_id,
            job_id=job_id,
            workers=workers,
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
            invalid_ids = [source_id for source_id in log_ids if source_id not in manifest_ids]
            if invalid_ids:
                message = (
                    f"Unable to process logs in '{log_directory}'. The following source IDs are not registered "
                    f"in the {CAMERA_MANIFEST_FILENAME}: {', '.join(invalid_ids)}. Registered source IDs: "
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
        prepare_tracker(tracker=tracker, jobs=jobs, universe=universe)

        # Resolves workers once and creates a shared ProcessPoolExecutor to reuse across all jobs, avoiding
        # repeated process pool creation.
        job_ids = generate_job_ids(source_ids=source_ids)
        resolved_workers = resolve_worker_count(requested_workers=workers)
        shared_executor = ProcessPoolExecutor(max_workers=resolved_workers) if resolved_workers > 1 else None

        try:
            for source_id in source_ids:
                execute_job(
                    log_path=archive_paths[source_id],
                    output_directory=timestamps_path,
                    source_id=source_id,
                    job_id=job_ids[source_id],
                    workers=resolved_workers,
                    tracker=tracker,
                    display_progress=display_progress,
                    executor=shared_executor,
                )
        finally:
            if shared_executor is not None:
                shared_executor.shutdown(wait=True)

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def extract_logged_camera_timestamps(
    log_path: Path,
    n_workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
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

        When an external executor is provided, batch processing is submitted to that executor instead of creating a
        new ProcessPoolExecutor. The caller is responsible for executor lifecycle management. This allows multiple
        archives with similar sizes to share a single process pool, avoiding the overhead of repeatedly spawning and
        tearing down worker processes.

    Args:
        log_path: The path to the .npz log file that stores the logged data generated by the VideoSystem
            instance during runtime.
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.
        executor: An optional pre-created ProcessPoolExecutor to use for parallel batch processing. When provided,
            the function submits work to this executor instead of creating its own. The caller must ensure the
            executor's worker count matches the n_workers value used for batch generation.

    Returns:
        A contiguous numpy array of frame acquisition timestamps. Each timestamp is stored as the number of
        microseconds elapsed since the UTC epoch onset.

    Raises:
        ValueError: If the target path does not exist, does not have a .npz suffix, or does not point to a file.
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

    # Resolves the number of workers if not already resolved by the caller. External executors are pre-sized, so
    # the caller provides a positive n_workers that matches the executor's pool size.
    if n_workers < 1:
        n_workers = resolve_worker_count(requested_workers=n_workers)

    # Generates batches optimized for parallel processing. The batch_multiplier of 4 creates (workers * 4) batches
    # for over-batching, which improves load distribution when processing times vary.
    batches = reader.get_batches(workers=n_workers, batch_multiplier=4)

    if not batches:  # pragma: no cover
        return np.array([], dtype=np.uint64)

    # Passes the pre-discovered onset timestamp to worker processes so each can construct a lightweight reader that
    # skips redundant onset scanning.
    onset_us = reader.onset_timestamp_us

    # Uses the provided executor or creates a managed one for this invocation. Managed executors are shut down after
    # use; external executors are left open for reuse across multiple archives.
    managed = executor is None
    active_executor = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)

    try:
        future_to_index = {
            active_executor.submit(
                _process_frame_message_batch, log_path=log_path, keys=batch_keys, onset_us=onset_us
            ): index
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
    finally:
        if managed:
            active_executor.shutdown(wait=True)

    # Concatenates batch arrays into a single contiguous array. Filters out None placeholders from batches that
    # yielded no frame messages.
    batch_arrays = [batch for batch in results if batch is not None and batch.size > 0]
    if not batch_arrays:
        return np.array([], dtype=np.uint64)

    return np.concatenate(batch_arrays)


def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]], universe: list[tuple[str, str]]) -> None:
    """Aligns the processing tracker's job registry with the jobs requested for the current pipeline invocation.

    Notes:
        Foreign entries are detected by comparing the tracker's existing job IDs against the manifest-derived
        universe of all possible jobs for the current camera manifest, not against the invocation's requested
        subset. This lets a subset invocation or a single concurrent remote job align the tracker without
        wiping previously-completed state for sibling jobs. Any existing entries that are not part of the
        universe are treated as architectural drift (the manifest itself has changed since the tracker was last
        written) and surfaced through a warning before the tracker is rebuilt.

        If the tracker file does not yet exist on disk, the helper initializes it with the requested jobs. If
        the file exists and contains job IDs that are not part of the universe, those entries are classified as
        foreign and the helper emits a warning before resetting and reinitializing the tracker. If the file
        exists with only universe-valid entries but is missing some requested jobs, the helper performs an
        additive ``initialize_jobs`` call that registers the missing entries without clobbering any existing
        state. If the file already contains every requested job, the helper is a no-op, which keeps
        ``initialize_jobs`` from emitting duplicate-entry warnings for the fully-aligned case.

    Args:
        tracker: The ProcessingTracker instance bound to the camera_timestamps output directory.
        jobs: The list of (job_name, specifier) tuples the current pipeline invocation intends to execute.
        universe: The list of (job_name, specifier) tuples enumerating every job the manifest could produce.
            Used exclusively for foreign-entry detection.
    """
    universe_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in universe
    }
    requested_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in jobs
    }

    if not tracker.file_path.exists():
        tracker.initialize_jobs(jobs=jobs)
        return

    existing_ids = set(tracker.find_jobs(job_name="").keys())
    foreign_ids = existing_ids - universe_ids

    if foreign_ids:
        console.echo(
            message=(
                f"The processing tracker at '{tracker.file_path}' contains {len(foreign_ids)} job entries "
                f"that are not part of the current camera manifest's job universe. Resetting and reinitializing "
                f"the tracker to match the requested jobs. Foreign job IDs: {sorted(foreign_ids)}."
            ),
            level=LogLevel.WARNING,
        )
        tracker.reset()
        tracker.initialize_jobs(jobs=jobs)
        return

    if not requested_ids.issubset(existing_ids):
        tracker.initialize_jobs(jobs=jobs)


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

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output Feather file is written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        display_progress: Determines whether to display a progress bar during timestamp extraction.
        executor: An optional pre-created ProcessPoolExecutor to reuse for parallel processing. When provided,
            the executor is passed through to extract_logged_camera_timestamps to avoid creating a new process pool.
    """
    console.echo(message=f"Running '{TIMESTAMP_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Extracts frame acquisition timestamps from the log archive as a contiguous numpy array.
        timestamps = extract_logged_camera_timestamps(
            log_path=log_path, n_workers=workers, display_progress=display_progress, executor=executor
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


def generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for each source ID.

    Args:
        source_ids: The list of source ID strings for which to generate job IDs.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=source_id)
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
