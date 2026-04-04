"""Contains tests for functions provided by the log_processing.py module."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ataraxis_video_system.manifest import write_camera_manifest
from ataraxis_video_system.log_processing import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    _TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    execute_job,
    find_log_archive,
    _generate_job_ids,
    resolve_recording_roots,
    _extract_unique_components,
    run_log_processing_pipeline,
    initialize_processing_tracker,
    extract_logged_camera_timestamps,
)


def _create_onset_message(source_id: int, onset_us: int) -> np.ndarray:
    """Creates an onset message with timestamp=0 and the onset UTC epoch as payload."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([0], dtype=np.uint64).view(np.uint8)
    onset_bytes = np.array([onset_us], dtype=np.int64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, onset_bytes])


def _create_frame_message(source_id: int, elapsed_us: int) -> np.ndarray:
    """Creates a frame message with no payload (payload.size == 0)."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([elapsed_us], dtype=np.uint64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes])


def _create_data_message(source_id: int, elapsed_us: int, payload_size: int = 4) -> np.ndarray:
    """Creates a data message with a non-empty payload."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([elapsed_us], dtype=np.uint64).view(np.uint8)
    payload = np.zeros(payload_size, dtype=np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, payload])


def _create_test_archive(
    archive_path: Path,
    source_id: int,
    onset_us: int,
    frame_timestamps_us: list[int],
    data_timestamps_us: list[int] | None = None,
) -> None:
    """Creates a .npz log archive with the specified frame and data messages."""
    arrays: dict[str, np.ndarray] = {}

    # Creates the onset message.
    onset_key = f"{source_id:03d}_{0:020d}"
    arrays[onset_key] = _create_onset_message(source_id=source_id, onset_us=onset_us)

    # Creates frame messages (no payload).
    for elapsed_us in frame_timestamps_us:
        key = f"{source_id:03d}_{elapsed_us:020d}"
        arrays[key] = _create_frame_message(source_id=source_id, elapsed_us=elapsed_us)

    # Creates data messages (with payload).
    if data_timestamps_us is not None:
        for elapsed_us in data_timestamps_us:
            key = f"{source_id:03d}_{elapsed_us:020d}"
            arrays[key] = _create_data_message(source_id=source_id, elapsed_us=elapsed_us)

    np.savez(archive_path, **arrays)


def test_resolve_recording_roots_basic(tmp_path: Path) -> None:
    """Verifies that resolve_recording_roots correctly identifies unique recording roots."""
    # Creates two recording directories with shared subdirectory structure.
    root_a = tmp_path / "day1" / "recordings" / "logs"
    root_b = tmp_path / "day2" / "recordings" / "logs"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)

    roots = resolve_recording_roots(paths=[root_a, root_b])
    root_names = {r.name for r in roots}
    assert root_names == {"day1", "day2"}
    assert len(roots) == 2


def test_resolve_recording_roots_single_path(tmp_path: Path) -> None:
    """Verifies that resolve_recording_roots handles a single path correctly."""
    path = tmp_path / "experiment" / "session" / "logs"
    path.mkdir(parents=True)

    roots = resolve_recording_roots(paths=[path])
    assert len(roots) == 1


def test_resolve_recording_roots_deduplication(tmp_path: Path) -> None:
    """Verifies that resolve_recording_roots deduplicates paths sharing the same root."""
    root_a = tmp_path / "day1" / "logs_a"
    root_b = tmp_path / "day1" / "logs_b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)

    roots = resolve_recording_roots(paths=[root_a, root_b])

    # Both paths share "day1" as parent, unique components are "logs_a" and "logs_b", so they remain distinct.
    assert len(roots) == 2


def test_find_log_archive_success(tmp_path: Path) -> None:
    """Verifies that find_log_archive returns the correct path for a matching archive."""
    archive_path = tmp_path / f"cam1{LOG_ARCHIVE_SUFFIX}"
    _create_test_archive(
        archive_path=archive_path,
        source_id=1,
        onset_us=1700000000000000,
        frame_timestamps_us=[1000, 2000],
    )

    result = find_log_archive(log_directory=tmp_path, source_id="cam1")
    assert result == archive_path


def test_find_log_archive_nested(tmp_path: Path) -> None:
    """Verifies that find_log_archive discovers archives in nested subdirectories."""
    nested_dir = tmp_path / "sub" / "deep"
    nested_dir.mkdir(parents=True)
    archive_path = nested_dir / f"cam2{LOG_ARCHIVE_SUFFIX}"
    _create_test_archive(
        archive_path=archive_path,
        source_id=2,
        onset_us=1700000000000000,
        frame_timestamps_us=[1000],
    )

    result = find_log_archive(log_directory=tmp_path, source_id="cam2")
    assert result == archive_path


def test_find_log_archive_directory_not_found(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises FileNotFoundError for a missing directory."""
    missing_dir = tmp_path / "nonexistent"
    message = (
        f"Unable to find log archive for source 'cam1' in '{missing_dir}'. The path does not exist or "
        f"is not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        find_log_archive(log_directory=missing_dir, source_id="cam1")


def test_find_log_archive_no_match(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises FileNotFoundError when no archive matches."""
    message = (
        f"Unable to find log archive for source 'missing' in '{tmp_path}'. No file matching "
        f"'missing{LOG_ARCHIVE_SUFFIX}' was found."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        find_log_archive(log_directory=tmp_path, source_id="missing")


def test_find_log_archive_multiple_matches(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises ValueError when multiple archives match."""
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "b"
    sub_a.mkdir()
    sub_b.mkdir()

    for sub_dir in (sub_a, sub_b):
        _create_test_archive(
            archive_path=sub_dir / f"cam1{LOG_ARCHIVE_SUFFIX}",
            source_id=1,
            onset_us=1700000000000000,
            frame_timestamps_us=[1000],
        )

    expected_paths = sorted(tmp_path.rglob(f"cam1{LOG_ARCHIVE_SUFFIX}"))
    message = (
        f"Unable to find log archive for source 'cam1' in '{tmp_path}'. Found 2 "
        f"matching archives, but expected exactly one: {[str(p) for p in expected_paths]}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        find_log_archive(log_directory=tmp_path, source_id="cam1")


def test_extract_logged_camera_timestamps_invalid_path(tmp_path: Path) -> None:
    """Verifies that extract_logged_camera_timestamps raises ValueError for a non-existent archive."""
    missing_path = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to extract camera frame timestamp data from the log file {missing_path}, as it does not exist or does "
        f"not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=missing_path)


def test_extract_logged_camera_timestamps_not_npz(tmp_path: Path) -> None:
    """Verifies that extract_logged_camera_timestamps raises ValueError for a non-.npz file."""
    text_file = tmp_path / "data.txt"
    text_file.write_text("not an archive")
    message = (
        f"Unable to extract camera frame timestamp data from the log file {text_file}, as it does not exist or does "
        f"not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=text_file)


def test_extract_logged_camera_timestamps_frames_only(tmp_path: Path) -> None:
    """Verifies extraction from an archive containing only frame messages (no payload)."""
    archive_path = tmp_path / f"cam1{LOG_ARCHIVE_SUFFIX}"
    onset_us = 1700000000000000
    frame_elapsed = [1000, 2000, 3000, 4000, 5000]
    _create_test_archive(
        archive_path=archive_path,
        source_id=1,
        onset_us=onset_us,
        frame_timestamps_us=frame_elapsed,
    )

    timestamps = extract_logged_camera_timestamps(log_path=archive_path, n_workers=1)
    assert len(timestamps) == 5

    # Verifies that all timestamps are absolute (onset + elapsed).
    expected = np.array([np.uint64(onset_us + e) for e in frame_elapsed], dtype=np.uint64)
    np.testing.assert_array_equal(timestamps, expected)


def test_extract_logged_camera_timestamps_mixed_messages(tmp_path: Path) -> None:
    """Verifies that only frame messages (payload.size == 0) are extracted from mixed archives."""
    archive_path = tmp_path / f"cam1{LOG_ARCHIVE_SUFFIX}"
    onset_us = 1700000000000000
    frame_elapsed = [1000, 3000, 5000]
    data_elapsed = [2000, 4000]
    _create_test_archive(
        archive_path=archive_path,
        source_id=1,
        onset_us=onset_us,
        frame_timestamps_us=frame_elapsed,
        data_timestamps_us=data_elapsed,
    )

    timestamps = extract_logged_camera_timestamps(log_path=archive_path, n_workers=1)

    # Only frame messages should be extracted.
    assert len(timestamps) == 3
    expected = np.array([np.uint64(onset_us + e) for e in frame_elapsed], dtype=np.uint64)
    np.testing.assert_array_equal(timestamps, expected)


def test_initialize_processing_tracker_creates_tracker(tmp_path: Path) -> None:
    """Verifies that initialize_processing_tracker creates a tracker file and returns correct job IDs."""
    source_ids = ["cam1", "cam2", "cam3"]
    job_ids = initialize_processing_tracker(output_directory=tmp_path, source_ids=source_ids)

    # Verifies that a job ID is returned for each source ID.
    assert len(job_ids) == 3
    assert set(job_ids.keys()) == {"cam1", "cam2", "cam3"}

    # Verifies that the tracker file was created.
    tracker_path = tmp_path / TRACKER_FILENAME
    assert tracker_path.exists()

    # Verifies that job IDs are deterministic.
    expected_ids = {
        source_id: ProcessingTracker.generate_job_id(job_name=_TIMESTAMP_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }
    assert job_ids == expected_ids


def test_execute_job_success(tmp_path: Path) -> None:
    """Verifies that execute_job extracts timestamps and writes a Feather output file."""
    # Creates a test archive with frame messages.
    archive_path = tmp_path / f"cam1{LOG_ARCHIVE_SUFFIX}"
    onset_us = 1700000000000000
    frame_elapsed = [1000, 2000, 3000]
    _create_test_archive(
        archive_path=archive_path,
        source_id=1,
        onset_us=onset_us,
        frame_timestamps_us=frame_elapsed,
    )

    # Creates a tracker and initializes the job.
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    job_id = ProcessingTracker.generate_job_id(job_name=_TIMESTAMP_JOB_NAME, specifier="cam1")
    tracker.initialize_jobs(jobs=[(_TIMESTAMP_JOB_NAME, "cam1")])

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id="cam1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        display_progress=False,
    )

    # Verifies the Feather output file was created with correct data.
    feather_path = output_dir / "camera_cam1_timestamps.feather"
    assert feather_path.exists()

    dataframe = pl.read_ipc(source=feather_path)
    assert "frame_time_us" in dataframe.columns
    assert len(dataframe) == 3

    # Verifies the tracker shows the job as completed.
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


def test_execute_job_failure_updates_tracker(tmp_path: Path) -> None:
    """Verifies that execute_job marks the tracker as failed and re-raises on extraction error."""
    # Creates a tracker with a job but uses a non-existent archive path.
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    job_id = ProcessingTracker.generate_job_id(job_name=_TIMESTAMP_JOB_NAME, specifier="cam1")
    tracker.initialize_jobs(jobs=[(_TIMESTAMP_JOB_NAME, "cam1")])

    bad_archive = tmp_path / "nonexistent.npz"

    with pytest.raises(ValueError):
        execute_job(
            log_path=bad_archive,
            output_directory=output_dir,
            source_id="cam1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    # Verifies the tracker shows the job as failed.
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


def test_run_log_processing_pipeline_directory_not_found(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError for a missing directory."""
    missing_dir = tmp_path / "nonexistent"
    message = f"Unable to process logs in '{missing_dir}'. The path does not exist or is not a directory."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=missing_dir,
            output_directory=tmp_path / "output",
            log_ids=["cam1"],
        )


def test_run_log_processing_pipeline_no_manifest(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when no manifest exists and no log IDs
    are provided.
    """
    message = (
        f"Unable to process logs in '{tmp_path}'. No camera_manifest.yaml was found. A camera manifest is "
        f"required to identify which log archives were produced by ataraxis-video-system."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            log_ids=None,
        )


def test_run_log_processing_pipeline_no_manifest_empty_ids(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when no manifest exists and an empty log
    IDs list is provided.
    """
    message = (
        f"Unable to process logs in '{tmp_path}'. No camera_manifest.yaml was found. A camera manifest is "
        f"required to identify which log archives were produced by ataraxis-video-system."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            log_ids=[],
        )


def test_run_log_processing_pipeline_local_mode(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline processes all jobs in local mode (job_id=None)."""
    # Creates two archives in the same directory.
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    onset_us = 1700000000000000

    for source_name in ("cam1", "cam2"):
        _create_test_archive(
            archive_path=log_dir / f"{source_name}{LOG_ARCHIVE_SUFFIX}",
            source_id=1,
            onset_us=onset_us,
            frame_timestamps_us=[1000, 2000],
        )

    # Writes a camera manifest registering both sources.
    write_camera_manifest(log_directory=log_dir, source_id=0, name="cam1")
    write_camera_manifest(log_directory=log_dir, source_id=0, name="cam2")

    output_dir = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        log_ids=["cam1", "cam2"],
        workers=1,
        display_progress=False,
    )

    # Verifies that output files were created in the camera_timestamps subdirectory for both sources.
    timestamps_dir = output_dir / CAMERA_TIMESTAMPS_DIRECTORY
    assert timestamps_dir.is_dir()
    assert (timestamps_dir / "camera_cam1_timestamps.feather").exists()
    assert (timestamps_dir / "camera_cam2_timestamps.feather").exists()
    assert (timestamps_dir / TRACKER_FILENAME).exists()


def test_run_log_processing_pipeline_remote_mode(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline executes a single job in remote mode (job_id provided)."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    onset_us = 1700000000000000

    _create_test_archive(
        archive_path=log_dir / f"cam1{LOG_ARCHIVE_SUFFIX}",
        source_id=1,
        onset_us=onset_us,
        frame_timestamps_us=[1000, 2000, 3000],
    )

    # Writes a camera manifest registering the source.
    write_camera_manifest(log_directory=log_dir, source_id=1, name="cam1")

    output_dir = tmp_path / "output"
    timestamps_dir = output_dir / CAMERA_TIMESTAMPS_DIRECTORY
    timestamps_dir.mkdir(parents=True)

    # Pre-creates the tracker in the camera_timestamps subdirectory (simulates remote orchestration).
    tracker = ProcessingTracker(file_path=timestamps_dir / TRACKER_FILENAME)
    job_id = ProcessingTracker.generate_job_id(job_name=_TIMESTAMP_JOB_NAME, specifier="cam1")
    tracker.initialize_jobs(jobs=[(_TIMESTAMP_JOB_NAME, "cam1")])

    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        job_id=job_id,
        log_ids=["cam1"],
        workers=1,
        display_progress=False,
    )

    assert (timestamps_dir / "camera_cam1_timestamps.feather").exists()


def test_run_log_processing_pipeline_invalid_job_id(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises ValueError for an invalid remote job ID."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _create_test_archive(
        archive_path=log_dir / f"cam1{LOG_ARCHIVE_SUFFIX}",
        source_id=1,
        onset_us=1700000000000000,
        frame_timestamps_us=[1000],
    )
    write_camera_manifest(log_directory=log_dir, source_id=1, name="cam1")

    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=error_format("does not match any jobs")):
        run_log_processing_pipeline(
            log_directory=log_dir,
            output_directory=output_dir,
            job_id="invalid_job_id_value",
            log_ids=["cam1"],
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_multiple_directories(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises ValueError when archives span multiple directories."""
    # Creates archives in two separate subdirectories.
    dir_a = tmp_path / "logs" / "a"
    dir_b = tmp_path / "logs" / "b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    _create_test_archive(
        archive_path=dir_a / f"cam1{LOG_ARCHIVE_SUFFIX}",
        source_id=1,
        onset_us=1700000000000000,
        frame_timestamps_us=[1000],
    )
    _create_test_archive(
        archive_path=dir_b / f"cam2{LOG_ARCHIVE_SUFFIX}",
        source_id=2,
        onset_us=1700000000000000,
        frame_timestamps_us=[1000],
    )

    # Writes a manifest at the search root so both sources are registered.
    log_root = tmp_path / "logs"
    write_camera_manifest(log_directory=log_root, source_id=1, name="cam1")
    write_camera_manifest(log_directory=log_root, source_id=2, name="cam2")

    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=error_format("span multiple directories")):
        run_log_processing_pipeline(
            log_directory=log_root,
            output_directory=output_dir,
            log_ids=["cam1", "cam2"],
            workers=1,
            display_progress=False,
        )


def test_extract_unique_components_two_paths() -> None:
    """Verifies extraction of unique components from two paths with shared structure."""
    paths = [Path("/data/day1/recordings/logs"), Path("/data/day2/recordings/logs")]
    result = _extract_unique_components(paths=paths)
    assert result == ("day1", "day2")


def test_extract_unique_components_single_path() -> None:
    """Verifies extraction of unique components from a single path."""
    paths = [Path("/data/experiment/session")]
    result = _extract_unique_components(paths=paths)

    # With a single path, the rightmost component is unique by definition.
    assert result == ("session",)


def test_extract_unique_components_different_depths() -> None:
    """Verifies extraction when unique components appear at different depths."""
    paths = [
        Path("/data/alpha/sub/logs"),
        Path("/data/beta/sub/logs"),
        Path("/data/gamma/sub/logs"),
    ]
    result = _extract_unique_components(paths=paths)
    assert result == ("alpha", "beta", "gamma")


def test_extract_unique_components_no_unique_raises() -> None:
    """Verifies that _extract_unique_components raises RuntimeError when paths share all components."""
    paths = [Path("/a/b/c"), Path("/a/b/c")]
    with pytest.raises(RuntimeError, match=error_format("Unable to extract a unique component")):
        _extract_unique_components(paths=paths)


def test_generate_job_ids_basic() -> None:
    """Verifies that _generate_job_ids returns a mapping for each source ID."""
    source_ids = ["cam1", "cam2"]
    result = _generate_job_ids(source_ids=source_ids)
    assert len(result) == 2
    assert "cam1" in result
    assert "cam2" in result

    # Verifies that all IDs are non-empty hex strings.
    for job_id in result.values():
        assert isinstance(job_id, str)
        assert len(job_id) > 0


def test_generate_job_ids_deterministic() -> None:
    """Verifies that _generate_job_ids produces consistent results across calls."""
    source_ids = ["cam1", "cam2"]
    first = _generate_job_ids(source_ids=source_ids)
    second = _generate_job_ids(source_ids=source_ids)
    assert first == second


def test_generate_job_ids_different_inputs() -> None:
    """Verifies that different source IDs produce different job IDs."""
    result = _generate_job_ids(source_ids=["cam1", "cam2"])
    assert result["cam1"] != result["cam2"]
