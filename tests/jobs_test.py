"""Contains tests for the classes and functions provided by the orchestration/jobs.py module."""

import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingTracker

from ataraxis_video_system.video.manifest import CAMERA_MANIFEST_FILENAME, CameraManifest, write_camera_manifest
from ataraxis_video_system.orchestration.jobs import (
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_PREFIX,
    CAMERA_TIMESTAMPS_SUFFIX,
    PendingJob,
    generate_job_ids,
    discover_camera_jobs,
    resolve_camera_timestamps_path,
)


def _write_archive(directory, source_id):
    """Writes a synthetic log archive for the target source into the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    create_test_archive(
        archive_path=directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        source_id=source_id,
        onset_us=1_000_000,
        frame_timestamps_us=[100, 200],
    )


def test_pending_job_creation():
    """Verifies that PendingJob stores every supplied field and applies the documented weight defaults."""
    job = PendingJob(
        log_directory="/logs",
        output_directory="/output",
        tracker_path="/output/camera_processing_tracker.yaml",
        job_id="abc123",
        source_id="1",
    )

    assert job.log_directory == "/logs"
    assert job.output_directory == "/output"
    assert job.tracker_path == "/output/camera_processing_tracker.yaml"
    assert job.job_id == "abc123"
    assert job.source_id == "1"
    assert job.core_weight == 1
    assert job.memory_mb == 0


def test_pending_job_weight_overrides():
    """Verifies that PendingJob honors explicitly supplied core and memory weights."""
    job = PendingJob(
        log_directory="/logs",
        output_directory="/output",
        tracker_path="/output/camera_processing_tracker.yaml",
        job_id="abc123",
        source_id="1",
        core_weight=5,
        memory_mb=2048,
    )

    assert job.core_weight == 5
    assert job.memory_mb == 2048


def test_pending_job_dispatch_key(tmp_path):
    """Verifies that dispatch_key pairs the stringified tracker path with the job identifier."""
    tracker_path = tmp_path / "camera_timestamps" / "camera_processing_tracker.yaml"
    job = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "output",
        tracker_path=tracker_path,
        job_id="deadbeef",
        source_id="7",
    )

    assert job.dispatch_key == (str(tracker_path), "deadbeef")


def test_pending_job_dispatch_key_separates_trackers(tmp_path):
    """Verifies that dispatch_key distinguishes identical job identifiers stored under different trackers."""
    first = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "first",
        tracker_path=tmp_path / "first" / "camera_processing_tracker.yaml",
        job_id="shared",
        source_id="1",
    )
    second = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "second",
        tracker_path=tmp_path / "second" / "camera_processing_tracker.yaml",
        job_id="shared",
        source_id="1",
    )

    assert first.dispatch_key != second.dispatch_key


def test_generate_job_ids():
    """Verifies that generate_job_ids maps every requested source to its tracker-derived job identifier."""
    job_ids = generate_job_ids(source_ids=["1", "2", "10"])

    assert set(job_ids) == {"1", "2", "10"}
    for source_id in ("1", "2", "10"):
        expected_id = ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=source_id)
        assert job_ids[source_id] == expected_id


def test_generate_job_ids_is_deterministic():
    """Verifies that generate_job_ids returns the same identifiers across repeated calls."""
    assert generate_job_ids(source_ids=["1", "2"]) == generate_job_ids(source_ids=["1", "2"])


def test_generate_job_ids_distinguishes_sources():
    """Verifies that generate_job_ids assigns a distinct identifier to every distinct source."""
    job_ids = generate_job_ids(source_ids=["1", "2", "3"])

    assert len(set(job_ids.values())) == 3


def test_generate_job_ids_empty_input():
    """Verifies that generate_job_ids returns an empty mapping when no sources are requested."""
    assert generate_job_ids(source_ids=[]) == {}


def test_resolve_camera_timestamps_path(tmp_path):
    """Verifies that resolve_camera_timestamps_path builds the feather path inside the requested directory."""
    path = resolve_camera_timestamps_path(output_directory=tmp_path, source_id="1")

    assert path == tmp_path / "camera_1_timestamps.feather"
    assert path.parent == tmp_path
    assert path.name == "camera_1_timestamps.feather"


def test_resolve_camera_timestamps_path_composition(tmp_path):
    """Verifies that resolve_camera_timestamps_path composes the filename from the prefix and suffix constants."""
    path = resolve_camera_timestamps_path(output_directory=tmp_path, source_id="42")

    assert path.name == f"{CAMERA_TIMESTAMPS_PREFIX}42{CAMERA_TIMESTAMPS_SUFFIX}"
    assert path.name.startswith(CAMERA_TIMESTAMPS_PREFIX)
    assert path.name.endswith(CAMERA_TIMESTAMPS_SUFFIX)


def test_discover_camera_jobs_missing_directory(tmp_path):
    """Verifies that discover_camera_jobs raises FileNotFoundError when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{missing_directory}'. The path does not exist or "
        f"is not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_camera_jobs(log_directory=missing_directory)


def test_discover_camera_jobs_not_a_directory(tmp_path):
    """Verifies that discover_camera_jobs raises FileNotFoundError when the log path points to a file."""
    file_path = tmp_path / "logs.txt"
    file_path.write_text("not a directory")
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{file_path}'. The path does not exist or "
        f"is not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_camera_jobs(log_directory=file_path)


def test_discover_camera_jobs_missing_manifest(tmp_path):
    """Verifies that discover_camera_jobs raises FileNotFoundError when the tree holds no camera manifest."""
    _write_archive(directory=tmp_path, source_id=1)
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{tmp_path}'. No "
        f"{CAMERA_MANIFEST_FILENAME} was found. A camera manifest is required to identify which log archives "
        f"were produced by ataraxis-video-system."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_camera_jobs(log_directory=tmp_path)


def test_discover_camera_jobs_empty_manifest(tmp_path):
    """Verifies that discover_camera_jobs raises ValueError when the camera manifest registers no sources."""
    manifest_path = tmp_path / CAMERA_MANIFEST_FILENAME
    CameraManifest(sources=[]).to_yaml(file_path=manifest_path)
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{tmp_path}'. The "
        f"{CAMERA_MANIFEST_FILENAME} at '{manifest_path}' contains no source entries."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        discover_camera_jobs(log_directory=tmp_path)


def test_discover_camera_jobs_resolves_manifest_sources(tmp_path):
    """Verifies that discover_camera_jobs returns one sorted string-specified entry per registered source."""
    for source_id in (1, 10, 2):
        write_camera_manifest(log_directory=tmp_path, source_id=source_id, name=f"cam{source_id}")
        _write_archive(directory=tmp_path, source_id=source_id)

    universe, possible = discover_camera_jobs(log_directory=tmp_path)

    assert universe == [(TIMESTAMP_JOB_NAME, "1"), (TIMESTAMP_JOB_NAME, "10"), (TIMESTAMP_JOB_NAME, "2")]
    assert possible == universe
    assert all(isinstance(specifier, str) for _, specifier in universe)


def test_discover_camera_jobs_deduplicates_repeated_sources(tmp_path):
    """Verifies that discover_camera_jobs collapses repeated manifest entries for the same source into one job."""
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1")
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1_again")
    _write_archive(directory=tmp_path, source_id=1)

    universe, possible = discover_camera_jobs(log_directory=tmp_path)

    assert universe == [(TIMESTAMP_JOB_NAME, "1")]
    assert possible == [(TIMESTAMP_JOB_NAME, "1")]


def test_discover_camera_jobs_finds_manifest_and_archives_in_subdirectories(tmp_path):
    """Verifies that discover_camera_jobs searches the whole tree for the manifest and the log archives."""
    logger_directory = tmp_path / "logger"
    logger_directory.mkdir()
    write_camera_manifest(log_directory=logger_directory, source_id=3, name="cam3")
    _write_archive(directory=logger_directory / "archives", source_id=3)

    universe, possible = discover_camera_jobs(log_directory=tmp_path)

    assert universe == [(TIMESTAMP_JOB_NAME, "3")]
    assert possible == [(TIMESTAMP_JOB_NAME, "3")]


def test_discover_camera_jobs_excludes_source_without_archive(tmp_path):
    """Verifies that a source registered without an archive stays in the universe but not in the possible set."""
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1")
    write_camera_manifest(log_directory=tmp_path, source_id=2, name="cam2")
    _write_archive(directory=tmp_path, source_id=1)

    universe, possible = discover_camera_jobs(log_directory=tmp_path)

    assert universe == [(TIMESTAMP_JOB_NAME, "1"), (TIMESTAMP_JOB_NAME, "2")]
    assert possible == [(TIMESTAMP_JOB_NAME, "1")]


def test_discover_camera_jobs_excludes_ambiguous_source(tmp_path):
    """Verifies that a source whose archive name resolves to several files is excluded from the possible set."""
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1")
    write_camera_manifest(log_directory=tmp_path, source_id=2, name="cam2")
    _write_archive(directory=tmp_path / "logger_one", source_id=1)
    _write_archive(directory=tmp_path / "logger_one", source_id=2)
    _write_archive(directory=tmp_path / "logger_two", source_id=2)

    universe, possible = discover_camera_jobs(log_directory=tmp_path)

    assert universe == [(TIMESTAMP_JOB_NAME, "1"), (TIMESTAMP_JOB_NAME, "2")]
    assert possible == [(TIMESTAMP_JOB_NAME, "1")]
