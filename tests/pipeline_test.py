"""Contains tests for functions provided by the orchestration/pipeline.py module."""

import polars as pl
import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_video_system.video.manifest import write_camera_manifest
from ataraxis_video_system.orchestration.jobs import (
    TRACKER_FILENAME,
    FRAME_TIME_COLUMN,
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    generate_job_ids,
    resolve_camera_timestamps_path,
)
from ataraxis_video_system.orchestration.pipeline import (
    execute_job,
    _execute_sized_job,
    run_log_processing_pipeline,
)

_ONSET_US = 1700000000000000
"""The UTC epoch onset, in microseconds, shared by every synthetic log archive built in this module."""

_FRAME_ELAPSED_US = [1000, 2000, 3000]
"""The elapsed frame acquisition timestamps, in microseconds, written into every synthetic log archive."""


def _build_camera_logs(log_directory, source_ids):
    """Creates one synthetic log archive and one camera manifest entry for each of the requested camera sources."""
    log_directory.mkdir(parents=True, exist_ok=True)
    for source_id in source_ids:
        create_test_archive(
            archive_path=log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
            source_id=source_id,
            onset_us=_ONSET_US,
            frame_timestamps_us=_FRAME_ELAPSED_US,
        )
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")


def _initialize_tracker(tracker_path, source_id):
    """Creates a processing tracker that already registers the extraction job of the target source."""
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=[(TIMESTAMP_JOB_NAME, source_id)])
    return tracker


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_success(tmp_path):
    """Verifies that execute_job extracts the frame timestamps, writes them as a feather file, and completes the job."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    execute_job(
        log_path=log_directory / f"1{LOG_ARCHIVE_SUFFIX}",
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        display_progress=False,
    )

    feather_path = resolve_camera_timestamps_path(output_directory=output_directory, source_id="1")
    assert feather_path.is_file()

    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [FRAME_TIME_COLUMN]
    assert dataframe[FRAME_TIME_COLUMN].to_list() == [_ONSET_US + elapsed for elapsed in _FRAME_ELAPSED_US]
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_failure_updates_tracker(tmp_path):
    """Verifies that execute_job marks the job as failed and re-raises when the timestamp extraction raises."""
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    missing_archive = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to extract camera frame timestamp data from the log file {missing_archive}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=missing_archive,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_does_not_create_output_directory(tmp_path):
    """Verifies that execute_job never creates the output directory it writes into, leaving it to its caller."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    tracker_directory = tmp_path / "tracker"
    tracker_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=tracker_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    # The output directory is deliberately left uncreated, as creating it is the caller's responsibility.
    output_directory = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        execute_job(
            log_path=log_directory / f"1{LOG_ARCHIVE_SUFFIX}",
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    assert not output_directory.exists()
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_does_not_register_tracker_jobs(tmp_path):
    """Verifies that execute_job never registers its job on the tracker it is handed, leaving it to its caller."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    output_directory = tmp_path / "output"
    output_directory.mkdir()

    # The tracker is deliberately left without any registered job, as registering the jobs is the caller's
    # responsibility.
    tracker_path = output_directory / TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = generate_job_ids(source_ids=["1"])["1"]

    message = (
        f"Unable to mark the job with ID '{job_id}' as running using the processing tracker at '{tracker_path}'. The "
        f"requested job must be tracked by the instance, but the instance is not configured to track it. The instance "
        f"is currently configured to track jobs with IDs: ."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=log_directory / f"1{LOG_ARCHIVE_SUFFIX}",
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    assert ProcessingTracker(file_path=tracker_path).snapshot() == {}
    assert not resolve_camera_timestamps_path(output_directory=output_directory, source_id="1").exists()


def test_run_log_processing_pipeline_directory_not_found(tmp_path):
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{missing_directory}'. The path does not exist or "
        f"is not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=missing_directory,
            output_directory=tmp_path / "output",
            log_ids=["1"],
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_no_manifest(tmp_path):
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when the log directory holds no manifest."""
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{tmp_path}'. No camera_manifest.yaml was found. A "
        f"camera manifest is required to identify which log archives were produced by ataraxis-video-system."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            log_ids=None,
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_no_manifest_empty_ids(tmp_path):
    """Verifies that an empty log ID list still resolves through the manifest and raises when no manifest exists."""
    message = (
        f"Unable to discover camera timestamp extraction jobs in '{tmp_path}'. No camera_manifest.yaml was found. A "
        f"camera manifest is required to identify which log archives were produced by ataraxis-video-system."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            log_ids=[],
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_all_sources(tmp_path):
    """Verifies that local mode processes every source the camera manifest registers when no log IDs are provided."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        log_ids=None,
        workers=1,
        display_progress=False,
    )

    timestamps_directory = output_directory / CAMERA_TIMESTAMPS_DIRECTORY
    assert timestamps_directory.is_dir()
    assert (timestamps_directory / TRACKER_FILENAME).is_file()

    tracker = ProcessingTracker(file_path=timestamps_directory / TRACKER_FILENAME)
    for source_id in ("1", "2"):
        feather_path = resolve_camera_timestamps_path(output_directory=timestamps_directory, source_id=source_id)
        assert feather_path.is_file()

        dataframe = pl.read_ipc(source=feather_path)
        assert dataframe.columns == [FRAME_TIME_COLUMN]
        assert len(dataframe) == len(_FRAME_ELAPSED_US)
        assert tracker.get_job_status(job_id=generate_job_ids(source_ids=[source_id])[source_id]) == (
            ProcessingStatus.SUCCEEDED
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_subset(tmp_path):
    """Verifies that local mode processes only the explicitly requested subset of the registered camera sources."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        log_ids=["1"],
        workers=1,
        display_progress=False,
    )

    timestamps_directory = output_directory / CAMERA_TIMESTAMPS_DIRECTORY
    assert resolve_camera_timestamps_path(output_directory=timestamps_directory, source_id="1").is_file()
    assert not resolve_camera_timestamps_path(output_directory=timestamps_directory, source_id="2").exists()

    # The unrequested source is never registered on the tracker, as only the requested subset is aligned.
    tracker = ProcessingTracker(file_path=timestamps_directory / TRACKER_FILENAME)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert job_ids["2"] not in tracker.snapshot()


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_unregistered_source_id(tmp_path):
    """Verifies that local mode raises ValueError when a requested source ID is absent from the camera manifest."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    message = (
        f"Unable to process logs in '{log_directory}'. The following source IDs are not registered in the camera "
        f"manifest: 3. Registered source IDs: 1, 2."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            log_ids=["3"],
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_multiple_directories(tmp_path):
    """Verifies that local mode raises ValueError when the resolved log archives span multiple directories."""
    log_directory = tmp_path / "logs"
    first_directory = log_directory / "a"
    second_directory = log_directory / "b"
    first_directory.mkdir(parents=True)
    second_directory.mkdir(parents=True)

    create_test_archive(
        archive_path=first_directory / f"1{LOG_ARCHIVE_SUFFIX}",
        source_id=1,
        onset_us=_ONSET_US,
        frame_timestamps_us=_FRAME_ELAPSED_US,
    )
    create_test_archive(
        archive_path=second_directory / f"2{LOG_ARCHIVE_SUFFIX}",
        source_id=2,
        onset_us=_ONSET_US,
        frame_timestamps_us=_FRAME_ELAPSED_US,
    )

    # Writes the manifest at the search root, so both archives are registered under a single manifest.
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    parents = sorted(str(parent) for parent in (first_directory, second_directory))
    message = (
        f"Unable to process logs in '{log_directory}'. The requested log archives span multiple directories: "
        f"{parents}. Each DataLogger output directory must be processed independently."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            log_ids=["1", "2"],
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_remote_mode(tmp_path):
    """Verifies that remote mode creates its own tracker and executes only the job matching the requested job ID."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_id = generate_job_ids(source_ids=["1"])["1"]
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_id,
        log_ids=["2"],
        workers=1,
        display_progress=False,
    )

    # The log_ids argument is ignored in remote mode, so only the archive matching the job ID is processed.
    timestamps_directory = output_directory / CAMERA_TIMESTAMPS_DIRECTORY
    assert resolve_camera_timestamps_path(output_directory=timestamps_directory, source_id="1").is_file()
    assert not resolve_camera_timestamps_path(output_directory=timestamps_directory, source_id="2").exists()

    tracker = ProcessingTracker(file_path=timestamps_directory / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_invalid_job_id(tmp_path):
    """Verifies that remote mode raises ValueError when the requested job ID names no job in the manifest universe."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    output_directory = tmp_path / "output"
    tracker_path = output_directory / CAMERA_TIMESTAMPS_DIRECTORY / TRACKER_FILENAME
    known_ids = sorted(generate_job_ids(source_ids=["1"]).values())
    message = (
        f"Unable to resolve the job with ID 'invalid_job_id_value' against the job universe of the processing tracker "
        f"at '{tracker_path}'. The identifier must name a job the pipeline could produce, but the universe holds only "
        f"the jobs with IDs: {', '.join(known_ids)}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=output_directory,
            job_id="invalid_job_id_value",
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_remote_jobs_share_tracker(tmp_path):
    """Verifies that two independent remote jobs sharing one tracker both succeed without resetting each other."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])

    # Dispatches each source as its own remote job against the same output directory, and therefore the same tracker.
    for source_id in ("1", "2"):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=output_directory,
            job_id=job_ids[source_id],
            workers=1,
            display_progress=False,
        )

    tracker = ProcessingTracker(file_path=output_directory / CAMERA_TIMESTAMPS_DIRECTORY / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_ids["2"]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_sized_job(tmp_path):
    """Verifies that _execute_sized_job sizes the job from its own archive and executes it at the resolved width."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    _execute_sized_job(
        log_path=log_directory / f"1{LOG_ARCHIVE_SUFFIX}",
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        ceiling=4,
        tracker=tracker,
        display_progress=False,
    )

    feather_path = resolve_camera_timestamps_path(output_directory=output_directory, source_id="1")
    assert feather_path.is_file()
    assert len(pl.read_ipc(source=feather_path)) == len(_FRAME_ELAPSED_US)
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
