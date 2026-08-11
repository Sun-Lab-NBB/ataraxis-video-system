"""Contains tests for functions provided by the worker.py module."""

from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    PARALLEL_PROCESSING_THRESHOLD,
    ProcessingStatus,
    ProcessingTracker,
)

from ataraxis_video_system.orchestration import worker
from ataraxis_video_system.video.timestamps import ExtractedDataColumns
from ataraxis_video_system.orchestration.jobs import (
    CAMERA_EXTRACTION_JOB_NAME,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    resolve_tracker_path,
    resolve_timestamps_path,
)
from ataraxis_video_system.orchestration.worker import execute_job, run_extraction_job

_SOURCE_ID: str = "1"
"""Stores the camera source identifier used by every synthetic log archive built by this module."""

_ONSET_US: int = 1_700_000_000_000_000
"""Stores the UTC epoch onset, in microseconds, shared by every synthetic log archive built by this module."""

_FRAME_ELAPSED_US: list[int] = [1000, 2000, 3000]
"""Stores the elapsed frame acquisition timestamps, in microseconds, written into every small synthetic archive."""


class _CountingExecutor(ProcessPoolExecutor):
    """Wraps a real process pool with a counter that records how many batches were submitted to it."""

    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)
        self.submissions = 0

    def submit(self, function, /, *args, **kwargs):
        """Records the submission before handing the work to the underlying pool."""
        self.submissions += 1
        return super().submit(function, *args, **kwargs)


def _build_archive(log_directory, source_id=_SOURCE_ID, frame_timestamps_us=None, data_timestamps_us=None):
    """Creates one synthetic log archive for the requested camera source and returns the path it was written to."""
    log_directory.mkdir(parents=True, exist_ok=True)
    archive_path = log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(
        archive_path=archive_path,
        source_id=int(source_id),
        onset_us=_ONSET_US,
        frame_timestamps_us=_FRAME_ELAPSED_US if frame_timestamps_us is None else frame_timestamps_us,
        data_timestamps_us=data_timestamps_us,
    )
    return archive_path


def _initialize_tracker(tracker_path, source_id=_SOURCE_ID):
    """Creates a processing tracker that already registers the extraction job of the target camera source."""
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=[(CAMERA_EXTRACTION_JOB_NAME, source_id)])
    return tracker


def _build_descriptor(log_directory, output_directory, source_id=_SOURCE_ID, core_weight=1):
    """Builds the descriptor of the extraction job reading the target source's archive under the log directory."""
    return JobDescriptor.for_archive(
        archive_path=log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        output_directory=output_directory,
        tracker_path=resolve_tracker_path(output_directory=output_directory),
        source_id=source_id,
        log_directory=log_directory,
        core_weight=core_weight,
    )


def _expected_timestamps(frame_timestamps_us=None):
    """Converts the elapsed frame timestamps of a synthetic archive into the absolute timestamps extraction returns."""
    elapsed = _FRAME_ELAPSED_US if frame_timestamps_us is None else frame_timestamps_us
    return [_ONSET_US + elapsed_us for elapsed_us in elapsed]


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_the_timestamps_and_completes_the_job(tmp_path):
    """Verifies that execute_job writes the extracted timestamps at the resolved path and completes the tracked job."""
    log_directory = tmp_path / "logs"
    archive_path = _build_archive(log_directory=log_directory)

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        display_progress=False,
    )

    # The output file is written at the path the layout resolver names, and nowhere else.
    feather_path = resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID)
    assert feather_path.is_file()
    assert feather_path.name == "camera_1_timestamps.feather"

    # The single column carries the name the extracted data columns enumeration declares.
    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [str(ExtractedDataColumns.FRAME_TIME)]
    assert dataframe.columns == ["frame_time_us"]
    assert dataframe[str(ExtractedDataColumns.FRAME_TIME)].dtype == pl.UInt64
    assert dataframe[str(ExtractedDataColumns.FRAME_TIME)].to_list() == _expected_timestamps()

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_an_empty_table_for_an_archive_without_frames(tmp_path):
    """Verifies that execute_job still writes the timestamp column when the archive holds no frame messages."""
    log_directory = tmp_path / "logs"
    archive_path = _build_archive(log_directory=log_directory, frame_timestamps_us=[], data_timestamps_us=[1000, 2000])

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        display_progress=False,
    )

    dataframe = pl.read_ipc(source=resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID))
    assert dataframe.columns == [str(ExtractedDataColumns.FRAME_TIME)]
    assert len(dataframe) == 0
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_the_failure_message_on_the_tracker(tmp_path):
    """Verifies that a failing extraction marks the job failed with the exception's message and re-raises."""
    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    missing_archive = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to extract camera frame timestamp data from the log file {missing_archive}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        execute_job(
            log_path=missing_archive,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED

    # The tracker records the message of the exception that failed the job, rather than a generic failure note.
    assert tracker.get_job_info(job_id=job_id).error_message == str(error_info.value)

    # A failed extraction writes no output file.
    assert not resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_does_not_create_the_output_directory(tmp_path):
    """Verifies that execute_job never creates the output directory it writes into, leaving it to its caller."""
    log_directory = tmp_path / "logs"
    archive_path = _build_archive(log_directory=log_directory)

    tracker_directory = tmp_path / "tracker"
    tracker = _initialize_tracker(tracker_path=tracker_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    # The output directory is deliberately left uncreated, as creating it is the caller's responsibility.
    output_directory = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
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
    archive_path = _build_archive(log_directory=log_directory)

    output_directory = tmp_path / "output"
    output_directory.mkdir()

    # The tracker is deliberately left without any registered job, as registering the jobs is the caller's
    # responsibility.
    tracker_path = output_directory / OutputLayout.TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to mark the job with ID '{job_id}' as running using the processing tracker at '{tracker_path}'. The "
        f"requested job must be tracked by the instance, but the instance is not configured to track it. The instance "
        f"is currently configured to track jobs with IDs: ."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
        )

    assert ProcessingTracker(file_path=tracker_path).snapshot() == {}
    assert not resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_reuses_the_provided_executor(tmp_path):
    """Verifies that execute_job submits its batch work to the caller's executor and leaves that executor usable."""
    log_directory = tmp_path / "logs"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 51))
    archive_path = _build_archive(log_directory=log_directory, frame_timestamps_us=frame_timestamps_us)

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    executor = _CountingExecutor(max_workers=2)
    try:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=2,
            tracker=tracker,
            display_progress=False,
            executor=executor,
        )

        # The batch work reached the caller's pool instead of a pool the extraction opened for itself.
        assert executor.submissions > 0

        # The caller owns the pool, so the job must not shut it down. A pool closed by the job would instead raise
        # a RuntimeError when asked to accept more work.
        assert executor.submit(abs, -5).result() == 5
    finally:
        executor.shutdown(wait=True)

    dataframe = pl.read_ipc(source=resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID))
    assert dataframe.columns == [str(ExtractedDataColumns.FRAME_TIME)]
    assert dataframe[str(ExtractedDataColumns.FRAME_TIME)].to_list() == _expected_timestamps(
        frame_timestamps_us=frame_timestamps_us
    )
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_leaves_the_executor_untouched_when_running_serially(tmp_path):
    """Verifies that a single-worker job processes its archive without submitting anything to the caller's pool."""
    log_directory = tmp_path / "logs"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 51))
    archive_path = _build_archive(log_directory=log_directory, frame_timestamps_us=frame_timestamps_us)

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    executor = _CountingExecutor(max_workers=1)
    try:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            display_progress=False,
            executor=executor,
        )

        assert executor.submissions == 0
    finally:
        executor.shutdown(wait=True)

    dataframe = pl.read_ipc(source=resolve_timestamps_path(output_directory=output_directory, source_id=_SOURCE_ID))
    assert dataframe[str(ExtractedDataColumns.FRAME_TIME)].to_list() == _expected_timestamps(
        frame_timestamps_us=frame_timestamps_us
    )
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_runs_the_job_from_its_descriptor(tmp_path):
    """Verifies that run_extraction_job runs the described job and records its outcome on the descriptor's tracker."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory)

    # The tracker already registers the job before the runner runs, as the preparation stage registers it.
    _initialize_tracker(tracker_path=job.tracker_path)

    run_extraction_job(job=job)

    feather_path = resolve_timestamps_path(output_directory=job.output_directory, source_id=job.source_id)
    assert feather_path.is_file()

    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [str(ExtractedDataColumns.FRAME_TIME)]
    assert dataframe[str(ExtractedDataColumns.FRAME_TIME)].to_list() == _expected_timestamps()

    # The status is read through a tracker instance the test opens itself, which is the same file the descriptor
    # names. The runner therefore recorded the outcome at the descriptor's own tracker path.
    assert job.job_id == generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]
    assert ProcessingTracker(file_path=job.tracker_path).get_job_status(job_id=job.job_id) == (
        ProcessingStatus.SUCCEEDED
    )


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_records_a_failure_on_the_descriptor_tracker(tmp_path):
    """Verifies that a job whose archive is absent fails on the tracker the descriptor names and re-raises."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory)
    _initialize_tracker(tracker_path=job.tracker_path)

    # The archive the descriptor names was never written, so the extraction raises inside the runner.
    assert not job.archive_path.exists()

    message = (
        f"Unable to extract camera frame timestamp data from the log file {job.archive_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        run_extraction_job(job=job)

    tracker = ProcessingTracker(file_path=job.tracker_path)
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == str(error_info.value)


def test_run_extraction_job_forwards_every_descriptor_field(tmp_path, monkeypatch):
    """Verifies that run_extraction_job derives every execute_job argument from the descriptor alone."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory, core_weight=4)
    _initialize_tracker(tracker_path=job.tracker_path)

    calls = []

    def _record_call(**kwargs):
        """Records the arguments the runner derived from the descriptor instead of running the extraction."""
        calls.append(kwargs)

    monkeypatch.setattr(target=worker, name="execute_job", value=_record_call)

    run_extraction_job(job=job)

    assert len(calls) == 1
    arguments = calls[0]
    assert arguments["log_path"] == job.archive_path
    assert arguments["output_directory"] == job.output_directory
    assert arguments["source_id"] == job.source_id
    assert arguments["job_id"] == job.job_id

    # The width the descriptor carries becomes the width of the extraction pool the job's body opens.
    assert arguments["workers"] == job.core_weight == 4

    # The tracker is opened by the runner from the descriptor's own path, because a tracker's file lock cannot cross
    # a process boundary.
    assert isinstance(arguments["tracker"], ProcessingTracker)
    assert arguments["tracker"].file_path == job.tracker_path

    # A pooled job has no console to draw on, and it never receives an outer pool to nest inside.
    assert arguments["display_progress"] is False
    assert "executor" not in arguments


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_runs_inside_a_process_pool(tmp_path):
    """Verifies that the runner and its descriptor both pickle into a spawned worker and complete the job there."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory)
    _initialize_tracker(tracker_path=job.tracker_path)

    executor = ProcessPoolExecutor(max_workers=1)
    try:
        assert executor.submit(run_extraction_job, job=job).result() is None
    finally:
        executor.shutdown(wait=True)

    feather_path = resolve_timestamps_path(output_directory=job.output_directory, source_id=job.source_id)
    assert pl.read_ipc(source=feather_path)[str(ExtractedDataColumns.FRAME_TIME)].to_list() == _expected_timestamps()
    assert ProcessingTracker(file_path=job.tracker_path).get_job_status(job_id=job.job_id) == (
        ProcessingStatus.SUCCEEDED
    )
