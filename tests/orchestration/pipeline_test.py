"""Contains tests for the sequential processing pipeline provided by the pipeline.py module."""

import polars as pl
import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_video_system.orchestration import pipeline, discovery
from ataraxis_video_system.video.manifest import CAMERA_MANIFEST_FILENAME, write_camera_manifest
from ataraxis_video_system.video.timestamps import ExtractedDataColumns
from ataraxis_video_system.orchestration.jobs import (
    OutputLayout,
    generate_job_ids,
    resolve_tracker_path,
    resolve_timestamps_path,
    resolve_output_directory,
)
from ataraxis_video_system.orchestration.errors import OrchestrationError, OrchestrationErrors
from ataraxis_video_system.orchestration.pipeline import run_log_processing_pipeline
from ataraxis_video_system.orchestration.allocation import resolve_core_budget

_ONSET_US = 1700000000000000
"""The UTC epoch onset, in microseconds, shared by every synthetic log archive built in this module."""

_FRAME_ELAPSED_US = [1000, 2000, 3000]
"""The elapsed frame acquisition timestamps, in microseconds, written into every synthetic log archive."""

_FRAME_COLUMN = str(ExtractedDataColumns.FRAME_TIME)
"""The name of the only column the extracted timestamp table carries."""


def _archive_path(log_directory, source_id):
    """Resolves the path of the synthetic log archive of the target camera source."""
    return log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"


def _build_camera_logs(log_directory, source_ids):
    """Creates one synthetic log archive and one camera manifest entry for each of the requested camera sources."""
    log_directory.mkdir(parents=True, exist_ok=True)
    for source_id in source_ids:
        create_test_archive(
            archive_path=_archive_path(log_directory=log_directory, source_id=source_id),
            source_id=source_id,
            onset_us=_ONSET_US,
            frame_timestamps_us=_FRAME_ELAPSED_US,
        )
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")


def _read_timestamps(output_directory, source_id):
    """Reads the extracted timestamps the pipeline wrote for the target camera source."""
    feather_path = resolve_timestamps_path(
        output_directory=resolve_output_directory(output_directory=output_directory), source_id=source_id
    )
    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [_FRAME_COLUMN]
    return dataframe[_FRAME_COLUMN].to_list()


def _open_tracker(output_directory):
    """Opens the processing tracker the pipeline aligned under the target output directory."""
    return ProcessingTracker(
        file_path=resolve_tracker_path(output_directory=resolve_output_directory(output_directory=output_directory))
    )


def _record_dispatches(monkeypatch):
    """Replaces the single-job runner the pipeline dispatches with a recorder and returns the recorded calls."""
    calls = []

    def _record(**arguments):
        calls.append(arguments)

    monkeypatch.setattr(pipeline, "execute_job", _record)
    return calls


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_all_sources(tmp_path):
    """Verifies that local mode processes every source the camera manifest registers when no source IDs are given."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        workers=1,
        display_progress=False,
    )

    # The pipeline materializes its own subdirectory and tracker under the nominated output root.
    timestamps_directory = resolve_output_directory(output_directory=output_directory)
    assert timestamps_directory.is_dir()
    assert timestamps_directory.name == str(OutputLayout.DIRECTORY_NAME)
    assert resolve_tracker_path(output_directory=timestamps_directory).is_file()

    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    for source_id in ("1", "2"):
        assert _read_timestamps(output_directory=output_directory, source_id=source_id) == [
            _ONSET_US + elapsed for elapsed in _FRAME_ELAPSED_US
        ]
        assert tracker.get_job_status(job_id=job_ids[source_id]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_subset(tmp_path):
    """Verifies that local mode processes only the explicitly requested subset of the registered camera sources."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        source_ids=["1"],
        workers=1,
        display_progress=False,
    )

    timestamps_directory = resolve_output_directory(output_directory=output_directory)
    assert resolve_timestamps_path(output_directory=timestamps_directory, source_id="1").is_file()
    assert not resolve_timestamps_path(output_directory=timestamps_directory, source_id="2").exists()

    # The unrequested source stays off the tracker, as the alignment registers the prepared subset alone.
    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert job_ids["2"] not in tracker.snapshot()


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_empty_source_ids(tmp_path):
    """Verifies that an empty source ID sequence resolves through the manifest and processes every registered source."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        source_ids=[],
        workers=1,
        display_progress=False,
    )

    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    for source_id in ("1", "2"):
        assert len(_read_timestamps(output_directory=output_directory, source_id=source_id)) == len(_FRAME_ELAPSED_US)
        assert tracker.get_job_status(job_id=job_ids[source_id]) == ProcessingStatus.SUCCEEDED


def test_run_log_processing_pipeline_missing_log_directory(tmp_path):
    """Verifies that the pipeline reports the missing manifest kind when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{missing_directory}'. The path does not exist or is "
        f"not a directory."
    )
    with pytest.raises(OrchestrationError, match=error_format(message)) as error:
        run_log_processing_pipeline(
            log_directory=missing_directory,
            output_directory=tmp_path / "output",
            source_ids=["1"],
            workers=1,
            display_progress=False,
        )

    assert error.value.kind == OrchestrationErrors.MISSING_LOG_MANIFEST

    # A failed resolution materializes nothing, as the output subdirectory is created only once the jobs resolve.
    assert not (tmp_path / "output").exists()


def test_run_log_processing_pipeline_missing_manifest(tmp_path):
    """Verifies that the pipeline reports the missing manifest kind when the log directory holds no camera manifest."""
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{tmp_path}'. No {CAMERA_MANIFEST_FILENAME} was "
        f"found. A camera manifest is required to identify which log archives were produced by ataraxis-video-system."
    )
    with pytest.raises(OrchestrationError, match=error_format(message)) as error:
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            workers=1,
            display_progress=False,
        )

    assert error.value.kind == OrchestrationErrors.MISSING_LOG_MANIFEST


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_unregistered_source_id(tmp_path):
    """Verifies that the pipeline reports the unknown job source kind for a source absent from the camera manifest."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    manifest_path = log_directory / CAMERA_MANIFEST_FILENAME
    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The following requested source IDs "
        f"are not registered in the {CAMERA_MANIFEST_FILENAME} at '{manifest_path}': 3. The corresponding log "
        f"archives were not produced by ataraxis-video-system. Registered source IDs: 1, 2."
    )
    with pytest.raises(OrchestrationError, match=error_format(message)) as error:
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            source_ids=["3"],
            workers=1,
            display_progress=False,
        )

    assert error.value.kind == OrchestrationErrors.UNKNOWN_JOB_SOURCE


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_split_logger_output(tmp_path):
    """Verifies that the pipeline reports the split logger output kind when the archives span several directories."""
    log_directory = tmp_path / "logs"
    first_directory = log_directory / "a"
    second_directory = log_directory / "b"
    first_directory.mkdir(parents=True)
    second_directory.mkdir(parents=True)

    create_test_archive(
        archive_path=_archive_path(log_directory=first_directory, source_id=1),
        source_id=1,
        onset_us=_ONSET_US,
        frame_timestamps_us=_FRAME_ELAPSED_US,
    )
    create_test_archive(
        archive_path=_archive_path(log_directory=second_directory, source_id=2),
        source_id=2,
        onset_us=_ONSET_US,
        frame_timestamps_us=_FRAME_ELAPSED_US,
    )

    # Writes the single manifest at the search root, so both archives are registered under one recording.
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    parents = sorted(str(parent) for parent in (first_directory, second_directory))
    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The resolved log archives sit in 2 "
        f"different directories: {parents}. Archives in separate directories were written by separate DataLogger "
        f"instances, and one recording writes one logger, so this tree holds more than one recording. Each DataLogger "
        f"output directory must be prepared and processed on its own invocation."
    )
    with pytest.raises(OrchestrationError, match=error_format(message)) as error:
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            source_ids=["1", "2"],
            workers=1,
            display_progress=False,
        )

    assert error.value.kind == OrchestrationErrors.SPLIT_LOGGER_OUTPUT


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_external_mode(tmp_path):
    """Verifies that external mode executes only the job the canonical job identifier names."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_ids["1"],
        source_ids=["2"],
        workers=1,
        display_progress=False,
    )

    # The source ID request is ignored in external mode, so only the archive the job identifier names is processed.
    timestamps_directory = resolve_output_directory(output_directory=output_directory)
    assert resolve_timestamps_path(output_directory=timestamps_directory, source_id="1").is_file()
    assert not resolve_timestamps_path(output_directory=timestamps_directory, source_id="2").exists()

    tracker = _open_tracker(output_directory=output_directory)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert job_ids["2"] not in tracker.snapshot()


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_external_mode_unresolved_sibling(tmp_path):
    """Verifies that external mode runs the named job even when a sibling source resolves to no log archive."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    # Registers a second source the tree holds no archive for, which the named job must not be judged against.
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_ids["1"],
        workers=1,
        display_progress=False,
    )

    assert len(_read_timestamps(output_directory=output_directory, source_id="1")) == len(_FRAME_ELAPSED_US)
    assert _open_tracker(output_directory=output_directory).get_job_status(job_id=job_ids["1"]) == (
        ProcessingStatus.SUCCEEDED
    )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_unknown_job_id(tmp_path):
    """Verifies that external mode reports the unknown job identifier kind when the manifest defines no such job."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    manifest_path = log_directory / CAMERA_MANIFEST_FILENAME
    message = (
        f"Unable to prepare the camera timestamp extraction job 'invalid_job_id_value' in '{log_directory}'. The "
        f"camera manifest at '{manifest_path}' defines no job with that identifier. Registered source IDs: 1."
    )
    with pytest.raises(OrchestrationError, match=error_format(message)) as error:
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            job_id="invalid_job_id_value",
            workers=1,
            display_progress=False,
        )

    assert error.value.kind == OrchestrationErrors.UNKNOWN_JOB_ID


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_external_jobs_share_tracker(tmp_path):
    """Verifies that a second external job aligned against the full universe leaves its sibling's outcome intact."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])

    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_ids["1"],
        workers=1,
        display_progress=False,
    )
    assert _open_tracker(output_directory=output_directory).get_job_status(job_id=job_ids["1"]) == (
        ProcessingStatus.SUCCEEDED
    )

    # The second invocation shares the tracker of the first, and the alignment against the manifest universe keeps it
    # from resetting the sibling job it does not request.
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_ids["2"],
        workers=1,
        display_progress=False,
    )

    tracker = _open_tracker(output_directory=output_directory)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_ids["2"]) == ProcessingStatus.SUCCEEDED
    assert len(_read_timestamps(output_directory=output_directory, source_id="1")) == len(_FRAME_ELAPSED_US)
    assert len(_read_timestamps(output_directory=output_directory, source_id="2")) == len(_FRAME_ELAPSED_US)


def test_run_log_processing_pipeline_reads_no_archive_before_dispatch(tmp_path, monkeypatch):
    """Verifies that the pipeline dispatches its jobs without opening or sizing a single log archive."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir(parents=True)

    # Writes unreadable stand-ins for the log archives, so any read before dispatch raises instead of being missed.
    for source_id in (1, 2):
        _archive_path(log_directory=log_directory, source_id=source_id).write_bytes(b"not-a-valid-npz-archive")
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")

    def _forbidden_footprint(**arguments):
        message = f"The pipeline sized a job from its archive: {arguments}."
        raise AssertionError(message)

    monkeypatch.setattr(discovery, "resolve_archive_footprint", _forbidden_footprint)
    calls = _record_dispatches(monkeypatch=monkeypatch)

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        workers=4,
        display_progress=False,
    )

    # Every registered source reaches the runner, in ascending source identifier order.
    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert [call["source_id"] for call in calls] == ["1", "2"]
    assert [call["job_id"] for call in calls] == [job_ids["1"], job_ids["2"]]
    assert [call["log_path"] for call in calls] == [
        _archive_path(log_directory=log_directory, source_id=source_id) for source_id in (1, 2)
    ]

    # Every job is dispatched at the requested ceiling, as narrowing that width is the extraction's own business.
    resolved_output = resolve_output_directory(output_directory=output_directory)
    assert {call["workers"] for call in calls} == {resolve_core_budget(requested_budget=4)}
    assert {call["output_directory"] for call in calls} == {resolved_output}
    assert {call["display_progress"] for call in calls} == {False}
    assert {call["tracker"].file_path for call in calls} == {resolve_tracker_path(output_directory=resolved_output)}

    # The preparation still materializes the output layout and registers both jobs on the shared tracker.
    tracker_snapshot = _open_tracker(output_directory=output_directory).snapshot()
    assert sorted(tracker_snapshot) == sorted(job_ids.values())
