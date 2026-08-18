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
from ataraxis_video_system.orchestration.pipeline import run_log_processing_pipeline
from ataraxis_video_system.orchestration.allocation import (
    CAMERA_EXTRACTION_JOB_CORES,
    _PARALLEL_EXTRACTION_THRESHOLD,
)

_ONSET_US: int = 1700000000000000
"""Stores the UTC epoch onset, in microseconds, shared by every synthetic log archive built in this module."""

_FRAME_ELAPSED_US: list[int] = [1000, 2000, 3000]
"""Stores the elapsed frame acquisition timestamps, in microseconds, written into every synthetic log archive."""

_FRAME_COLUMN: str = str(ExtractedDataColumns.FRAME_TIME)
"""Stores the name of the only column the extracted timestamp table carries."""


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
    """Verifies that the pipeline raises FileNotFoundError when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{missing_directory}'. The path does not exist or is "
        f"not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=missing_directory,
            output_directory=tmp_path / "output",
            source_ids=["1"],
            workers=1,
            display_progress=False,
        )

    # A failed resolution materializes nothing, as the output subdirectory is created only once the jobs resolve.
    assert not (tmp_path / "output").exists()


def test_run_log_processing_pipeline_missing_manifest(tmp_path):
    """Verifies that the pipeline fails loudly when the recording's tree holds no camera manifest."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. Its tree holds no "
        f"{CAMERA_MANIFEST_FILENAME}, so no source in it is registered and no requested source can be prepared. The "
        f"archives beneath it were not produced by ataraxis-video-system, or the recording was logged without a "
        f"manifest."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(log_directory=log_directory, output_directory=tmp_path / "output")


def test_run_log_processing_pipeline_dispatches_a_named_width_verbatim(tmp_path, monkeypatch):
    """Verifies that a positive worker count reaches every job unchanged and without any archive being sized."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir(parents=True)

    # Writes unreadable stand-ins for the log archives, so a read the named width forbids raises instead of passing
    # unnoticed.
    for source_id in (1, 2):
        _archive_path(log_directory=log_directory, source_id=source_id).write_bytes(b"not-a-valid-npz-archive")
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")

    def _forbidden_footprint(**arguments):
        message = f"The pipeline sized a job the caller had already given a width: {arguments}."
        raise AssertionError(message)

    monkeypatch.setattr(target=pipeline, name="resolve_archive_footprint", value=_forbidden_footprint)
    monkeypatch.setattr(target=discovery, name="resolve_archive_footprint", value=_forbidden_footprint)
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

    # A caller that names a width has already decided how wide its jobs run, so the figure reaches the runner as given.
    resolved_output = resolve_output_directory(output_directory=output_directory)
    assert {call["workers"] for call in calls} == {4}
    assert {call["output_directory"] for call in calls} == {resolved_output}
    assert {call["display_progress"] for call in calls} == {False}
    assert {call["tracker"].file_path for call in calls} == {resolve_tracker_path(output_directory=resolved_output)}

    # The preparation still materializes the output layout and registers both jobs on the shared tracker.
    tracker_snapshot = _open_tracker(output_directory=output_directory).snapshot()
    assert sorted(tracker_snapshot) == sorted(job_ids.values())


def test_run_log_processing_pipeline_resolves_an_unset_width_per_archive(tmp_path, monkeypatch):
    """Verifies that an unset worker count sizes each job from the archive that job reads."""
    log_directory = tmp_path / "logs"
    _build_camera_logs(log_directory=log_directory, source_ids=(1,))

    # The second source holds enough messages to earn the pooled shape, so the two jobs of one invocation resolve to
    # different widths and a single shared figure cannot satisfy both assertions below.
    create_test_archive(
        archive_path=_archive_path(log_directory=log_directory, source_id=2),
        source_id=2,
        onset_us=_ONSET_US,
        frame_timestamps_us=list(range(1, _PARALLEL_EXTRACTION_THRESHOLD + 1)),
    )
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    sized_archives = []
    footprint_resolver = pipeline.resolve_archive_footprint

    def _record_footprint(archive_path):
        sized_archives.append(archive_path)
        return footprint_resolver(archive_path=archive_path)

    monkeypatch.setattr(target=pipeline, name="resolve_archive_footprint", value=_record_footprint)
    calls = _record_dispatches(monkeypatch=monkeypatch)

    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        workers=-1,
        display_progress=False,
    )

    assert sized_archives == [_archive_path(log_directory=log_directory, source_id=source_id) for source_id in (1, 2)]
    assert [call["workers"] for call in calls] == [1, CAMERA_EXTRACTION_JOB_CORES]


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

    monkeypatch.setattr(target=pipeline, name="execute_job", value=_record)
    return calls
