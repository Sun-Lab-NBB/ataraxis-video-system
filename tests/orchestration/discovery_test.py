"""Contains tests for the classes and functions provided by the discovery.py module, and for the descriptor mapping
round trip a prepared job set exercises.
"""

import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingStatus,
    ProcessingTracker,
)

from ataraxis_video_system.orchestration import discovery
from ataraxis_video_system.video.manifest import (
    CAMERA_MANIFEST_FILENAME,
    CameraManifest,
    CameraSourceData,
    write_camera_manifest,
)
from ataraxis_video_system.orchestration.jobs import (
    CAMERA_EXTRACTION_JOB_NAME,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    resolve_tracker_path,
    resolve_output_directory,
)
from ataraxis_video_system.orchestration.discovery import (
    JobSet,
    JobSource,
    JobUniverse,
    size_job,
    prepare_jobs,
    resolve_jobs,
)
from ataraxis_video_system.orchestration.allocation import (
    CAMERA_EXTRACTION_JOB_CORES,
    _PARALLEL_EXTRACTION_THRESHOLD,
    resolve_job_workers,
    estimate_job_memory_mb,
)

_ONSET_US: int = 1700000000000000
"""Stores the UTC epoch onset, in microseconds, written into every synthetic log archive built through
_build_archive().
"""

_WIDE_ARCHIVE_MESSAGES: int = _PARALLEL_EXTRACTION_THRESHOLD
"""Stores the message count of the archive used to exercise the multi-core branch of the sizing model."""


def test_job_source_fields(tmp_path):
    """Verifies that JobSource stores the source identifier, the manifest name, and the resolved archive path."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    source = JobSource(source_id="1", name="cam1", archive_path=archive_path)

    assert source.source_id == "1"
    assert source.name == "cam1"
    assert source.archive_path == archive_path

    # The record is frozen, so a resolved source never drifts after a consumer receives it.
    with pytest.raises(AttributeError):
        source.source_id = "2"


def test_resolve_jobs_resolves_manifest_sources(tmp_path):
    """Verifies that resolve_jobs returns one sorted string-specified entry per source the manifest registers."""
    _build_recording(log_directory=tmp_path, source_ids=(1, 10, 2))

    universe = resolve_jobs(log_directory=tmp_path)

    assert isinstance(universe, JobUniverse)
    assert universe.log_directory == tmp_path
    assert universe.manifest_path == tmp_path / CAMERA_MANIFEST_FILENAME
    assert universe.universe == (
        (CAMERA_EXTRACTION_JOB_NAME, "1"),
        (CAMERA_EXTRACTION_JOB_NAME, "10"),
        (CAMERA_EXTRACTION_JOB_NAME, "2"),
    )
    assert universe.possible == universe.universe
    assert all(isinstance(specifier, str) for _, specifier in universe.universe)
    assert [source.source_id for source in universe.sources] == ["1", "10", "2"]
    assert [source.name for source in universe.sources] == ["cam1", "cam10", "cam2"]


def test_resolve_jobs_archives_property(tmp_path):
    """Verifies that the archives property keys every resolved archive by the source identifier that produced it."""
    _build_recording(log_directory=tmp_path, source_ids=(1, 2))
    write_camera_manifest(log_directory=tmp_path, source_id=3, name="cam3")

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.archives == {
        "1": tmp_path / f"1{LOG_ARCHIVE_SUFFIX}",
        "2": tmp_path / f"2{LOG_ARCHIVE_SUFFIX}",
    }
    # The source without an archive is registered, but it contributes no entry to the archive mapping.
    assert "3" not in universe.archives
    assert len(universe.sources) == 3


def test_resolve_jobs_deduplicates_repeated_sources(tmp_path):
    """Verifies that a source re-registered under a new name resolves to one job carrying the last name written."""
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1")
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1_again")
    _build_archive(directory=tmp_path, source_id=1)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CAMERA_EXTRACTION_JOB_NAME, "1"),)
    assert universe.possible == ((CAMERA_EXTRACTION_JOB_NAME, "1"),)
    # The last entry written for a repeated source supplies its colloquial name.
    assert [source.name for source in universe.sources] == ["cam1_again"]


def test_resolve_jobs_finds_manifest_and_archives_in_subdirectories(tmp_path):
    """Verifies that resolve_jobs searches the whole tree for the manifest and for the log archives."""
    logger_directory = tmp_path / "logger"
    logger_directory.mkdir()
    write_camera_manifest(log_directory=logger_directory, source_id=3, name="cam3")
    _build_archive(directory=logger_directory / "archives", source_id=3)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.manifest_path == logger_directory / CAMERA_MANIFEST_FILENAME
    assert universe.universe == ((CAMERA_EXTRACTION_JOB_NAME, "3"),)
    assert universe.possible == ((CAMERA_EXTRACTION_JOB_NAME, "3"),)
    assert universe.archives == {"3": logger_directory / "archives" / f"3{LOG_ARCHIVE_SUFFIX}"}


def test_resolve_jobs_excludes_source_without_archive(tmp_path):
    """Verifies that a source registered without an archive stays in the universe but not in the possible set."""
    _build_recording(log_directory=tmp_path, source_ids=(1,))
    write_camera_manifest(log_directory=tmp_path, source_id=2, name="cam2")

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CAMERA_EXTRACTION_JOB_NAME, "1"), (CAMERA_EXTRACTION_JOB_NAME, "2"))
    assert universe.possible == ((CAMERA_EXTRACTION_JOB_NAME, "1"),)
    assert universe.sources[1].archive_path is None


def test_resolve_jobs_excludes_ambiguous_source(tmp_path):
    """Verifies that a source whose archive name resolves to several files is excluded from the possible set."""
    write_camera_manifest(log_directory=tmp_path, source_id=1, name="cam1")
    write_camera_manifest(log_directory=tmp_path, source_id=2, name="cam2")
    _build_archive(directory=tmp_path / "logger_one", source_id=1)
    _build_archive(directory=tmp_path / "logger_one", source_id=2)
    _build_archive(directory=tmp_path / "logger_two", source_id=2)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CAMERA_EXTRACTION_JOB_NAME, "1"), (CAMERA_EXTRACTION_JOB_NAME, "2"))
    # An archive name matching several files spans several loggers, which is ambiguous rather than redundant.
    assert universe.possible == ((CAMERA_EXTRACTION_JOB_NAME, "1"),)
    assert universe.sources[1].archive_path is None


def test_resolve_jobs_writes_nothing(tmp_path):
    """Verifies that resolve_jobs leaves the log directory untouched, materializing no output and no tracker."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    before = _snapshot_tree(directory=tmp_path)

    universe = resolve_jobs(log_directory=log_directory)

    assert universe.possible == ((CAMERA_EXTRACTION_JOB_NAME, "1"), (CAMERA_EXTRACTION_JOB_NAME, "2"))
    assert _snapshot_tree(directory=tmp_path) == before
    assert not (log_directory / OutputLayout.DIRECTORY_NAME).exists()
    assert not list(tmp_path.rglob(OutputLayout.TRACKER_FILENAME))


def test_resolve_jobs_missing_directory(tmp_path):
    """Verifies that resolve_jobs raises FileNotFoundError when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{missing_directory}'. The path does not exist or is "
        f"not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        resolve_jobs(log_directory=missing_directory)


def test_resolve_jobs_not_a_directory(tmp_path):
    """Verifies that resolve_jobs raises FileNotFoundError when the log path points to a file."""
    file_path = tmp_path / "logs.txt"
    file_path.write_text("not a directory")
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{file_path}'. The path does not exist or is not a "
        f"directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        resolve_jobs(log_directory=file_path)


def test_resolve_jobs_returns_an_empty_universe_when_the_tree_holds_no_manifest(tmp_path):
    """Verifies that resolve_jobs reports a tree holding no camera manifest as holding no camera jobs."""
    _build_archive(directory=tmp_path, source_id=1)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.manifest_path is None
    assert universe.sources == ()
    assert universe.universe == ()
    assert universe.possible == ()
    assert universe.archives == {}


def test_resolve_jobs_empty_manifest(tmp_path):
    """Verifies that resolve_jobs raises ValueError when the manifest registers no sources."""
    manifest_path = tmp_path / CAMERA_MANIFEST_FILENAME
    CameraManifest(sources=[]).to_yaml(file_path=manifest_path)
    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{tmp_path}'. The {CAMERA_MANIFEST_FILENAME} at "
        f"'{manifest_path}' contains no source entries."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        resolve_jobs(log_directory=tmp_path)


def test_resolve_jobs_ambiguous_log_directory(tmp_path):
    """Verifies that resolve_jobs rejects a tree holding several manifests instead of resolving the first one."""
    _build_recording(log_directory=tmp_path / "recording_one", source_ids=(1,))
    _build_recording(log_directory=tmp_path / "recording_two", source_ids=(2,))

    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{tmp_path}'. The directory tree holds 2 "
        f"{CAMERA_MANIFEST_FILENAME} files, which means it spans several recordings or several DataLogger instances:"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        resolve_jobs(log_directory=tmp_path)
    # The report names every manifest found, so the caller can split the tree into its individual recordings.
    assert str(tmp_path / "recording_one" / CAMERA_MANIFEST_FILENAME) in str(failure.value)
    assert str(tmp_path / "recording_two" / CAMERA_MANIFEST_FILENAME) in str(failure.value)


def test_prepare_jobs_creates_output_directory_and_tracker(tmp_path):
    """Verifies that prepare_jobs materializes its own output subdirectory and the tracker recording every job."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))

    job_set = prepare_jobs(log_directory=log_directory, output_directory=output_root)

    assert isinstance(job_set, JobSet)
    assert job_set.log_directory == log_directory
    assert job_set.output_directory == resolve_output_directory(output_directory=output_root)
    assert job_set.output_directory == output_root / OutputLayout.DIRECTORY_NAME
    assert job_set.output_directory.is_dir()
    assert job_set.tracker_path == resolve_tracker_path(output_directory=job_set.output_directory)
    assert job_set.tracker_path.is_file()
    assert job_set.skipped_sources == ()


def test_prepare_jobs_builds_descriptors(tmp_path):
    """Verifies that prepare_jobs builds one fully addressed descriptor per source, in source identifier order."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 10, 2))

    job_set = prepare_jobs(log_directory=log_directory, output_directory=output_root)

    assert [job.source_id for job in job_set.jobs] == ["1", "10", "2"]
    identifiers = generate_job_ids(source_ids=["1", "10", "2"])
    for job in job_set.jobs:
        assert job.log_directory == log_directory
        assert job.archive_path == log_directory / f"{job.source_id}{LOG_ARCHIVE_SUFFIX}"
        assert job.archive_path.is_file()
        assert job.output_directory == job_set.output_directory
        assert job.tracker_path == job_set.tracker_path
        assert job.job_name == CAMERA_EXTRACTION_JOB_NAME
        assert job.job_id == identifiers[job.source_id]
        assert job.core_weight == CAMERA_EXTRACTION_JOB_CORES

    # Every descriptor addresses a distinct tracker entry, so no two jobs of one set collide during dispatch.
    assert len({job.dispatch_key for job in job_set.jobs}) == len(job_set.jobs)


def test_prepare_jobs_reads_no_archive(tmp_path, monkeypatch):
    """Verifies that prepare_jobs resolves every job without opening or sizing a single log archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))

    def _explode(**arguments):
        """Fails the call, standing in for the archive pass prepare_jobs must never perform."""
        message = f"prepare_jobs read an archive: {arguments}."
        raise AssertionError(message)

    monkeypatch.setattr(target=discovery, name="resolve_archive_footprint", value=_explode)

    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")

    assert len(job_set.jobs) == 2
    # Every job carries the declared allocation, because narrowing it to the shape its archive earns belongs to the
    # sizing pass.
    assert {job.core_weight for job in job_set.jobs} == {CAMERA_EXTRACTION_JOB_CORES}


def test_prepare_jobs_accepts_unreadable_archive(tmp_path):
    """Verifies that prepare_jobs prepares a job whose archive cannot be decoded, since it never decodes one."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=5, name="cam5")
    (log_directory / f"5{LOG_ARCHIVE_SUFFIX}").write_text("This is not a valid numpy archive.")

    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")

    assert [job.source_id for job in job_set.jobs] == ["5"]
    assert job_set.jobs[0].core_weight == CAMERA_EXTRACTION_JOB_CORES


def test_prepare_jobs_selects_requested_sources(tmp_path):
    """Verifies that prepare_jobs prepares only the requested sources while keeping the universe complete."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2, 3))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        source_ids=["3", "1"],
    )

    # The requested sources are prepared in ascending identifier order, whatever order the caller named them in.
    assert [job.source_id for job in job_set.jobs] == ["1", "3"]
    assert job_set.universe == (
        (CAMERA_EXTRACTION_JOB_NAME, "1"),
        (CAMERA_EXTRACTION_JOB_NAME, "2"),
        (CAMERA_EXTRACTION_JOB_NAME, "3"),
    )


def test_prepare_jobs_collapses_a_repeated_source_id(tmp_path):
    """Verifies that a repeated source identifier yields one descriptor rather than two sharing a dispatch key."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        source_ids=["1", "1", "2", "1"],
    )

    # Two descriptors for one source share a dispatch key, which leaves the batch engine tracking one running job
    # while two workers extract the same archive and write the same output file.
    assert [job.source_id for job in job_set.jobs] == ["1", "2"]
    assert len({job.dispatch_key for job in job_set.jobs}) == len(job_set.jobs)


def test_prepare_jobs_selects_single_job_by_id(tmp_path):
    """Verifies that a requested job identifier selects one job and overrides any requested source identifiers."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    identifiers = generate_job_ids(source_ids=["1", "2"])

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        source_ids=["1"],
        job_id=identifiers["2"],
    )

    assert [job.source_id for job in job_set.jobs] == ["2"]
    assert job_set.jobs[0].job_id == identifiers["2"]


def test_prepare_jobs_job_id_survives_missing_sibling_archive(tmp_path):
    """Verifies that a job identifier resolves against the manifest even when a sibling source has no archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")
    identifiers = generate_job_ids(source_ids=["1", "2"])

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        job_id=identifiers["1"],
    )

    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.skipped_sources == ()


def test_prepare_jobs_unknown_job_id(tmp_path):
    """Verifies that prepare_jobs raises ValueError for a job identifier the manifest omits."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))

    message = (
        f"Unable to prepare the camera timestamp extraction job 'deadbeefdeadbeef' in '{log_directory}'. The camera "
        f"manifest at"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", job_id="deadbeefdeadbeef")
    assert "deadbeefdeadbeef" in str(failure.value)


def test_prepare_jobs_unknown_source_under_strict_sourcing(tmp_path):
    """Verifies that prepare_jobs raises ValueError for a source the manifest does not register."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))

    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The following requested source IDs "
        f"are not registered in the {CAMERA_MANIFEST_FILENAME} at"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", source_ids=["1", "9"])
    assert CAMERA_MANIFEST_FILENAME in str(failure.value)


def test_prepare_jobs_missing_archive_under_strict_sourcing(tmp_path):
    """Verifies that prepare_jobs raises FileNotFoundError for a registered source holding no archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    with pytest.raises(FileNotFoundError):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", source_ids=["2"])


def test_prepare_jobs_ambiguous_archive_under_strict_sourcing(tmp_path):
    """Verifies that prepare_jobs raises FileNotFoundError when a source matches several archives."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")
    _build_archive(directory=log_directory / "logger_one", source_id=2)
    _build_archive(directory=log_directory / "logger_two", source_id=2)

    with pytest.raises(FileNotFoundError):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")


def test_prepare_jobs_records_skipped_sources_without_strict_sourcing(tmp_path):
    """Verifies that lenient sourcing records every source it cannot prepare with its reason instead of raising."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        source_ids=["1", "2", "9"],
        strict_sources=False,
    )

    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.skipped_sources == (
        ("2", "The source's log archive is absent or resolves to more than one file."),
        ("9", "The source is not registered in the camera manifest."),
    )
    # The skipped sources stay in the universe the tracker is aligned against.
    assert (CAMERA_EXTRACTION_JOB_NAME, "2") in job_set.universe


def test_prepare_jobs_returns_an_empty_set_when_lenient_sourcing_skips_every_source(tmp_path):
    """Verifies that a lenient request preparing no job returns a set rather than failing on tracker alignment."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=1, name="one")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="two")
    create_test_archive(archive_path=log_directory / "1.npz", source_id=1, onset_us=0, frame_timestamps_us=[1, 2])

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "out",
        source_ids=["2"],
        strict_sources=False,
    )

    assert job_set.jobs == ()
    assert [source_id for source_id, _ in job_set.skipped_sources] == ["2"]


def test_prepare_jobs_creates_no_output_directory_when_it_prepares_no_job(tmp_path):
    """Verifies that a lenient request preparing no job leaves the caller's output path untouched."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=1, name="one")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="two")
    create_test_archive(archive_path=log_directory / "1.npz", source_id=1, onset_us=0, frame_timestamps_us=[1, 2])
    output_directory = tmp_path / "out"

    prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        source_ids=["2"],
        strict_sources=False,
    )

    assert not resolve_output_directory(output_directory=output_directory).exists()


def test_prepare_jobs_split_logger_output(tmp_path):
    """Verifies that prepare_jobs raises ValueError when the archives span several directories."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")
    _build_archive(directory=log_directory / "logger_one", source_id=1)
    _build_archive(directory=log_directory / "logger_two", source_id=2)

    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The resolved log archives sit in 2 "
        f"different directories:"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")
    assert str(log_directory / "logger_one") in str(failure.value)
    assert str(log_directory / "logger_two") in str(failure.value)


def test_prepare_jobs_guards_run_before_any_write(tmp_path):
    """Verifies that a rejected preparation creates neither the output subdirectory nor the tracker."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")
    write_camera_manifest(log_directory=log_directory, source_id=2, name="cam2")
    _build_archive(directory=log_directory / "logger_one", source_id=1)
    _build_archive(directory=log_directory / "logger_two", source_id=2)

    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The resolved log archives sit in 2 "
        f"different directories:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=output_root)

    assert not output_root.exists()
    assert not list(tmp_path.rglob(OutputLayout.TRACKER_FILENAME))


def test_prepare_jobs_propagates_manifest_guards(tmp_path):
    """Verifies that prepare_jobs surfaces the resolution guards of the universe it prepares from."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    _build_archive(directory=log_directory, source_id=1)
    _build_archive(directory=log_directory / "second", source_id=2)
    CameraManifest(sources=[CameraSourceData(id=1, name="one")]).to_yaml(
        file_path=log_directory / CAMERA_MANIFEST_FILENAME
    )
    CameraManifest(sources=[CameraSourceData(id=2, name="two")]).to_yaml(
        file_path=log_directory / "second" / CAMERA_MANIFEST_FILENAME
    )

    message = (
        f"Unable to resolve camera timestamp extraction jobs in '{log_directory}'. The directory tree holds 2 "
        f"{CAMERA_MANIFEST_FILENAME} files,"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")


def test_prepare_jobs_rejects_a_tree_holding_no_manifest(tmp_path):
    """Verifies that prepare_jobs raises for a tree holding no camera manifest under either sourcing mode."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    _build_archive(directory=log_directory, source_id=1)

    message = (
        f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. Its tree holds no "
        f"{CAMERA_MANIFEST_FILENAME}, so no source in it is registered and no requested source can be prepared. The "
        f"archives beneath it were not produced by ataraxis-video-system, or the recording was logged without a "
        f"manifest."
    )

    # The absent manifest is a property of the directory, so both sourcing modes report it the same way.
    for strict_sources in (True, False):
        with pytest.raises(FileNotFoundError, match=error_format(message)):
            prepare_jobs(
                log_directory=log_directory,
                output_directory=tmp_path / "output",
                strict_sources=strict_sources,
            )


def test_prepare_jobs_registers_prepared_jobs_on_the_tracker(tmp_path):
    """Verifies that prepare_jobs registers every prepared job on the tracker as a scheduled job."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))

    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", source_ids=["1"])

    tracker = ProcessingTracker(file_path=job_set.tracker_path)
    identifiers = generate_job_ids(source_ids=["1", "2"])
    # Only the requested job is registered, since the universe governs foreign detection rather than registration.
    assert tracker.find_jobs() == {identifiers["1"]: (CAMERA_EXTRACTION_JOB_NAME, "1")}
    assert tracker.get_job_status(job_id=identifiers["1"]) == ProcessingStatus.SCHEDULED


def test_prepare_jobs_preserves_sibling_job_state(tmp_path):
    """Verifies that preparing one source leaves the recorded outcome of a sibling source's job untouched."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    identifiers = generate_job_ids(source_ids=["1", "2"])

    first = prepare_jobs(log_directory=log_directory, output_directory=output_root, source_ids=["1"])
    tracker = ProcessingTracker(file_path=first.tracker_path)
    tracker.start_job(job_id=identifiers["1"])
    tracker.complete_job(job_id=identifiers["1"])

    second = prepare_jobs(log_directory=log_directory, output_directory=output_root, source_ids=["2"])

    assert second.tracker_path == first.tracker_path
    tracker = ProcessingTracker(file_path=second.tracker_path)
    assert tracker.get_job_status(job_id=identifiers["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=identifiers["2"]) == ProcessingStatus.SCHEDULED


def test_prepare_jobs_discards_out_of_universe_tracker_entries(tmp_path):
    """Verifies that a tracker entry outside the manifest universe is discarded when the jobs are prepared."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1,))

    # Registers a job for a source the manifest never registered, standing in for a stale tracker.
    resolved_output = resolve_output_directory(output_directory=output_root)
    resolved_output.mkdir(parents=True)
    tracker_path = resolve_tracker_path(output_directory=resolved_output)
    foreign_job = (CAMERA_EXTRACTION_JOB_NAME, "99")
    ProcessingTracker(file_path=tracker_path).align_jobs(jobs=[foreign_job], universe=[foreign_job])

    job_set = prepare_jobs(log_directory=log_directory, output_directory=output_root)

    tracker = ProcessingTracker(file_path=job_set.tracker_path)
    identifiers = generate_job_ids(source_ids=["1", "99"])
    assert identifiers["99"] not in tracker.find_jobs()
    assert tracker.find_jobs() == {identifiers["1"]: (CAMERA_EXTRACTION_JOB_NAME, "1")}


def test_size_job_applies_the_memory_model(tmp_path):
    """Verifies that size_job reports the cores and the memory the allocation model resolves for the job's archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,), message_count=_WIDE_ARCHIVE_MESSAGES)
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")

    sized_job, sizing, footprint = size_job(job=job_set.jobs[0])

    expected_cores = resolve_job_workers(footprint=footprint)
    assert sized_job.core_weight == expected_cores
    # An archive holding the parallel extraction threshold takes the pooled shape at the declared allocation.
    assert sized_job.core_weight == CAMERA_EXTRACTION_JOB_CORES
    assert sizing.cores == expected_cores
    assert sizing.memory_mb == estimate_job_memory_mb(footprint=footprint, cores=expected_cores)
    assert footprint.message_count == _WIDE_ARCHIVE_MESSAGES
    assert footprint.archive_bytes == job_set.jobs[0].archive_path.stat().st_size


def test_size_job_narrows_a_small_archive_to_one_core(tmp_path):
    """Verifies that size_job narrows a job below the parallel extraction threshold to the sequential shape."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,), message_count=_PARALLEL_EXTRACTION_THRESHOLD - 1)
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")

    sized_job, sizing, footprint = size_job(job=job_set.jobs[0])

    # The prepared descriptor carries the declared allocation until the archive it reads is weighed against it.
    assert job_set.jobs[0].core_weight == CAMERA_EXTRACTION_JOB_CORES
    assert sized_job.core_weight == 1
    assert sizing.cores == 1
    assert footprint.message_count == _PARALLEL_EXTRACTION_THRESHOLD - 1


def test_size_job_rejects_an_unreadable_archive(tmp_path):
    """Verifies that size_job rejects an archive it cannot read rather than charging a baseline floor."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")
    (log_directory / f"1{LOG_ARCHIVE_SUFFIX}").write_text("This is not a valid numpy archive.")
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")
    message = (
        f"Unable to size the camera timestamp extraction job that reads the log archive "
        f"{job_set.jobs[0].archive_path}. The archive cannot be read, so the job reading it cannot run. Verify that "
        f"the path names a readable .npz log archive."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        size_job(job=job_set.jobs[0])


def test_size_job_preserves_descriptor_identity(tmp_path):
    """Verifies that size_job returns the supplied descriptor with its width replaced and every other field kept."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    job = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output").jobs[0]

    sized_job, _, _ = size_job(job=job)

    assert sized_job is not job
    assert sized_job.core_weight == 1
    assert sized_job.dispatch_key == job.dispatch_key
    for field_name in ("log_directory", "archive_path", "output_directory", "tracker_path", "job_name", "source_id"):
        assert getattr(sized_job, field_name) == getattr(job, field_name)


def test_job_descriptor_from_mapping_missing_key(tmp_path):
    """Verifies that JobDescriptor.from_mapping raises ValueError for an incomplete mapping."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")
    mapping = job_set.jobs[0].to_mapping()
    del mapping["archive_path"]

    message = (
        "Unable to read a camera timestamp extraction job descriptor from the supplied mapping. The following "
        "required keys are absent: archive_path."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=mapping)


def test_job_descriptor_from_mapping_unreadable_value(tmp_path):
    """Verifies that JobDescriptor.from_mapping raises ValueError for an unreadable value."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output")
    mapping = job_set.jobs[0].to_mapping()
    mapping["core_weight"] = "not an integer"

    message = (
        "Unable to read a camera timestamp extraction job descriptor from the supplied mapping. One of its values "
        "cannot be read as the type its field declares:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=mapping)


def test_job_descriptor_round_trips_through_a_mapping(tmp_path):
    """Verifies that a prepared descriptor survives the flat mapping the interface layer exchanges it through."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    job = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output").jobs[0]

    assert JobDescriptor.from_mapping(mapping=job.to_mapping()) == job


def _build_archive(directory, source_id, message_count=3):
    """Writes a synthetic log archive holding the requested number of frame messages for the target source."""
    directory.mkdir(parents=True, exist_ok=True)
    create_test_archive(
        archive_path=directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        source_id=source_id,
        onset_us=_ONSET_US,
        frame_timestamps_us=list(range(1, message_count + 1)),
    )


def _build_recording(log_directory, source_ids, message_count=3):
    """Writes one camera manifest entry and one synthetic log archive for each of the requested camera sources."""
    log_directory.mkdir(parents=True, exist_ok=True)
    for source_id in source_ids:
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")
        _build_archive(directory=log_directory, source_id=source_id, message_count=message_count)


def _snapshot_tree(directory):
    """Captures the path, size, and modification time of every filesystem entry under the target directory."""
    return {
        path: (path.is_dir(), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(directory.rglob("*"))
    }
