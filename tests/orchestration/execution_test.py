"""Contains tests for the classes and functions provided by the execution.py module."""

from threading import Thread

import pytest
from tests.log_archives import create_test_archive
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_video_system.video.manifest import write_camera_manifest
from ataraxis_video_system.orchestration.jobs import (
    TRACKER_FILENAME,
    TIMESTAMP_JOB_NAME,
    PendingJob,
    generate_job_ids,
    resolve_camera_timestamps_path,
)
from ataraxis_video_system.orchestration.execution import (
    JobExecutionState,
    _run_job,
    size_pending_job,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
    _select_admissible_jobs,
)

_ONSET_US: int = 1700000000000000
"""Stores the UTC epoch onset written into every synthetic log archive built by this module."""

_MANAGER_TIMEOUT_SECONDS: int = 180
"""Stores the time a test waits for the execution manager thread to drain its queues before failing."""


@pytest.fixture
def execution_state_guard():
    """Clears the module-global execution state after a test that replaces it, so no state leaks between tests."""
    yield
    set_execution_state(state=None)


def _build_job(directory, source_id, core_weight=1, memory_mb=0):
    """Builds a pending job rooted in the target directory and carrying the requested core and memory weights."""
    return PendingJob(
        log_directory=directory,
        output_directory=directory,
        tracker_path=directory / TRACKER_FILENAME,
        job_id=f"job_{source_id}",
        source_id=source_id,
        core_weight=core_weight,
        memory_mb=memory_mb,
    )


def _build_archive(directory, source_id, frame_count):
    """Creates a synthetic log archive for the target source and returns its path."""
    archive_path = directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(
        archive_path=archive_path,
        source_id=source_id,
        onset_us=_ONSET_US,
        frame_timestamps_us=[1000 * (index + 1) for index in range(frame_count)],
    )
    return archive_path


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_empty_pending():
    """Verifies that selecting from an empty queue returns an empty admitted list and an empty deferred list."""
    admitted, deferred = _select_admissible_jobs(
        pending=[],
        core_budget=8,
        memory_budget_mb=8192,
        used_cores=0,
        used_memory_mb=0,
    )

    assert admitted == []
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_forces_oversized_job_when_idle(tmp_path):
    """Verifies that a job wider than the whole budget is still admitted while nothing else is running."""
    oversized = _build_job(directory=tmp_path, source_id="1", core_weight=16, memory_mb=8192)

    admitted, deferred = _select_admissible_jobs(
        pending=[oversized],
        core_budget=4,
        memory_budget_mb=1024,
        used_cores=0,
        used_memory_mb=0,
    )

    assert admitted == [oversized]
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_defers_oversized_job_when_busy(tmp_path):
    """Verifies that a job wider than the whole budget is deferred while another job is already running."""
    oversized = _build_job(directory=tmp_path, source_id="1", core_weight=16, memory_mb=8192)

    admitted, deferred = _select_admissible_jobs(
        pending=[oversized],
        core_budget=4,
        memory_budget_mb=1024,
        used_cores=1,
        used_memory_mb=1024,
    )

    assert admitted == []
    assert deferred == [oversized]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_backfills_past_unfitting_job(tmp_path):
    """Verifies that the scan continues past a job that does not fit, so a lighter job takes the spare capacity."""
    heavy = _build_job(directory=tmp_path, source_id="1", core_weight=8, memory_mb=1000)
    light = _build_job(directory=tmp_path, source_id="2", core_weight=2, memory_mb=500)

    admitted, deferred = _select_admissible_jobs(
        pending=[heavy, light],
        core_budget=4,
        memory_budget_mb=10000,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [light]
    assert deferred == [heavy]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_gated_by_memory(tmp_path):
    """Verifies that a job whose memory exceeds the remaining budget is deferred even when every core is free."""
    job = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=512)

    admitted, deferred = _select_admissible_jobs(
        pending=[job],
        core_budget=64,
        memory_budget_mb=1024,
        used_cores=1,
        used_memory_mb=1000,
    )

    assert admitted == []
    assert deferred == [job]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_gated_by_cores(tmp_path):
    """Verifies that a job whose cores exceed the remaining budget is deferred even when memory is abundant."""
    job = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=10)

    admitted, deferred = _select_admissible_jobs(
        pending=[job],
        core_budget=4,
        memory_budget_mb=100000,
        used_cores=4,
        used_memory_mb=0,
    )

    assert admitted == []
    assert deferred == [job]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_considers_heaviest_first(tmp_path):
    """Verifies that the heaviest job is weighed against the budgets first, so it is placed ahead of a lighter one."""
    light = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1024)
    heavy = _build_job(directory=tmp_path, source_id="2", core_weight=1, memory_mb=2048)

    admitted, deferred = _select_admissible_jobs(
        pending=[light, heavy],
        core_budget=100,
        memory_budget_mb=2048,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [heavy]
    assert deferred == [light]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_breaks_memory_ties_by_core_weight(tmp_path):
    """Verifies that two jobs holding equal memory are ordered by their core weight, widest first."""
    narrow = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1024)
    wide = _build_job(directory=tmp_path, source_id="2", core_weight=3, memory_mb=1024)

    admitted, deferred = _select_admissible_jobs(
        pending=[narrow, wide],
        core_budget=4,
        memory_budget_mb=100000,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [wide]
    assert deferred == [narrow]


@pytest.mark.xdist_group(name="orchestration")
def test_size_pending_job_resolves_archive(tmp_path):
    """Verifies that sizing a job backed by a real archive reads the archive and sets the job's weights in place."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    _build_archive(directory=log_directory, source_id=1, frame_count=3)

    job = _build_job(directory=tmp_path, source_id="1", core_weight=99, memory_mb=99)
    job.log_directory = log_directory

    footprint = size_pending_job(job=job, core_budget=8)

    assert footprint.modeled
    assert footprint.message_count == 3
    assert footprint.archive_bytes > 0
    assert job.core_weight == 1
    assert job.memory_mb >= 1024
    assert job.memory_mb % 1024 == 0


@pytest.mark.xdist_group(name="orchestration")
def test_size_pending_job_missing_archive(tmp_path):
    """Verifies that sizing a job whose archive cannot be resolved falls back to the single-core baseline."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    job = _build_job(directory=tmp_path, source_id="1", core_weight=99, memory_mb=99)
    job.log_directory = log_directory

    footprint = size_pending_job(job=job, core_budget=8)

    assert not footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == 0
    assert job.core_weight == 1
    assert job.memory_mb >= 1024


@pytest.mark.xdist_group(name="orchestration")
def test_group_jobs_by_tracker(tmp_path):
    """Verifies that every job in the registry is grouped under the tracker path that records it."""
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()

    first_jobs = [_build_job(directory=first_directory, source_id=str(index)) for index in (1, 2, 3)]
    second_jobs = [_build_job(directory=second_directory, source_id=str(index)) for index in (4, 5)]

    state = JobExecutionState()
    for job in first_jobs + second_jobs:
        state.all_jobs[job.dispatch_key] = job

    grouped = group_jobs_by_tracker(state=state)

    assert set(grouped.keys()) == {first_directory / TRACKER_FILENAME, second_directory / TRACKER_FILENAME}
    assert grouped[first_directory / TRACKER_FILENAME] == first_jobs
    assert grouped[second_directory / TRACKER_FILENAME] == second_jobs


@pytest.mark.xdist_group(name="orchestration")
def test_execution_state_round_trip(execution_state_guard):
    """Verifies that the stored execution state is returned unchanged and can be cleared back to None."""
    assert get_execution_state() is None

    state = JobExecutionState(core_budget=4, memory_budget_mb=4096)
    set_execution_state(state=state)
    assert get_execution_state() is state

    replacement = JobExecutionState(core_budget=2, memory_budget_mb=2048)
    set_execution_state(state=replacement)
    assert get_execution_state() is replacement

    set_execution_state(state=None)
    assert get_execution_state() is None


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_dispatches_pending_jobs(tmp_path):
    """Verifies that the manager dispatches every queued job and drains its queues once the jobs finish."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    for source_id in (1, 2):
        _build_archive(directory=log_directory, source_id=source_id, frame_count=4)
        write_camera_manifest(log_directory=log_directory, source_id=source_id, name=f"cam{source_id}")

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker_path = output_directory / TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)

    universe = [(TIMESTAMP_JOB_NAME, "1"), (TIMESTAMP_JOB_NAME, "2")]
    tracker.align_jobs(jobs=universe, universe=universe)
    job_ids = generate_job_ids(source_ids=["1", "2"])

    state = JobExecutionState(core_budget=4, memory_budget_mb=8192)
    for source_id in ("1", "2"):
        job = PendingJob(
            log_directory=log_directory,
            output_directory=output_directory,
            tracker_path=tracker_path,
            job_id=job_ids[source_id],
            source_id=source_id,
        )
        size_pending_job(job=job, core_budget=state.core_budget)
        state.all_jobs[job.dispatch_key] = job
        state.pending_jobs.append(job)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pending_jobs == []
    assert state.active_jobs == []

    for source_id in ("1", "2"):
        assert tracker.get_job_status(job_id=job_ids[source_id]) == ProcessingStatus.SUCCEEDED
        assert resolve_camera_timestamps_path(output_directory=output_directory, source_id=source_id).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_cancellation_skips_dispatch(tmp_path):
    """Verifies that a canceled session exits without dispatching any of the jobs still queued for execution."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    _build_archive(directory=log_directory, source_id=1, frame_count=4)
    write_camera_manifest(log_directory=log_directory, source_id=1, name="cam1")

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker_path = output_directory / TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=[(TIMESTAMP_JOB_NAME, "1")], universe=[(TIMESTAMP_JOB_NAME, "1")])
    job_ids = generate_job_ids(source_ids=["1"])

    job = PendingJob(
        log_directory=log_directory,
        output_directory=output_directory,
        tracker_path=tracker_path,
        job_id=job_ids["1"],
        source_id="1",
    )
    size_pending_job(job=job, core_budget=4)

    state = JobExecutionState(core_budget=4, memory_budget_mb=8192, canceled=True)
    state.all_jobs[job.dispatch_key] = job
    state.pending_jobs.append(job)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pending_jobs == [job]
    assert state.active_jobs == []
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SCHEDULED
    assert not resolve_camera_timestamps_path(output_directory=output_directory, source_id="1").exists()


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_marks_unresolvable_archive_as_failed(tmp_path):
    """Verifies that a job whose archive cannot be resolved is recorded as failed rather than left unfinished."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker_path = output_directory / TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=[(TIMESTAMP_JOB_NAME, "1")])
    job_ids = generate_job_ids(source_ids=["1"])

    job = PendingJob(
        log_directory=log_directory,
        output_directory=output_directory,
        tracker_path=tracker_path,
        job_id=job_ids["1"],
        source_id="1",
    )

    _run_job(job=job)

    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_survives_unreadable_tracker(tmp_path):
    """Verifies that a job whose tracker file cannot be deserialized returns instead of raising out of its thread."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker_path = output_directory / TRACKER_FILENAME
    tracker_path.write_text("- not a tracker mapping\n")

    job = PendingJob(
        log_directory=log_directory,
        output_directory=output_directory,
        tracker_path=tracker_path,
        job_id="unresolvable_job_id",
        source_id="1",
    )

    _run_job(job=job)

    assert tracker_path.read_text() == "- not a tracker mapping\n"
