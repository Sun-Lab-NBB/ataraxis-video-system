"""Contains tests for the functions provided by the timestamps.py module."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, PARALLEL_PROCESSING_THRESHOLD, LogArchiveReader

from ataraxis_video_system.video import timestamps as timestamps_module
from ataraxis_video_system.video.timestamps import extract_logged_camera_timestamps

_SOURCE_ID: int = 101
"""Stores the source ID used by every synthetic log archive built by this module."""

_ONSET_US: int = 1_700_000_000_000_000
"""Stores the UTC epoch onset, in microseconds, used by every synthetic log archive built by this module."""


def _build_archive(archive_path, frame_timestamps_us, data_timestamps_us=None):
    """Builds a synthetic log archive holding the requested frame and data messages."""
    create_test_archive(
        archive_path=archive_path,
        source_id=_SOURCE_ID,
        onset_us=_ONSET_US,
        frame_timestamps_us=frame_timestamps_us,
        data_timestamps_us=data_timestamps_us,
    )


def _expected_timestamps(frame_timestamps_us):
    """Converts the elapsed frame timestamps of a synthetic archive into the absolute timestamps extraction returns."""
    return np.array([_ONSET_US + elapsed_us for elapsed_us in frame_timestamps_us], dtype=np.uint64)


def test_extract_logged_camera_timestamps_invalid_path(tmp_path):
    """Verifies that extract_logged_camera_timestamps rejects paths that do not point to an existing .npz file."""
    # A path that does not exist at all.
    missing_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    message = (
        f"Unable to extract camera frame timestamp data from the log file {missing_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=missing_path)

    # An existing file that does not use the .npz suffix.
    text_path = tmp_path / "camera_log.txt"
    text_path.touch()
    message = (
        f"Unable to extract camera frame timestamp data from the log file {text_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=text_path)

    # A directory that carries the .npz suffix.
    directory_path = tmp_path / f"directory{LOG_ARCHIVE_SUFFIX}"
    directory_path.mkdir()
    message = (
        f"Unable to extract camera frame timestamp data from the log file {directory_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=directory_path)


def test_extract_logged_camera_timestamps_filters_data_messages(tmp_path):
    """Verifies that extraction returns the absolute timestamps of payload-free frame messages only."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = [1000, 2000, 3000, 4000, 5000]
    _build_archive(
        archive_path=archive_path,
        frame_timestamps_us=frame_timestamps_us,
        data_timestamps_us=[6000, 7000, 8000],
    )

    extracted = extract_logged_camera_timestamps(log_path=archive_path, workers=1)

    assert extracted.dtype == np.uint64
    np.testing.assert_array_equal(extracted, _expected_timestamps(frame_timestamps_us=frame_timestamps_us))


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_camera_timestamps_single_worker_skips_the_process_pool(tmp_path, monkeypatch):
    """Verifies that a single-worker request processes an above-threshold archive without creating a process pool."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 101))
    _build_archive(archive_path=archive_path, frame_timestamps_us=frame_timestamps_us)

    def _forbidden_pool(*args, **kwargs):
        """Fails the test if the extraction function creates a process pool for a single-worker request."""
        pytest.fail("extract_logged_camera_timestamps created a process pool for a single-worker request.")

    monkeypatch.setattr(timestamps_module, "ProcessPoolExecutor", _forbidden_pool)

    extracted = extract_logged_camera_timestamps(log_path=archive_path, workers=1)

    np.testing.assert_array_equal(extracted, _expected_timestamps(frame_timestamps_us=frame_timestamps_us))


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_camera_timestamps_parallel_matches_sequential(tmp_path):
    """Verifies that parallel extraction of an above-threshold archive reproduces the sequential result exactly."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 101))
    _build_archive(
        archive_path=archive_path,
        frame_timestamps_us=frame_timestamps_us,
        data_timestamps_us=[PARALLEL_PROCESSING_THRESHOLD + 500, PARALLEL_PROCESSING_THRESHOLD + 501],
    )

    sequential = extract_logged_camera_timestamps(log_path=archive_path, workers=1)

    # Runs with the progress bar enabled, which is the default behavior of the function.
    parallel = extract_logged_camera_timestamps(log_path=archive_path, workers=2)

    assert parallel.dtype == np.uint64
    np.testing.assert_array_equal(parallel, sequential)
    np.testing.assert_array_equal(parallel, _expected_timestamps(frame_timestamps_us=frame_timestamps_us))


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_camera_timestamps_parallel_without_frame_messages(tmp_path):
    """Verifies that parallel extraction returns an empty array when an above-threshold archive holds no frames."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        frame_timestamps_us=[],
        data_timestamps_us=list(range(1, PARALLEL_PROCESSING_THRESHOLD + 1)),
    )

    extracted = extract_logged_camera_timestamps(log_path=archive_path, workers=2, display_progress=False)

    assert extracted.dtype == np.uint64
    assert extracted.size == 0


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_camera_timestamps_resolves_the_worker_count(tmp_path, monkeypatch):
    """Verifies that a worker count below one is auto-resolved before the message batches are generated."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 51))
    _build_archive(archive_path=archive_path, frame_timestamps_us=frame_timestamps_us)

    resolution_requests = []

    def _resolve_worker_count(requested_workers):
        """Records the requested worker count and resolves it to a pool size that keeps the test cheap."""
        resolution_requests.append(requested_workers)
        return 2

    monkeypatch.setattr(timestamps_module, "resolve_worker_count", _resolve_worker_count)

    extracted = extract_logged_camera_timestamps(log_path=archive_path, workers=-1, display_progress=False)

    assert resolution_requests == [-1]
    np.testing.assert_array_equal(extracted, _expected_timestamps(frame_timestamps_us=frame_timestamps_us))


def test_process_frame_message_batch_uses_the_supplied_onset(tmp_path):
    """Verifies that the batch worker offsets the payload-free frame messages of its batch by the onset it is given."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = [10, 30]
    _build_archive(archive_path=archive_path, frame_timestamps_us=frame_timestamps_us, data_timestamps_us=[20])

    # A below-threshold archive yields a single batch, which already excludes the onset message key.
    keys = LogArchiveReader(archive_path=archive_path).get_batches(workers=1)[0]

    # Deliberately differs from the archive's own onset. A worker that rediscovers the onset by rescanning the
    # archive, instead of honoring the value the caller pre-discovered, produces the archive's onset here instead.
    override_onset_us = np.uint64(_ONSET_US + 5_000)

    extracted = timestamps_module._process_frame_message_batch(
        log_path=archive_path, keys=keys, onset_us=override_onset_us
    )

    # The 20 us data message carries a payload, so it must not appear among the extracted frame timestamps.
    expected = np.array([override_onset_us + elapsed_us for elapsed_us in frame_timestamps_us], dtype=np.uint64)
    np.testing.assert_array_equal(extracted, expected)

    # The caller concatenates the returned batches, which requires a contiguous uint64 buffer from every worker.
    assert extracted.dtype == np.uint64
    assert extracted.flags["C_CONTIGUOUS"]


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_camera_timestamps_external_executor(tmp_path):
    """Verifies that extraction submits batch work to a caller-owned executor and leaves that executor usable."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    frame_timestamps_us = list(range(1, PARALLEL_PROCESSING_THRESHOLD + 51))
    _build_archive(archive_path=archive_path, frame_timestamps_us=frame_timestamps_us)

    sequential = extract_logged_camera_timestamps(log_path=archive_path, workers=1)

    executor = ProcessPoolExecutor(max_workers=2)
    try:
        parallel = extract_logged_camera_timestamps(
            log_path=archive_path, workers=2, display_progress=False, executor=executor
        )

        np.testing.assert_array_equal(parallel, sequential)

        # The caller owns the pool, so extraction must not shut it down. A pool closed by extraction would instead
        # raise a RuntimeError when asked to accept more work.
        assert executor.submit(abs, -5).result() == 5
    finally:
        executor.shutdown(wait=True)
