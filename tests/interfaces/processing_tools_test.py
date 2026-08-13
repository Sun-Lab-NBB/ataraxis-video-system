"""Contains tests for the frame statistics functions provided by the processing_tools.py module."""

import numpy as np
import polars as pl

from ataraxis_video_system.video import ExtractedDataColumns
from ataraxis_video_system.interfaces.processing_tools import (
    _analyze_single_feather,
    _read_feather_timestamps,
    _compute_frame_statistics,
)


def test_compute_frame_statistics_clean_run():
    """Verifies that _compute_frame_statistics reports no drops for evenly spaced timestamps."""
    timestamps = np.arange(0, 100_000, 1000, dtype=np.uint64)

    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    assert statistics["basic_stats"]["total_frames"] == 100
    assert statistics["basic_stats"]["first_timestamp_us"] == 0
    assert statistics["basic_stats"]["last_timestamp_us"] == 99_000
    assert statistics["basic_stats"]["duration_us"] == 99_000
    assert statistics["basic_stats"]["duration_seconds"] == 0.099
    assert statistics["basic_stats"]["estimated_fps"] == 1000.0

    assert statistics["inter_frame_timing"]["mean_us"] == 1000.0
    assert statistics["inter_frame_timing"]["median_us"] == 1000.0
    assert statistics["inter_frame_timing"]["std_us"] == 0.0
    assert statistics["inter_frame_timing"]["min_us"] == 1000
    assert statistics["inter_frame_timing"]["max_us"] == 1000
    assert statistics["inter_frame_timing"]["mean_ms"] == 1.0

    drop_analysis = statistics["frame_drop_analysis"]
    assert drop_analysis["threshold_us"] == 1500.0
    assert drop_analysis["threshold_source"] == "auto_1.5x_median"
    assert drop_analysis["total_gaps_detected"] == 0
    assert drop_analysis["total_estimated_dropped_frames"] == 0
    assert drop_analysis["drop_rate_percent"] == 0.0
    assert drop_analysis["longest_gap_us"] == 0
    assert drop_analysis["longest_gap_ms"] == 0.0
    assert drop_analysis["drop_locations"] == []
    assert not drop_analysis["drop_locations_truncated"]


def test_compute_frame_statistics_detects_drop():
    """Verifies that _compute_frame_statistics detects and locates an interval that passes the drop threshold."""
    timestamps = _build_timestamps(intervals_us=[1000] * 10 + [10_000] + [1000] * 10)

    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    assert statistics["basic_stats"]["total_frames"] == 22
    assert statistics["inter_frame_timing"]["median_us"] == 1000.0
    assert statistics["inter_frame_timing"]["max_us"] == 10_000

    drop_analysis = statistics["frame_drop_analysis"]
    assert drop_analysis["threshold_us"] == 1500.0
    assert drop_analysis["threshold_source"] == "auto_1.5x_median"
    assert drop_analysis["total_gaps_detected"] == 1
    assert drop_analysis["total_estimated_dropped_frames"] == 9
    assert drop_analysis["drop_rate_percent"] == 29.0323
    assert drop_analysis["longest_gap_us"] == 10_000
    assert drop_analysis["longest_gap_ms"] == 10.0
    assert not drop_analysis["drop_locations_truncated"]
    assert drop_analysis["drop_locations"] == [
        {"frame_index": 10, "gap_us": 10_000, "gap_ms": 10.0, "estimated_frames_lost": 9}
    ]


def test_compute_frame_statistics_nets_a_gap_against_the_following_interval():
    """Verifies that a gap the next interval repays reports no lost frame."""
    timestamps = _build_timestamps(intervals_us=[1000] * 5 + [1900, 100] + [1000] * 5)

    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    drop_analysis = statistics["frame_drop_analysis"]
    assert drop_analysis["total_gaps_detected"] == 1
    assert drop_analysis["jitter_compensated_gaps"] == 1
    assert drop_analysis["total_estimated_dropped_frames"] == 0
    assert drop_analysis["drop_locations"] == [
        {"frame_index": 5, "gap_us": 1900, "gap_ms": 1.9, "estimated_frames_lost": 0}
    ]


def test_compute_frame_statistics_charges_a_gap_the_following_interval_does_not_repay():
    """Verifies that a gap followed by an ordinary interval still reports its lost frame."""
    timestamps = _build_timestamps(intervals_us=[1000] * 5 + [2000] + [1000] * 5)

    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    drop_analysis = statistics["frame_drop_analysis"]
    assert drop_analysis["total_gaps_detected"] == 1
    assert drop_analysis["jitter_compensated_gaps"] == 0
    assert drop_analysis["total_estimated_dropped_frames"] == 1


def test_compute_frame_statistics_user_specified_threshold():
    """Verifies that a nonzero drop threshold overrides the automatically detected one."""
    timestamps = _build_timestamps(intervals_us=[1000] * 5 + [4000] + [1000] * 5)

    automatic = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)
    manual = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=5000, max_drop_locations=50)

    assert automatic["frame_drop_analysis"]["threshold_source"] == "auto_1.5x_median"
    assert automatic["frame_drop_analysis"]["total_gaps_detected"] == 1

    # A threshold above the deliberate gap classifies the same recording as drop-free.
    assert manual["frame_drop_analysis"]["threshold_us"] == 5000.0
    assert manual["frame_drop_analysis"]["threshold_source"] == "user_specified"
    assert manual["frame_drop_analysis"]["total_gaps_detected"] == 0
    assert manual["frame_drop_analysis"]["drop_locations"] == []


def test_compute_frame_statistics_caps_drop_locations():
    """Verifies that the drop location list is capped while the reported gap total counts every drop."""
    timestamps = _build_timestamps(intervals_us=([1000] * 4 + [9000]) * 20)

    capped = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=5)
    uncapped = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    assert len(capped["frame_drop_analysis"]["drop_locations"]) == 5
    assert capped["frame_drop_analysis"]["drop_locations_truncated"]
    assert capped["frame_drop_analysis"]["total_gaps_detected"] == 20
    assert capped["frame_drop_analysis"]["total_estimated_dropped_frames"] == 160
    capped_indices = [location["frame_index"] for location in capped["frame_drop_analysis"]["drop_locations"]]
    assert capped_indices == [4, 9, 14, 19, 24]

    assert len(uncapped["frame_drop_analysis"]["drop_locations"]) == 20
    assert not uncapped["frame_drop_analysis"]["drop_locations_truncated"]
    assert uncapped["frame_drop_analysis"]["total_gaps_detected"] == 20
    assert uncapped["frame_drop_analysis"]["total_estimated_dropped_frames"] == 160


def test_compute_frame_statistics_empty_array():
    """Verifies that an empty timestamp array yields a frame count of zero and no timing statistics."""
    statistics = _compute_frame_statistics(
        timestamps=np.array([], dtype=np.uint64), drop_threshold_us=0, max_drop_locations=50
    )

    assert statistics == {
        "basic_stats": {"total_frames": 0},
        "inter_frame_timing": {},
        "frame_drop_analysis": {},
    }


def test_compute_frame_statistics_single_timestamp():
    """Verifies that a single timestamp yields a zero duration and a zero frame rate instead of a division error."""
    statistics = _compute_frame_statistics(
        timestamps=np.array([1_700_000_000_000_000], dtype=np.uint64), drop_threshold_us=0, max_drop_locations=50
    )

    assert statistics == {
        "basic_stats": {
            "total_frames": 1,
            "first_timestamp_us": 1_700_000_000_000_000,
            "last_timestamp_us": 1_700_000_000_000_000,
            "duration_us": 0,
            "duration_seconds": 0.0,
            "estimated_fps": 0.0,
        },
        "inter_frame_timing": {},
        "frame_drop_analysis": {},
    }


def test_compute_frame_statistics_zero_median_interval():
    """Verifies that a zero median interval falls back to unit spacing instead of dividing by zero."""
    timestamps = _build_timestamps(intervals_us=[0, 0, 0, 5000])

    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    assert statistics["inter_frame_timing"]["median_us"] == 0.0
    assert statistics["frame_drop_analysis"]["threshold_us"] == 0.0
    assert statistics["frame_drop_analysis"]["total_gaps_detected"] == 1
    assert statistics["frame_drop_analysis"]["total_estimated_dropped_frames"] == 4999
    assert statistics["frame_drop_analysis"]["drop_locations"] == [
        {"frame_index": 3, "gap_us": 5000, "gap_ms": 5.0, "estimated_frames_lost": 4999}
    ]


def test_compute_frame_statistics_repeated_timestamps():
    """Verifies that a recording spanning no time reports a zero frame rate instead of a division error."""
    statistics = _compute_frame_statistics(
        timestamps=np.full(shape=8, fill_value=7000, dtype=np.uint64), drop_threshold_us=1, max_drop_locations=50
    )

    assert statistics["basic_stats"]["duration_us"] == 0
    assert statistics["basic_stats"]["duration_seconds"] == 0.0
    assert statistics["basic_stats"]["estimated_fps"] == 0.0
    assert statistics["inter_frame_timing"]["mean_us"] == 0.0
    assert statistics["frame_drop_analysis"]["total_gaps_detected"] == 0


def test_read_feather_timestamps_reads_column(tmp_path):
    """Verifies that _read_feather_timestamps returns the frame timestamp column of a valid feather file."""
    feather_path = _write_feather(
        feather_path=tmp_path / "camera_101_timestamps.feather", timestamps=[1000, 2000, 3000, 4000]
    )

    timestamps, error_message = _read_feather_timestamps(feather_file=str(feather_path))

    assert error_message is None
    assert timestamps is not None
    assert timestamps.dtype == np.uint64
    assert timestamps.tolist() == [1000, 2000, 3000, 4000]


def test_read_feather_timestamps_missing_column(tmp_path):
    """Verifies that _read_feather_timestamps reports the schema mismatch of a file without the timestamp column."""
    feather_path = _write_feather(
        feather_path=tmp_path / "camera_101_timestamps.feather", timestamps=[1000, 2000], column="other_column"
    )

    timestamps, error_message = _read_feather_timestamps(feather_file=str(feather_path))

    assert timestamps is None
    assert error_message == f"Missing required '{ExtractedDataColumns.FRAME_TIME}' column. Found: ['other_column']"


def test_analyze_single_feather_composes_reader_and_statistics(tmp_path):
    """Verifies that _analyze_single_feather tags the statistics of the read file with that file's path."""
    timestamps = np.arange(0, 10_000, 1000, dtype=np.uint64)
    feather_path = _write_feather(feather_path=tmp_path / "camera_101_timestamps.feather", timestamps=timestamps)

    result = _analyze_single_feather(feather_file=str(feather_path), drop_threshold_us=0, max_drop_locations=50)
    statistics = _compute_frame_statistics(timestamps=timestamps, drop_threshold_us=0, max_drop_locations=50)

    assert result == {"file": str(feather_path), **statistics}

    missing_path = tmp_path / "absent.feather"
    missing_result = _analyze_single_feather(feather_file=str(missing_path), drop_threshold_us=0, max_drop_locations=50)

    assert missing_result == {"file": str(missing_path), "error": f"File does not exist: {missing_path}"}


def _build_timestamps(intervals_us):
    """Builds a uint64 timestamp array that starts at zero and steps by each of the requested intervals."""
    return np.cumsum([0, *intervals_us], dtype=np.uint64)


def _write_feather(feather_path, timestamps, column=str(ExtractedDataColumns.FRAME_TIME)):
    """Writes the given timestamps to a feather file under the requested column name."""
    pl.DataFrame({column: np.asarray(timestamps, dtype=np.uint64)}).write_ipc(file=feather_path)
    return feather_path
