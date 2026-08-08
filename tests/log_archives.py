"""Provides synthetic log archive builders shared by the orchestration and timestamp extraction test modules."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def create_onset_message(source_id: int, onset_us: int) -> NDArray[np.uint8]:
    """Creates an onset message with timestamp=0 and the onset UTC epoch as payload."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([0], dtype=np.uint64).view(np.uint8)
    onset_bytes = np.array([onset_us], dtype=np.uint64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, onset_bytes])


def create_frame_message(source_id: int, elapsed_us: int) -> NDArray[np.uint8]:
    """Creates a frame message with no payload (payload.size == 0)."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([elapsed_us], dtype=np.uint64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes])


def create_data_message(source_id: int, elapsed_us: int, payload_size: int = 4) -> NDArray[np.uint8]:
    """Creates a data message with a non-empty payload."""
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([elapsed_us], dtype=np.uint64).view(np.uint8)
    payload = np.zeros(payload_size, dtype=np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, payload])


def create_test_archive(
    archive_path: Path,
    source_id: int,
    onset_us: int,
    frame_timestamps_us: list[int],
    data_timestamps_us: list[int] | None = None,
) -> None:
    """Creates a .npz log archive with the specified frame and data messages."""
    arrays: dict[str, NDArray[np.uint8]] = {}

    # Creates the onset message.
    onset_key = f"{source_id:03d}_{0:020d}"
    arrays[onset_key] = create_onset_message(source_id=source_id, onset_us=onset_us)

    # Creates frame messages (no payload).
    for elapsed_us in frame_timestamps_us:
        key = f"{source_id:03d}_{elapsed_us:020d}"
        arrays[key] = create_frame_message(source_id=source_id, elapsed_us=elapsed_us)

    # Creates data messages (with payload).
    if data_timestamps_us is not None:
        for elapsed_us in data_timestamps_us:
            key = f"{source_id:03d}_{elapsed_us:020d}"
            arrays[key] = create_data_message(source_id=source_id, elapsed_us=elapsed_us)

    np.savez(archive_path, **arrays)
