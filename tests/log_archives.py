"""Provides synthetic log archive builders shared by the orchestration and timestamp extraction test modules."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import convert_scalar_to_bytes


def create_test_archive(
    archive_path: Path,
    source_id: int,
    onset_us: int,
    frame_timestamps_us: list[int],
    data_timestamps_us: list[int] | None = None,
) -> None:
    """Creates a .npz log archive with the specified frame and data messages."""
    arrays: dict[str, NDArray[np.uint8]] = {}

    onset_key = f"{source_id:03d}_{0:020d}"
    arrays[onset_key] = _create_onset_message(source_id=source_id, onset_us=onset_us)

    for elapsed_us in frame_timestamps_us:
        key = f"{source_id:03d}_{elapsed_us:020d}"
        arrays[key] = _create_frame_message(source_id=source_id, elapsed_us=elapsed_us)

    if data_timestamps_us is not None:
        for elapsed_us in data_timestamps_us:
            key = f"{source_id:03d}_{elapsed_us:020d}"
            arrays[key] = _create_data_message(source_id=source_id, elapsed_us=elapsed_us)

    np.savez(archive_path, **arrays)


def _create_onset_message(source_id: int, onset_us: int) -> NDArray[np.uint8]:
    """Creates an onset message with timestamp=0 and the onset UTC epoch as payload."""
    source_bytes = convert_scalar_to_bytes(value=source_id, dtype=np.dtype(np.uint8))
    timestamp_bytes = convert_scalar_to_bytes(value=0, dtype=np.dtype(np.uint64))
    onset_bytes = convert_scalar_to_bytes(value=onset_us, dtype=np.dtype(np.uint64))
    return np.concatenate([source_bytes, timestamp_bytes, onset_bytes])


def _create_frame_message(source_id: int, elapsed_us: int) -> NDArray[np.uint8]:
    """Creates a frame message with no payload (payload.size == 0)."""
    source_bytes = convert_scalar_to_bytes(value=source_id, dtype=np.dtype(np.uint8))
    timestamp_bytes = convert_scalar_to_bytes(value=elapsed_us, dtype=np.dtype(np.uint64))
    return np.concatenate([source_bytes, timestamp_bytes])


def _create_data_message(source_id: int, elapsed_us: int, payload_size: int = 4) -> NDArray[np.uint8]:
    """Creates a data message with a non-empty payload."""
    source_bytes = convert_scalar_to_bytes(value=source_id, dtype=np.dtype(np.uint8))
    timestamp_bytes = convert_scalar_to_bytes(value=elapsed_us, dtype=np.dtype(np.uint64))
    payload = np.zeros(payload_size, dtype=np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, payload])
