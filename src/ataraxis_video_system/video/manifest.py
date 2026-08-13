"""Provides data classes and a helper function for managing camera log manifest files.

Camera log manifests identify DataLogger archives produced by ataraxis-video-system and associate each source ID
with a human-readable name. The manifest file lives alongside the log archives in the DataLogger output directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from filelock import FileLock
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path


CAMERA_MANIFEST_FILENAME: str = "camera_manifest.yaml"
"""The filename used for camera log manifest files within DataLogger output directories."""

_MANIFEST_LOCK_TIMEOUT: float = 10.0
"""The maximum time, in seconds, to wait for the manifest's .lock file before aborting the registration."""


@dataclass(frozen=True, slots=True)
class CameraSourceData:
    """Stores the identification data for a single camera source registered in a log manifest."""

    id: int = 0
    """The source_id used by the VideoSystem instance when logging to the DataLogger."""
    name: str = ""
    """A colloquial human-readable name for the camera source (e.g., 'face_camera')."""


@dataclass
class CameraManifest(YamlConfig):
    """Stores camera source identification data for all VideoSystem instances sharing a DataLogger.

    Each entry in the ``sources`` list corresponds to one VideoSystem instance that logs frame timestamps
    to the same DataLogger output directory.
    """

    sources: list[CameraSourceData] = field(default_factory=list)
    """The list of camera source entries registered in this manifest."""


def write_camera_manifest(log_directory: Path, source_id: int, name: str) -> None:
    """Writes or updates the camera manifest file in the specified log directory.

    If the manifest file already exists (another VideoSystem instance has already registered), reads the existing
    manifest, replaces the entry registered under the same source_id or appends a new entry when the manifest carries
    none, and writes it back. Otherwise, creates a new manifest with a single entry.

    Notes:
        The read, the replacement, and the write are performed under a lock file held beside the manifest, since the
        three steps do not form an atomic sequence on their own. Both the threads that concurrent MCP tool calls run on
        and the separate processes that each VideoSystem instance registers from reach this function, so the lock is a
        file lock rather than a thread lock.

    Args:
        log_directory: The path to the DataLogger output directory where the manifest file is stored.
        source_id: The source_id of the VideoSystem instance to register.
        name: The colloquial human-readable name for the camera source.

    Raises:
        Timeout: If the manifest's .lock file cannot be acquired within the timeout period.
        YAMLError: If an existing camera manifest does not hold a well-formed YAML document.
        MissingValueError: If an existing camera manifest omits a field the CameraManifest class requires.
    """
    manifest_path = log_directory / CAMERA_MANIFEST_FILENAME
    lock = FileLock(lock_file=str(manifest_path.with_suffix(manifest_path.suffix + ".lock")))

    with lock.acquire(timeout=_MANIFEST_LOCK_TIMEOUT):
        # Reads the existing manifest if one has already been written by another VideoSystem instance sharing
        # this DataLogger.
        manifest = CameraManifest.from_yaml(file_path=manifest_path) if manifest_path.exists() else CameraManifest()

        # Replaces the entry a re-registering source already owns, since two entries sharing one source_id leave every
        # downstream reader keyed by that id silently dropping one of them.
        replaced_index = next(
            (index for index, source in enumerate(manifest.sources) if source.id == source_id),
            None,
        )
        if replaced_index is None:
            manifest.sources.append(CameraSourceData(id=source_id, name=name))
        else:
            manifest.sources[replaced_index] = CameraSourceData(id=source_id, name=name)

        manifest.to_yaml(file_path=manifest_path)
