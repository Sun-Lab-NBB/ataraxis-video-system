"""Provides data classes and a helper function for managing camera log manifest files.

Camera log manifests identify DataLogger archives produced by ataraxis-video-system and associate each source ID
with a human-readable name. The manifest file lives alongside the log archives in the DataLogger output directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path


CAMERA_MANIFEST_FILENAME: str = "camera_manifest.yaml"
"""The filename used for camera log manifest files within DataLogger output directories."""


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
    to the same DataLogger output directory. The manifest file enables downstream tools to identify which
    log archives were produced by ataraxis-video-system and to associate source IDs with human-readable names.
    """

    sources: list[CameraSourceData] = field(default_factory=list)
    """The list of camera source entries registered in this manifest."""


def write_camera_manifest(log_directory: Path, source_id: int, name: str) -> None:
    """Writes or updates the camera manifest file in the specified log directory.

    If the manifest file already exists (another VideoSystem instance has already registered), reads the
    existing manifest, appends the new source entry, and writes it back. Otherwise, creates a new manifest
    with a single entry.

    Args:
        log_directory: The path to the DataLogger output directory where the manifest file is stored.
        source_id: The source_id of the VideoSystem instance to register.
        name: The colloquial human-readable name for the camera source.
    """
    manifest_path = log_directory / CAMERA_MANIFEST_FILENAME

    # Reads the existing manifest if one has already been written by another VideoSystem instance sharing
    # this DataLogger.
    manifest = CameraManifest.from_yaml(file_path=manifest_path) if manifest_path.exists() else CameraManifest()

    # Appends the new source entry and writes the updated manifest back to disk.
    manifest.sources.append(CameraSourceData(id=source_id, name=name))
    manifest.to_yaml(file_path=manifest_path)
