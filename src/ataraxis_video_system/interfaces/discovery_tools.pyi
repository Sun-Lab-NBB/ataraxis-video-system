from typing import Any
from pathlib import Path

from ..video import (
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    CAMERA_MANIFEST_FILENAME as CAMERA_MANIFEST_FILENAME,
    CAMERA_TIMESTAMPS_DIRECTORY as CAMERA_TIMESTAMPS_DIRECTORY,
    CameraManifest as CameraManifest,
    write_camera_manifest as write_camera_manifest,
    resolve_recording_roots as resolve_recording_roots,
)
from .mcp_instance import (
    mcp as mcp,
    scan_archive_source_ids as scan_archive_source_ids,
)

_FEATHER_PREFIX: str
_FEATHER_SUFFIX: str

def read_camera_manifest_tool(manifest_path: str) -> dict[str, Any]: ...
def write_camera_manifest_tool(log_directory: str, source_id: int, name: str) -> dict[str, Any]: ...
def discover_camera_data_tool(root_directory: str) -> dict[str, Any]: ...
def validate_video_file_tool(video_file: str) -> dict[str, Any]: ...
def assemble_log_archives_tool(
    log_directory: str, *, remove_sources: bool = True, verify_integrity: bool = False
) -> dict[str, Any]: ...
def _resolve_log_directory_roots(log_directory_paths: list[Path]) -> dict[Path, Path]: ...
def _match_video_file(
    all_video_files: tuple[Path, ...], log_directory: Path, source_id: int, name: str
) -> str | None: ...
def _find_feather_file(timestamps_dirs: tuple[Path, ...], source_id: int) -> Path | None: ...
