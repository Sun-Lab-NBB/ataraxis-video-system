from typing import Any
from pathlib import Path

from ..video import (
    CAMERA_MANIFEST_FILENAME as CAMERA_MANIFEST_FILENAME,
    CameraManifest as CameraManifest,
    write_camera_manifest as write_camera_manifest,
)
from .responses import (
    page_fields as page_fields,
    project_item as project_item,
    resolve_page as resolve_page,
    item_breakdown as item_breakdown,
    reject_unknown as reject_unknown,
    resolve_detail_limit as resolve_detail_limit,
)
from .mcp_instance import (
    mcp as mcp,
    scan_archive_source_ids as scan_archive_source_ids,
)
from ..orchestration import (
    OutputLayout as OutputLayout,
    resolve_timestamps_path as resolve_timestamps_path,
)

_SOURCE_AXES: tuple[str, ...]
_SOURCE_SEMI_DETAIL_FIELDS: tuple[str, ...]
_SOURCE_DETAIL_FIELDS: tuple[str, ...]

def read_camera_manifest_tool(manifest_path: str) -> dict[str, Any]: ...
def write_camera_manifest_tool(log_directory: str, source_id: int, name: str) -> dict[str, Any]: ...
def discover_camera_data_tool(
    root_directory: str,
    source_ids: list[str] | None = None,
    name: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def validate_video_file_tool(video_file: str) -> dict[str, Any]: ...
def assemble_log_archives_tool(
    log_directory: str, *, remove_sources: bool = True, verify_integrity: bool = False
) -> dict[str, Any]: ...
def _resolve_log_directory_roots(log_directory_paths: list[Path]) -> dict[Path, Path]: ...
def _match_video_file(
    all_video_files: tuple[Path, ...], log_directory: Path, source_id: int, name: str
) -> str | None: ...
def _find_feather_file(
    timestamps_directories: tuple[Path, ...], log_directory: Path, source_id: int
) -> Path | None: ...
def _count_shared_components(log_directory: Path, candidate: Path) -> int: ...
