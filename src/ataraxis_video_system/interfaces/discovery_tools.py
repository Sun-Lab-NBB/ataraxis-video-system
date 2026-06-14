"""Provides MCP tools for camera manifest management, recording discovery, video validation, and archive assembly."""

import json
from typing import Any
from pathlib import Path
import subprocess

from ataraxis_data_structures import assemble_log_archives

from ..video import (
    LOG_ARCHIVE_SUFFIX,
    CAMERA_MANIFEST_FILENAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    CameraManifest,
    write_camera_manifest,
    resolve_recording_roots,
)
from .mcp_instance import mcp, scan_archive_source_ids

_FEATHER_PREFIX: str = "camera_"
"""Filename prefix for camera timestamp feather files."""

_FEATHER_SUFFIX: str = "_timestamps.feather"
"""Filename suffix for camera timestamp feather files."""


@mcp.tool()
def read_camera_manifest_tool(manifest_path: str) -> dict[str, Any]:
    """Reads a camera manifest file and returns its contents.

    Reads the specified camera_manifest.yaml file and returns the list of camera sources registered in it.
    Each source entry contains the numeric source ID and the colloquial name assigned to the camera.

    Args:
        manifest_path: The absolute path to the camera_manifest.yaml file to read.

    Returns:
        A dictionary containing the list of sources with their IDs and names, or an error message.
    """
    path = Path(manifest_path)

    if not path.exists():
        return {"error": f"Manifest file does not exist: {manifest_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {manifest_path}"}

    try:
        manifest = CameraManifest.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Failed to read manifest: {error}"}

    return {
        "manifest_path": manifest_path,
        "sources": [{"id": source.id, "name": source.name} for source in manifest.sources],
        "total_sources": len(manifest.sources),
    }


@mcp.tool()
def write_camera_manifest_tool(
    log_directory: str,
    source_id: int,
    name: str,
) -> dict[str, Any]:
    """Writes or updates a camera manifest file in the specified log directory.

    Registers a camera source in the camera_manifest.yaml file located in the target log directory. If
    the manifest already exists, appends the new source entry. Otherwise, creates a new manifest. Use this
    tool to retroactively tag existing log archives as axvs-produced, or to manually register additional
    camera sources.

    Args:
        log_directory: The absolute path to the DataLogger output directory where the manifest file is stored.
        source_id: The numeric source ID to register in the manifest.
        name: The colloquial human-readable name for the camera source (e.g., 'face_camera').

    Returns:
        A dictionary confirming the write operation with the manifest path and registered source, or an error message.
    """
    dir_path = Path(log_directory)

    if not dir_path.exists():
        return {"error": f"Directory does not exist: {log_directory}"}

    if not dir_path.is_dir():
        return {"error": f"Path is not a directory: {log_directory}"}

    if not name:
        return {"error": "The 'name' parameter must be a non-empty string."}

    try:
        write_camera_manifest(log_directory=dir_path, source_id=source_id, name=name)
    except Exception as error:
        return {"error": f"Failed to write manifest: {error}"}

    manifest_path = dir_path / CAMERA_MANIFEST_FILENAME
    return {
        "manifest_path": str(manifest_path),
        "registered_source": {"id": source_id, "name": name},
        "status": "success",
    }


@mcp.tool()
def discover_camera_data_tool(root_directory: str) -> dict[str, Any]:
    """Discovers confirmed camera recordings under a root directory.

    Recursively searches for camera_manifest.yaml files to identify camera sources. Only sources whose log
    archives (``{source_id}_log.npz``) exist on disk are included. For each confirmed source, resolves the
    paired video file and processed timestamp feather file from pre-collected file indices. Video files are
    matched by camera name first, then by source ID pattern, preferring the closest match by path proximity
    to the log directory. Returns a flat list of resolved source entries.

    Args:
        root_directory: The absolute path to the root directory to search. Searched recursively.

    Returns:
        A dictionary containing a 'sources' list where each entry has 'recording_root', 'source_id', 'name',
        'log_archive', 'video_file', 'timestamps_file', and 'log_directory' keys, a flat 'log_directories'
        list for batch processing, and aggregate counts. Video and timestamp paths are None when the
        corresponding file cannot be found.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # Discovers all camera manifests and collects only sources whose log archives exist on disk.
    confirmed_sources: list[tuple[Path, int, str, Path]] = []
    log_dirs_with_archives: set[Path] = set()

    try:
        for manifest_path in sorted(root_path.rglob(CAMERA_MANIFEST_FILENAME)):
            log_dir = manifest_path.parent

            try:
                manifest = CameraManifest.from_yaml(file_path=manifest_path)
            except Exception:  # noqa: S112
                continue

            if not manifest.sources:
                continue

            for source in manifest.sources:
                archive_path = log_dir / f"{source.id}{LOG_ARCHIVE_SUFFIX}"
                if not archive_path.exists():
                    continue

                confirmed_sources.append((log_dir, source.id, source.name, archive_path))
                log_dirs_with_archives.add(log_dir)
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    if not confirmed_sources:
        return {
            "sources": [],
            "log_directories": [],
            "total_sources": 0,
            "total_log_directories": 0,
        }

    # Pre-collects all video files and camera_timestamps directories under the search root in two rglob passes.
    # Avoids redundant filesystem walks when resolving multiple sources.
    all_video_files = tuple(sorted(root_path.rglob("*.mp4")))
    timestamps_dirs = tuple(
        candidate for candidate in sorted(root_path.rglob(CAMERA_TIMESTAMPS_DIRECTORY)) if candidate.is_dir()
    )

    # Resolves recording roots and builds the log-directory-to-root mapping.
    log_dir_paths = sorted(log_dirs_with_archives)
    log_dir_to_root = _resolve_log_dir_roots(log_dir_paths=log_dir_paths)

    # Builds the flat list of resolved source entries. Each entry pairs the confirmed log archive with its
    # recording root, matched video file, and processed timestamp feather file.
    sources_output: list[dict[str, Any]] = []
    for log_dir, source_id, name, archive_path in confirmed_sources:
        video_path = _match_video_file(
            all_video_files=all_video_files, log_directory=log_dir, source_id=source_id, name=name
        )
        feather_path = _find_feather_file(timestamps_dirs=timestamps_dirs, source_id=source_id)

        sources_output.append(
            {
                "recording_root": str(log_dir_to_root[log_dir]),
                "source_id": str(source_id),
                "name": name,
                "log_archive": str(archive_path),
                "video_file": video_path,
                "timestamps_file": str(feather_path) if feather_path is not None else None,
                "log_directory": str(log_dir),
            }
        )

    return {
        "sources": sources_output,
        "log_directories": sorted(str(log_dir) for log_dir in log_dir_paths),
        "total_sources": len(sources_output),
        "total_log_directories": len(log_dir_paths),
    }


@mcp.tool()
def validate_video_file_tool(video_file: str) -> dict[str, Any]:
    """Validates a video file and extracts metadata using ffprobe.

    Runs ffprobe on the specified video file to extract duration, frame count, codec, resolution, file size,
    and bit rate. Verifies video integrity after a recording session.

    Important:
        The AI agent calling this tool MUST ask the user to provide the video_file path before calling this
        tool. Do not assume or guess the video file path.

    Args:
        video_file: The absolute path to the video file to validate. Must be provided by the user.

    Returns:
        A dictionary containing video metadata (duration, frame count, codec, resolution, file size, bit rate)
        on success, or an error dictionary if the file cannot be read or ffprobe is not available.
    """
    file_path = Path(video_file)

    if not file_path.exists():
        return {"error": f"File not found: {video_file}"}

    if not file_path.is_file():
        return {"error": f"Not a file: {video_file}"}

    # Runs ffprobe to extract stream and format metadata in JSON format. Limits analysis to the first 10 MB and
    # 5 seconds of the file to avoid scanning the entire file for containers that lack header-level metadata.
    # Selects only the first video stream to skip audio, subtitle, and data streams.
    try:
        probe_result = subprocess.run(
            args=[
                "ffprobe",
                "-v",
                "quiet",
                "-probesize",
                "10000000",
                "-analyzeduration",
                "5000000",
                "-select_streams",
                "v:0",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return {"error": "ffprobe is not available on the system PATH. Install FFMPEG to use this tool."}
    except subprocess.CalledProcessError as e:
        return {"error": f"ffprobe failed: {e.stderr.strip() if e.stderr else 'unknown error'}"}

    # Parses the JSON output from ffprobe.
    try:
        probe_data = json.loads(probe_result.stdout)
    except json.JSONDecodeError:
        return {"error": "Unable to parse ffprobe output."}

    # Extracts the first video stream from the probe output.
    video_stream: dict[str, Any] | None = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        return {"error": "No video stream found in file."}

    format_info = probe_data.get("format", {})

    # Extracts frame count. ffprobe may report it as nb_frames or may not include it for some containers.
    frame_count_raw = video_stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw and frame_count_raw != "N/A" else None

    # Extracts duration from the format-level metadata (more reliable than stream-level for MP4 containers).
    duration_raw = format_info.get("duration")
    duration_seconds = round(float(duration_raw), 6) if duration_raw else None

    # Extracts bit rate from the format-level metadata.
    bit_rate_raw = format_info.get("bit_rate")
    bit_rate_bps = int(bit_rate_raw) if bit_rate_raw else None

    # Extracts file size, falling back to filesystem stat if ffprobe does not report it.
    size_raw = format_info.get("size")
    file_size_bytes = int(size_raw) if size_raw else file_path.stat().st_size

    return {
        "file": video_file,
        "valid": True,
        "codec": video_stream.get("codec_name"),
        "codec_long_name": video_stream.get("codec_long_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "bit_rate_bps": bit_rate_bps,
        "file_size_bytes": file_size_bytes,
        "pixel_format": video_stream.get("pix_fmt"),
        "frame_rate": video_stream.get("r_frame_rate"),
    }


@mcp.tool()
def assemble_log_archives_tool(
    log_directory: str,
    *,
    remove_sources: bool = True,
    verify_integrity: bool = False,
) -> dict[str, Any]:
    """Consolidates raw .npy log entries in a DataLogger output directory into .npz archives by source ID.

    Assembles the raw .npy files produced by a DataLogger instance into consolidated .npz archives, one per unique
    source ID. This is required before the log processing pipeline can extract frame timestamps.

    This tool is useful when log archives need to be assembled independently of a video session stop operation,
    for example when processing log directories from previous sessions or when the automatic assembly was skipped or
    failed.

    Important:
        The AI agent calling this tool MUST ask the user to provide the log_directory path before calling this
        tool. Do not assume or guess the log directory path.

    Args:
        log_directory: The absolute path to the DataLogger output directory containing raw .npy log entries. Must
            be provided by the user.
        remove_sources: Determines whether to remove the original .npy files after successful archive assembly.
        verify_integrity: Determines whether to verify archive integrity against original log entries before
            removing sources.

    Returns:
        A dictionary containing the assembly status, directory path, list of created archive filenames, extracted
        source IDs, and archive count. Returns an error dictionary if the directory does not exist or assembly
        fails.
    """
    directory_path = Path(log_directory)

    if not directory_path.exists():
        return {"error": f"Directory not found: {log_directory}"}

    if not directory_path.is_dir():
        return {"error": f"Not a directory: {log_directory}"}

    try:
        assemble_log_archives(
            log_directory=directory_path,
            remove_sources=remove_sources,
            verify_integrity=verify_integrity,
            verbose=False,
        )
    except Exception as e:
        return {"error": f"Archive assembly failed: {e}"}

    # Scans for created archives and extracts source IDs from filenames.
    source_ids = scan_archive_source_ids(directory=directory_path)
    archives = [f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids]

    return {
        "status": "assembled",
        "directory": log_directory,
        "archives": archives,
        "source_ids": source_ids,
        "archive_count": len(archives),
    }


def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]:
    """Resolves each log directory to its recording root.

    Uses unique path component detection to identify recording session boundaries. Falls back to using each
    log directory's parent when unique component detection fails (e.g., single log directory).

    Args:
        log_dir_paths: The sorted list of log directory paths to resolve.

    Returns:
        A mapping from each log directory to its recording root path.
    """
    try:
        recording_roots = resolve_recording_roots(paths=log_dir_paths)
    except RuntimeError:
        recording_roots = tuple(dict.fromkeys(log_dir.parent for log_dir in log_dir_paths))

    log_dir_to_root: dict[Path, Path] = {}
    for log_dir in log_dir_paths:
        for root in recording_roots:
            if log_dir == root or root in log_dir.parents:
                log_dir_to_root[log_dir] = root
                break
        else:
            log_dir_to_root[log_dir] = log_dir.parent

    return log_dir_to_root


def _match_video_file(
    all_video_files: tuple[Path, ...],
    log_directory: Path,
    source_id: int,
    name: str,
) -> str | None:
    """Matches a confirmed source to a pre-collected video file by name or source ID.

    Searches the pre-collected video file list for a match, preferring the closest file by path proximity
    to the log directory. Tries the camera name pattern first (``{name}`` in filename stem), then falls back
    to the source ID pattern (``{source_id:03d}`` in filename stem). When multiple candidates match, selects
    the one sharing the most leading path components with the log directory.

    Args:
        all_video_files: Pre-collected ``.mp4`` file paths from the search root.
        log_directory: The directory containing the camera manifest. Used as the proximity reference.
        source_id: The numeric source ID from the manifest.
        name: The colloquial camera name from the manifest (e.g., ``'body_camera'``). Tried first before
            falling back to the source ID.

    Returns:
        The string path to the matched video file, or None if no match is found.
    """
    log_parts = log_directory.parts

    # Counts leading path components shared between a candidate video and the log directory. Higher values
    # indicate closer proximity in the directory tree.
    def proximity(video_path: Path) -> int:
        shared = 0
        for log_part, video_part in zip(log_parts, video_path.parts, strict=False):
            if log_part != video_part:
                break
            shared += 1
        return shared

    # Tries name-based matching first, since users may rename video files to meaningful names.
    if name:
        name_matches = [video for video in all_video_files if name in video.stem]
        if name_matches:
            return str(max(name_matches, key=proximity))

    # Falls back to source ID pattern using the zero-padded VideoSystem naming convention.
    id_pattern = f"{source_id:03d}"
    id_matches = [video for video in all_video_files if id_pattern in video.stem]
    if id_matches:
        return str(max(id_matches, key=proximity))

    return None


def _find_feather_file(timestamps_dirs: tuple[Path, ...], source_id: int) -> Path | None:
    """Searches pre-discovered ``camera_timestamps/`` directories for a processed feather file matching a source ID.

    Performs a flat (non-recursive) glob inside each ``camera_timestamps/`` directory for a feather file matching the
    ``camera_{source_id}_timestamps.feather`` naming convention. The caller is responsible for pre-discovering
    ``camera_timestamps/`` directories via a single ``rglob`` pass over the search root.

    Args:
        timestamps_dirs: Pre-discovered ``camera_timestamps/`` directory paths collected from the search root.
        source_id: The numeric source ID to search for.

    Returns:
        The path to the feather file, or None if not found.
    """
    pattern = f"{_FEATHER_PREFIX}{source_id}{_FEATHER_SUFFIX}"
    for timestamps_dir in timestamps_dirs:
        matches = list(timestamps_dir.glob(pattern))
        if matches:
            return matches[0]
    return None
