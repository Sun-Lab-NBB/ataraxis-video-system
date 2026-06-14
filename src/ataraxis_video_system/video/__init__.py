"""Provides the core library assets for camera acquisition, video encoding, GenICam configuration, camera manifest
management, and frame-acquisition timestamp log processing.

This package re-exports every asset consumed outside the package (for example, by the CLI and MCP server modules in the
``interfaces`` package), so cross-package callers access these assets through the package namespace rather than reaching
into individual modules.
"""

from .saver import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import (
    CameraInterfaces,
    HarvestersCamera,
    CameraInformation,
    add_cti_file,
    check_cti_file,
    discover_camera_ids,
)
from .manifest import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData, write_camera_manifest
from .video_system import MAXIMUM_QUANTIZATION_VALUE, VideoSystem
from .configuration import (
    DEFAULT_BLACKLISTED_NODES,
    GenicamNodeInfo,
    GenicamConfiguration,
    read_genicam_node,
    format_genicam_node,
    enumerate_genicam_nodes,
)
from .log_processing import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY,
    PARALLEL_PROCESSING_THRESHOLD,
    execute_job,
    prepare_tracker,
    find_log_archive,
    generate_job_ids,
    resolve_recording_roots,
    run_log_processing_pipeline,
)

__all__ = [
    "CAMERA_MANIFEST_FILENAME",
    "CAMERA_TIMESTAMPS_DIRECTORY",
    "DEFAULT_BLACKLISTED_NODES",
    "LOG_ARCHIVE_SUFFIX",
    "MAXIMUM_QUANTIZATION_VALUE",
    "PARALLEL_PROCESSING_THRESHOLD",
    "TIMESTAMP_JOB_NAME",
    "TRACKER_FILENAME",
    "CameraInformation",
    "CameraInterfaces",
    "CameraManifest",
    "CameraSourceData",
    "EncoderSpeedPresets",
    "GenicamConfiguration",
    "GenicamNodeInfo",
    "HarvestersCamera",
    "InputPixelFormats",
    "OutputPixelFormats",
    "VideoEncoders",
    "VideoSystem",
    "add_cti_file",
    "check_cti_file",
    "check_ffmpeg_availability",
    "check_gpu_availability",
    "discover_camera_ids",
    "enumerate_genicam_nodes",
    "execute_job",
    "find_log_archive",
    "format_genicam_node",
    "generate_job_ids",
    "prepare_tracker",
    "read_genicam_node",
    "resolve_recording_roots",
    "run_log_processing_pipeline",
    "write_camera_manifest",
]
