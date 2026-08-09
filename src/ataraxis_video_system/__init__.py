"""Interfaces with a wide range of cameras to flexibly record visual stream data as video files.

See the `documentation <https://ataraxis-video-system-api-docs.netlify.app/>`_ for the description of
available assets. See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-video-system>`_
for more details.

Authors: Ivan Kondratyev (Inkaros), Jacob Groner, Natalie Yeung
"""

import os
import multiprocessing as mp

# Applies important library-wide configurations to optimize runtime performance.
if mp.get_start_method(allow_none=True) is None:
    # Makes the library behave the same way across all platforms.
    mp.set_start_method("spawn")

# Improves frame rendering (display) on Windows operating systems.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# The QT bundled with OpenCV (used for live image rendering) does not include the wayland support plugin. This forces
# QT to use the X11 compatibility layer when it is called from a Wayland system.
if "WAYLAND_DISPLAY" in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # pragma: no cover

# Silences the benign Qt teardown warnings (e.g., "QObject::killTimer: Timers cannot be stopped from another thread")
# that OpenCV's bundled Qt writes to stderr when the live frame-display window is destroyed from the producer process's
# display thread. setdefault() preserves any value the operator has already exported, and setting the rule here ensures
# every spawned subprocess inherits it.
os.environ.setdefault("QT_LOGGING_RULES", "default.warning=false")


from .video import (
    CAMERA_MANIFEST_FILENAME,
    DEFAULT_BLACKLISTED_NODES,
    VideoSystem,
    VideoEncoders,
    CameraManifest,
    GenicamNodeInfo,
    CameraInterfaces,
    CameraSourceData,
    CameraInformation,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    GenicamConfiguration,
    add_cti_file,
    check_cti_file,
    discover_camera_ids,
    harvester_connection,
    write_camera_manifest,
    check_gpu_availability,
    check_ffmpeg_availability,
    resolve_camera_video_path,
    extract_logged_camera_timestamps,
)
from .orchestration import (
    TRACKER_FILENAME,
    FRAME_TIME_COLUMN,
    TIMESTAMP_JOB_NAME,
    TIMESTAMP_JOB_CORES,
    CAMERA_TIMESTAMPS_PREFIX,
    CAMERA_TIMESTAMPS_SUFFIX,
    CAMERA_TIMESTAMPS_DIRECTORY,
    ArchiveFootprint,
    execute_job,
    generate_job_ids,
    resolve_job_workers,
    discover_camera_jobs,
    estimate_job_memory_mb,
    resolve_archive_footprint,
    run_log_processing_pipeline,
    resolve_camera_timestamps_path,
)

__all__ = [
    "CAMERA_MANIFEST_FILENAME",
    "CAMERA_TIMESTAMPS_DIRECTORY",
    "CAMERA_TIMESTAMPS_PREFIX",
    "CAMERA_TIMESTAMPS_SUFFIX",
    "DEFAULT_BLACKLISTED_NODES",
    "FRAME_TIME_COLUMN",
    "TIMESTAMP_JOB_CORES",
    "TIMESTAMP_JOB_NAME",
    "TRACKER_FILENAME",
    "ArchiveFootprint",
    "CameraInformation",
    "CameraInterfaces",
    "CameraManifest",
    "CameraSourceData",
    "EncoderSpeedPresets",
    "GenicamConfiguration",
    "GenicamNodeInfo",
    "InputPixelFormats",
    "OutputPixelFormats",
    "VideoEncoders",
    "VideoSystem",
    "add_cti_file",
    "check_cti_file",
    "check_ffmpeg_availability",
    "check_gpu_availability",
    "discover_camera_ids",
    "discover_camera_jobs",
    "estimate_job_memory_mb",
    "execute_job",
    "extract_logged_camera_timestamps",
    "generate_job_ids",
    "harvester_connection",
    "resolve_archive_footprint",
    "resolve_camera_timestamps_path",
    "resolve_camera_video_path",
    "resolve_job_workers",
    "run_log_processing_pipeline",
    "write_camera_manifest",
]
