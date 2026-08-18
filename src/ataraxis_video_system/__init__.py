"""Interfaces with a wide range of cameras to flexibly record visual stream data as video files.

See the `documentation <https://ataraxis-video-system-api-docs.netlify.app/>`_ for the description of
available assets. See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-video-system>`_
for more details.

Authors: Ivan Kondratyev (Inkaros), Jacob Groner, Natalie Yeung
"""

import os
import multiprocessing as mp

# Applies library-wide configurations that keep multiprocessing and frame display behaving consistently across
# platforms. This block must run before the imports below, as the start method has to be set before any process
# spawns, and the OpenCV and Qt variables are read while the 'video' subpackage imports execute.
if mp.get_start_method(allow_none=True) is None:
    # Makes the library behave the same way across all platforms.
    mp.set_start_method("spawn")

# Improves frame rendering (display) on Windows operating systems.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# The QT bundled with OpenCV (used for live image rendering) does not include the wayland support plugin. This forces
# QT to use the X11 compatibility layer when it is called from a Wayland system.
if "WAYLAND_DISPLAY" in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# Silences the benign Qt teardown warnings (e.g., "QObject::killTimer: Timers cannot be stopped from another thread")
# that OpenCV's bundled Qt writes to stderr when the live frame-display window is destroyed from the producer process's
# display thread. setdefault() preserves any value the operator has already exported, and setting the rule here ensures
# every spawned subprocess inherits it.
os.environ.setdefault("QT_LOGGING_RULES", "default.warning=false")


from .video import (
    CAMERA_MANIFEST_FILENAME,
    DEFAULT_BLACKLISTED_NODES,
    GENICAM_UNAVAILABLE_REASON,
    MAXIMUM_QUANTIZATION_VALUE,
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
    ExtractedDataColumns,
    GenicamConfiguration,
    add_cti_file,
    check_cti_file,
    discover_camera_ids,
    write_camera_manifest,
    check_gpu_availability,
    check_ffmpeg_availability,
    genicam_runtime_available,
    read_camera_configuration,
    resolve_camera_video_path,
    extract_logged_camera_timestamps,
)
from .orchestration import (
    CAMERA_EXTRACTION_JOB_NAME,
    CAMERA_EXTRACTION_JOB_CORES,
    JobSizing,
    JobSource,
    JobUniverse,
    OutputLayout,
    execute_job,
    resolve_jobs,
    generate_job_ids,
    size_archive_job,
    resolve_timestamps_path,
    run_log_processing_pipeline,
)

__all__ = [
    "CAMERA_EXTRACTION_JOB_CORES",
    "CAMERA_EXTRACTION_JOB_NAME",
    "CAMERA_MANIFEST_FILENAME",
    "DEFAULT_BLACKLISTED_NODES",
    "GENICAM_UNAVAILABLE_REASON",
    "MAXIMUM_QUANTIZATION_VALUE",
    "CameraInformation",
    "CameraInterfaces",
    "CameraManifest",
    "CameraSourceData",
    "EncoderSpeedPresets",
    "ExtractedDataColumns",
    "GenicamConfiguration",
    "GenicamNodeInfo",
    "InputPixelFormats",
    "JobSizing",
    "JobSource",
    "JobUniverse",
    "OutputLayout",
    "OutputPixelFormats",
    "VideoEncoders",
    "VideoSystem",
    "add_cti_file",
    "check_cti_file",
    "check_ffmpeg_availability",
    "check_gpu_availability",
    "discover_camera_ids",
    "execute_job",
    "extract_logged_camera_timestamps",
    "generate_job_ids",
    "genicam_runtime_available",
    "read_camera_configuration",
    "resolve_camera_video_path",
    "resolve_jobs",
    "resolve_timestamps_path",
    "run_log_processing_pipeline",
    "size_archive_job",
    "write_camera_manifest",
]
