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


from .saver import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import CameraInterfaces, CameraInformation, add_cti_file, check_cti_file, discover_camera_ids
from .manifest import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData
from .video_system import VideoSystem
from .configuration import DEFAULT_BLACKLISTED_NODES, GenicamNodeInfo, GenicamConfiguration
from .log_processing import run_log_processing_pipeline

__all__ = [
    "CAMERA_MANIFEST_FILENAME",
    "DEFAULT_BLACKLISTED_NODES",
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
    "run_log_processing_pipeline",
]
