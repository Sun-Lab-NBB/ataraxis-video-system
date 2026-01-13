"""A Python library that interfaces with a wide range of cameras to flexibly record visual stream data as video files.

See https://github.com/Sun-Lab-NBB/ataraxis-video-system for more details.
API documentation: https://ataraxis-video-system-api-docs.netlify.app/
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
    os.environ["QT_QPA_PLATFORM"] = "xcb"


from .saver import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import CameraInterfaces, CameraInformation, add_cti_file, check_cti_file, discover_camera_ids
from .video_system import VideoSystem, extract_logged_camera_timestamps

__all__ = [
    "CameraInformation",
    "CameraInterfaces",
    "EncoderSpeedPresets",
    "InputPixelFormats",
    "OutputPixelFormats",
    "VideoEncoders",
    "VideoSystem",
    "add_cti_file",
    "check_cti_file",
    "check_ffmpeg_availability",
    "check_gpu_availability",
    "discover_camera_ids",
    "extract_logged_camera_timestamps",
]
