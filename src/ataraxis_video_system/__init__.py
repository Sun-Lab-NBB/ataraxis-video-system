"""A Python library that interfaces with a wide range of cameras to flexibly record visual stream data as videos.

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


from .saver import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import CameraInterfaces, CameraInformation, get_opencv_ids, get_harvesters_ids
from .video_system import VideoSystem, extract_logged_video_system_data

__all__ = [
    "CameraInformation",
    "CameraInterfaces",
    "EncoderSpeedPresets",
    "InputPixelFormats",
    "OutputPixelFormats",
    "VideoEncoders",
    "VideoSystem",
    "check_ffmpeg_availability",
    "check_gpu_availability",
    "extract_logged_video_system_data",
    "get_harvesters_ids",
    "get_opencv_ids",
]
