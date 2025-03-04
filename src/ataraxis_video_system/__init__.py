"""A Python library that interfaces with a wide range of cameras to flexibly record visual stream data as images or
videos.

See https://github.com/Sun-Lab-NBB/ataraxis-video-system for more details.
API documentation: https://ataraxis-video-system-api-docs.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner, Natalie Yeung
"""

from .saver import (
    VideoCodecs,
    ImageFormats,
    VideoFormats,
    SaverBackends,
    CPUEncoderPresets,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from .camera import CameraBackends
from .video_system import VideoSystem, extract_logged_video_system_data

__all__ = [
    "VideoSystem",
    "CameraBackends",
    "SaverBackends",
    "VideoCodecs",
    "VideoFormats",
    "ImageFormats",
    "GPUEncoderPresets",
    "CPUEncoderPresets",
    "InputPixelFormats",
    "OutputPixelFormats",
    "extract_logged_video_system_data",
]
