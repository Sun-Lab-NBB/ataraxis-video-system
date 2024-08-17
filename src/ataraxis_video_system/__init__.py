"""This library exposes methods that allow interfacing with a wide range of cameras and recording visual stream data
from these cameras as images or videos.

See https://github.com/Sun-Lab-NBB/ataraxis-video-system for more details.
API documentation: https://ataraxis-video-system-api-docs.netlify.app/
Authors: Jacob Groner, Ivan Kondratyev (Inkaros)
"""

from .saver import VideoSaver
from .video_system import VideoSystem

__all__ = ["VideoSystem", "VideoSaver"]
