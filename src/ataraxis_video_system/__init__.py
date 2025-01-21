"""This library interfaces with a wide range of cameras to flexibly record visual stream data as images or videos.

See https://github.com/Sun-Lab-NBB/ataraxis-video-system for more details.
API documentation: https://ataraxis-video-system-api-docs.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner, Natalie Yeung
"""

from .saver import VideoSaver
from .video_system import VideoSystem

__all__ = ["VideoSaver", "VideoSystem"]
