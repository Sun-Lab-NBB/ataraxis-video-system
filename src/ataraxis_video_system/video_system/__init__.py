"""This package exposes the VideoSystem class...

Add brief description of exposed classes and the purpose of the package
"""

from .vsc import Camera, VideoSystem
from .interactive_run import interactive_run

__all__ = ["Camera", "VideoSystem", "interactive_run"]
