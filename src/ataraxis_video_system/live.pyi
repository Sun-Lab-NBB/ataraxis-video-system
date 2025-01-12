from typing import Any

from .saver import (
    ImageSaver as ImageSaver,
    VideoSaver as VideoSaver,
    ImageFormats as ImageFormats,
    CPUEncoderPresets as CPUEncoderPresets,
    GPUEncoderPresets as GPUEncoderPresets,
    InputPixelFormats as InputPixelFormats,
)
from .camera import CameraBackends as CameraBackends
from .video_system import VideoSystem as VideoSystem

def _validate_positive_int(_ctx: Any, _param: Any, value: Any) -> int | None:
    """Ensures that the provided integer value is positive."""

def _validate_positive_float(_ctx: Any, _param: Any, value: Any) -> float | None:
    """Ensures that the provided float value is positive."""

def live_run(
    camera_backend: str,
    camera_index: int,
    saver_backend: str,
    output_directory: str,
    display_frames: bool,
    monochrome: bool,
    cti_path: str | None,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Instantiates Camera, Saver, and VideoSystem classes using the input parameters and runs the VideoSystem with
    manual control from the user.

    This method uses input() command to allow interfacing with the running system through a terminal window. The user
    has to use the terminal to quite the runtime and enable / disable saving acquired frames to non-volatile memory.

    Notes:
        This CLI script does not support the full range of features offered through the library. Specifically, it does
        not allow fine-tuning Saver class parameters and supports limited Camera and VideoSystem class behavior
        configuration. Use the API for production applications.
    """
