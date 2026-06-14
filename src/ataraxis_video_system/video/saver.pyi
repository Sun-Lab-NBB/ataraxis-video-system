from enum import IntEnum, StrEnum
from typing import Any, Self
from pathlib import Path
from threading import Thread
from subprocess import Popen

import numpy as np
from numpy.typing import NDArray as NDArray

class VideoEncoders(StrEnum):
    H264 = "H264"
    H265 = "H265"

class EncoderSpeedPresets(IntEnum):
    FASTEST = 1
    FASTER = 2
    FAST = 3
    MEDIUM = 4
    SLOW = 5
    SLOWER = 6
    SLOWEST = 7
    @property
    def gpu_preset(self) -> str: ...
    @property
    def cpu_preset(self) -> str: ...

class InputPixelFormats(StrEnum):
    MONOCHROME = "gray"
    BGR = "bgr24"

class OutputPixelFormats(StrEnum):
    YUV420 = "yuv420p"
    YUV444 = "yuv444p"

def check_gpu_availability() -> bool: ...
def check_ffmpeg_availability() -> bool: ...

class VideoSaver:
    _system_id: int
    _ffmpeg_command: list[str]
    _repr_body: str
    _ffmpeg_process: Popen[bytes] | None
    _stderr_output: bytes
    _stderr_thread: Thread | None
    def __init__(
        self,
        system_id: int,
        output_file: Path,
        frame_width: int,
        frame_height: int,
        frame_rate: float,
        gpu: int = -1,
        video_encoder: VideoEncoders | str = ...,
        encoder_speed_preset: EncoderSpeedPresets | int = ...,
        input_pixel_format: InputPixelFormats | str = ...,
        output_pixel_format: OutputPixelFormats | str = ...,
        quantization_parameter: int = 15,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> None: ...
    @property
    def is_active(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def save_frame(self, frame: NDArray[np.integer[Any]]) -> None: ...
    def _drain_stderr(self) -> None: ...
