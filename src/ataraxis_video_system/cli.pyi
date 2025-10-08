from pathlib import Path

from _typeshed import Incomplete

from .saver import (
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
    check_gpu_availability as check_gpu_availability,
    check_ffmpeg_availability as check_ffmpeg_availability,
)
from .camera import (
    CameraInterfaces as CameraInterfaces,
    add_cti_file as add_cti_file,
    get_opencv_ids as get_opencv_ids,
    get_harvesters_ids as get_harvesters_ids,
)
from .video_system import VideoSystem as VideoSystem

CONTEXT_SETTINGS: Incomplete

def axvs_cli() -> None: ...
def set_cti_file(file_path: Path) -> None: ...
def list_camera_indices(interface: str) -> None: ...
def check_requirements() -> None: ...
def live_run(
    interface: str,
    camera_index: int,
    gpu_index: int,
    output_directory: Path,
    width: int,
    height: int,
    frame_rate: int,
    *,
    monochrome: bool,
) -> None: ...
