from typing import Literal
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
    check_cti_file as check_cti_file,
    discover_camera_ids as discover_camera_ids,
)
from .video_system import VideoSystem as VideoSystem

CONTEXT_SETTINGS: Incomplete

def axvs_cli() -> None: ...
def set_cti_file(file_path: Path) -> None: ...
def check_cti_status() -> None: ...
def list_camera_indices() -> None: ...
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
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None: ...
