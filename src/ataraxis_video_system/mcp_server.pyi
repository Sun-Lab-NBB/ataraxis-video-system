from typing import Literal

from _typeshed import Incomplete
from ataraxis_data_structures import DataLogger

from .saver import (
    VideoEncoders as VideoEncoders,
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

mcp: Incomplete
_active_session: VideoSystem | None
_active_logger: DataLogger | None

def list_cameras() -> str: ...
def get_cti_status() -> str: ...
def set_cti_file(file_path: str) -> str: ...
def check_runtime_requirements() -> str: ...
def start_video_session(
    output_directory: str,
    interface: str = "opencv",
    camera_index: int = 0,
    width: int = 600,
    height: int = 400,
    frame_rate: int = 30,
    gpu_index: int = -1,
    display_frame_rate: int | None = 25,
    *,
    monochrome: bool = False,
) -> str: ...
def stop_video_session() -> str: ...
def start_frame_saving() -> str: ...
def stop_frame_saving() -> str: ...
def get_session_status() -> str: ...
def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
