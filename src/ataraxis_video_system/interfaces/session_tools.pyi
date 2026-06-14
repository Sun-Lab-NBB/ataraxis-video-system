from typing import Any

from ataraxis_data_structures import DataLogger

from ..video import (
    MAXIMUM_QUANTIZATION_VALUE as MAXIMUM_QUANTIZATION_VALUE,
    VideoSystem as VideoSystem,
    VideoEncoders as VideoEncoders,
    CameraInterfaces as CameraInterfaces,
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
)
from .mcp_instance import (
    mcp as mcp,
    scan_archive_source_ids as scan_archive_source_ids,
)

_active_session: VideoSystem | None
_active_logger: DataLogger | None
_session_info: dict[str, Any] | None

def start_video_session_tool(
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
    video_encoder: str = "H264",
    encoder_speed_preset: int = 3,
    output_pixel_format: str = "yuv420p",
    quantization_parameter: int = 15,
) -> str: ...
def stop_video_session_tool() -> dict[str, Any]: ...
def start_frame_saving_tool() -> str: ...
def stop_frame_saving_tool() -> str: ...
def get_session_status_tool() -> dict[str, Any]: ...
