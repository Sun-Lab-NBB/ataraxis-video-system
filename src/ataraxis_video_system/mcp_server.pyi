from typing import Literal

from mcp.server.fastmcp import FastMCP
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
    HarvestersCamera as HarvestersCamera,
    add_cti_file as add_cti_file,
    check_cti_file as check_cti_file,
    discover_camera_ids as discover_camera_ids,
)
from .video_system import VideoSystem as VideoSystem
from .configuration import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration as GenicamConfiguration,
    format_genicam_node as format_genicam_node,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
)

mcp: FastMCP
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
def read_genicam_node(
    camera_index: int = 0, node_name: str = "", blacklisted_nodes: list[str] | None = None
) -> str: ...
def write_genicam_node(camera_index: int, node_name: str, value: str) -> str: ...
def dump_genicam_config(camera_index: int, output_file: str, blacklisted_nodes: list[str] | None = None) -> str: ...
def load_genicam_config(
    camera_index: int, config_file: str, *, strict_identity: bool = False, blacklisted_nodes: list[str] | None = None
) -> str: ...
def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
