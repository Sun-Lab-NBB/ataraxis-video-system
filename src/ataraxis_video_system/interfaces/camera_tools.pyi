from ..video import (
    CameraInterfaces as CameraInterfaces,
    add_cti_file as add_cti_file,
    check_cti_file as check_cti_file,
    discover_camera_ids as discover_camera_ids,
    check_gpu_availability as check_gpu_availability,
    check_ffmpeg_availability as check_ffmpeg_availability,
)
from .mcp_instance import mcp as mcp

def list_cameras_tool() -> str: ...
def get_cti_status_tool() -> str: ...
def set_cti_file_tool(file_path: str) -> str: ...
def check_runtime_requirements_tool() -> str: ...
