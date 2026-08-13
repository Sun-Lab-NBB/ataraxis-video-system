from typing import Literal

from .camera_tools import list_cameras_tool as list_cameras_tool
from .mcp_instance import mcp as mcp
from .session_tools import start_video_session_tool as start_video_session_tool
from .discovery_tools import read_camera_manifest_tool as read_camera_manifest_tool
from .processing_tools import cancel_log_processing_tool as cancel_log_processing_tool
from .configuration_tools import read_genicam_node_tool as read_genicam_node_tool

def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
