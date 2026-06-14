from typing import Literal

from . import (
    camera_tools as camera_tools,
    session_tools as session_tools,
    discovery_tools as discovery_tools,
    processing_tools as processing_tools,
    configuration_tools as configuration_tools,
)
from .mcp_instance import mcp as mcp

def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
def run_mcp_server() -> None: ...
