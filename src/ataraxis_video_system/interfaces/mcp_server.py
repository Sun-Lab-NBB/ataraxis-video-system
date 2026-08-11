"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes camera discovery, CTI file management, runtime requirements checking, video session management, GenICam
configuration, camera manifest management, log archive assembly, video file validation, recording discovery, batch log
processing, and processed timestamp analysis and cleanup through the MCP protocol. AI agents use these tools to
interact with the library's core features.
"""

from __future__ import annotations

from typing import Literal

from .camera_tools import list_cameras_tool  # noqa: F401 - imported for its @mcp.tool() registrations.
from .mcp_instance import mcp
from .session_tools import start_video_session_tool  # noqa: F401 - imported for its @mcp.tool() registrations.
from .discovery_tools import read_camera_manifest_tool  # noqa: F401 - imported for its @mcp.tool() registrations.
from .processing_tools import cancel_log_processing_tool  # noqa: F401 - imported for its @mcp.tool() registrations.
from .configuration_tools import read_genicam_node_tool  # noqa: F401 - imported for its @mcp.tool() registrations.


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication, 'sse' for server-sent-events HTTP communication, and 'streamable-http' for HTTP-based
            communication.
    """
    # Delegates to the MCPServer run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin. For 'streamable-http' it runs an HTTP server that
    # accepts connections until explicitly terminated.
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        mcp.run(transport=transport, json_response=True)
        return

    mcp.run(transport=transport)
