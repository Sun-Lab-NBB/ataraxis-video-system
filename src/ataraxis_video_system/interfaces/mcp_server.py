"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes camera discovery, CTI file management, runtime requirements checking, video session management, GenICam
configuration, camera manifest management, log archive assembly, video and log validation, recording discovery, and
batch log processing functionality through the MCP protocol, enabling AI agents to programmatically interact with the
library's core features.
"""

from __future__ import annotations

from typing import Literal

from . import (
    camera_tools,  # noqa: F401
    session_tools,  # noqa: F401
    discovery_tools,  # noqa: F401
    processing_tools,  # noqa: F401
    configuration_tools,  # noqa: F401
)
from .mcp_instance import mcp


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication, 'sse' for server-sent-events HTTP communication, and 'streamable-http' for HTTP-based
            communication.
    """
    # Delegates to the FastMCP run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin; for 'streamable-http' it runs an HTTP server
    # that accepts connections until explicitly terminated.
    mcp.run(transport=transport)


def run_mcp_server() -> None:
    """Starts the MCP server with stdio transport.

    Serves as a CLI entry point, launching the MCP server using the stdio transport protocol recommended for Claude
    Desktop integration.
    """
    run_server(transport="stdio")
