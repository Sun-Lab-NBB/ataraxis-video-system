"""Provides the shared MCP server instance and a cross-tool helper function used by the MCP tool modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server import MCPServer
from ataraxis_data_structures import discover_log_archives

if TYPE_CHECKING:
    from pathlib import Path

mcp: MCPServer = MCPServer(name="ataraxis-video-system")
"""Stores the MCP server instance used to expose tools to AI agents."""


def scan_archive_source_ids(directory: Path) -> list[str]:
    """Scans a directory for assembled log archives and extracts source IDs from their filenames.

    Inspects the directory's own entries alone, since one DataLogger writes every archive it assembles side by side.
    A directory that does not exist yields no source IDs rather than raising, so callers may scan a path before the
    logger has written to it.

    Args:
        directory: The directory to scan for log archives.

    Returns:
        A sorted list of source ID strings extracted from archive filenames.
    """
    if not directory.is_dir():
        return []

    return sorted(discover_log_archives(log_directory=directory))
