"""Provides the shared FastMCP server instance and a cross-tool helper function used by the MCP tool modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from ..video import LOG_ARCHIVE_SUFFIX

if TYPE_CHECKING:
    from pathlib import Path

mcp: FastMCP = FastMCP(name="ataraxis-video-system", json_response=True)
"""Stores the MCP server instance used to expose tools to AI agents."""


def scan_archive_source_ids(directory: Path) -> list[str]:
    """Scans a directory for assembled log archives and extracts source IDs from their filenames.

    Matches files ending with the log archive suffix and strips the suffix to recover the source ID string. Results
    are returned in sorted order.

    Args:
        directory: The directory to scan for log archives.

    Returns:
        A sorted list of source ID strings extracted from archive filenames.
    """
    return sorted(
        source_id
        for archive_path in directory.glob(f"*{LOG_ARCHIVE_SUFFIX}")
        if (source_id := archive_path.name.removesuffix(LOG_ARCHIVE_SUFFIX))
    )
