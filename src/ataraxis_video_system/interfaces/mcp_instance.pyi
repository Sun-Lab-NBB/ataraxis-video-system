from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..video import LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX

mcp: FastMCP

def scan_archive_source_ids(directory: Path) -> list[str]: ...
