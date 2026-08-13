from pathlib import Path

from mcp.server import MCPServer

mcp: MCPServer

def scan_archive_source_ids(directory: Path) -> list[str]: ...
