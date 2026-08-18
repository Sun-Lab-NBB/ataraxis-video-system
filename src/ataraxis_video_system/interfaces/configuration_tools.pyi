from ..video import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration as GenicamConfiguration,
    format_genicam_node as format_genicam_node,
    harvester_connection as harvester_connection,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
)
from .mcp_instance import mcp as mcp

def read_genicam_node_tool(
    camera_index: int = 0, node_name: str = "", blacklisted_nodes: list[str] | None = None
) -> str: ...
def write_genicam_node_tool(camera_index: int, node_name: str, value: str) -> str: ...
def dump_genicam_config_tool(
    camera_index: int, output_file: str, blacklisted_nodes: list[str] | None = None
) -> str: ...
def load_genicam_config_tool(
    camera_index: int, config_file: str, *, strict_identity: bool = False, blacklisted_nodes: list[str] | None = None
) -> str: ...
def _describe_genicam_error(error: Exception) -> str: ...
def _resolve_blacklist(blacklisted_nodes: list[str] | None) -> frozenset[str]: ...
