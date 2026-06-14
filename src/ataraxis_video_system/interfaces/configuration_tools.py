"""Provides MCP tools for reading, writing, dumping, and loading GenICam node configurations on Harvesters cameras."""

from pathlib import Path
import contextlib
from collections.abc import Generator

from ..video import (
    DEFAULT_BLACKLISTED_NODES,
    HarvestersCamera,
    GenicamConfiguration,
    read_genicam_node as read_node_info,
    format_genicam_node,
    enumerate_genicam_nodes,
)
from .mcp_instance import mcp


@mcp.tool()
def read_genicam_node_tool(
    camera_index: int = 0,
    node_name: str = "",
    blacklisted_nodes: list[str] | None = None,
) -> str:
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, returns detailed information about that specific node. If no node name is provided,
    lists all writable (ReadWrite) value nodes with their current values.

    Args:
        camera_index: The index of the Harvesters camera to read from.
        node_name: The name of a specific GenICam node to read (e.g., "Width", "ExposureTime"). If empty, all nodes
            are listed.
        blacklisted_nodes: A list of GenICam node names to exclude from enumeration. When None, uses the built-in
            blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific nodes that report
            ReadWrite access but reject writes at the hardware level. Pass an empty list to disable blacklisting.

    Returns:
        Detailed node information for a single node, or a newline-separated summary of all nodes.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Single-node mode: returns a detailed formatted description including the node's type, current value,
            # valid range or enumeration entries, and access mode.
            if node_name:
                return format_genicam_node(node_map=camera.node_map, name=node_name)

            # All-nodes mode: enumerates every writable node and reads its current value. Nodes that raise exceptions
            # during read (e.g., due to access restrictions or transient hardware state) are reported as <unreadable>
            # rather than aborting the entire listing.
            node_map = camera.node_map
            names = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=blacklist)
            lines = [f"Found {len(names)} writable GenICam nodes:"]
            for name in names:
                try:
                    info = read_node_info(node_map=node_map, name=name)
                    lines.append(f"  {info.name} = {info.value}")
                except Exception:
                    lines.append(f"  {name} = <unreadable>")
            return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


@mcp.tool()
def write_genicam_node_tool(camera_index: int, node_name: str, value: str) -> str:
    """Sets a GenICam node value on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type based on the node's type.

    Args:
        camera_index: The index of the Harvesters camera to write to.
        node_name: The name of the GenICam node to write (e.g., "Width", "ExposureTime").
        value: The string value to write. Automatically converted to the node's native type.

    Returns:
        A confirmation with the node name and written value, or an error description.
    """
    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Delegates value conversion and writing to the camera's set_node_value method. That method casts the
            # string value to int, float, or bool by node type, keeping enumeration and string nodes as raw strings.
            camera.set_node_value(name=node_name, value=value)
    except Exception as error:
        return f"Error: {error}"
    else:
        return f"Node '{node_name}' set to {value}"


@mcp.tool()
def dump_genicam_config_tool(
    camera_index: int,
    output_file: str,
    blacklisted_nodes: list[str] | None = None,
) -> str:
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    Important:
        The AI agent calling this tool MUST ask the user to provide the output_file path before calling this tool.
        Do not assume or guess the output file path.

    Args:
        camera_index: The index of the Harvesters camera to dump the configuration from.
        output_file: The absolute path to the output YAML file. Must be provided by the user.
        blacklisted_nodes: A list of GenICam node names to exclude from the configuration dump. When None, uses
            the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific
            nodes that report ReadWrite access but reject writes at the hardware level. Pass an empty list to
            disable blacklisting.

    Returns:
        A confirmation with the number of nodes saved, or an error description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Reads every accessible node from the camera's GenICam node map and packages them into a
            # GenicamConfiguration object that stores the camera's model and serial number along with the name and
            # current value of each node.
            config = camera.get_configuration(blacklisted_nodes=blacklist)

            # Serializes the configuration to a YAML file that can later be loaded back onto this or another camera
            # of the same model.
            config.to_yaml(file_path=Path(output_file))
            return f"Configuration saved: {len(config.nodes)} nodes written to {output_file}"
    except Exception as error:
        return f"Error: {error}"


@mcp.tool()
def load_genicam_config_tool(
    camera_index: int,
    config_file: str,
    *,
    strict_identity: bool = False,
    blacklisted_nodes: list[str] | None = None,
) -> str:
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Important:
        The AI agent calling this tool MUST ask the user to provide the config_file path before calling this tool.
        Do not assume or guess the configuration file path.

    Args:
        camera_index: The index of the Harvesters camera to load the configuration onto.
        config_file: The absolute path to the YAML configuration file to load. Must be provided by the user.
        strict_identity: Determines whether to abort on camera identity mismatch instead of warning.
        blacklisted_nodes: A list of GenICam node names to silently skip during validation and write operations.
            When None, uses the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes
            vendor-specific nodes that report ReadWrite access but reject writes at the hardware level. Pass an
            empty list to disable blacklisting.

    Returns:
        The number of nodes applied and any errors encountered, or an error description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    # Validates the config file path before opening a camera connection.
    path = Path(config_file)
    if not path.exists():
        return f"Error: File not found at {config_file}"

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Deserializes the YAML configuration and applies each writable node value to the connected camera. When
            # strict_identity is True, the camera model and serial number must match the values stored in the YAML
            # file; otherwise, a mismatch produces a warning but proceeds with the write.
            config = GenicamConfiguration.from_yaml(file_path=path)
            camera.apply_configuration(config=config, strict_identity=strict_identity, blacklisted_nodes=blacklist)
    except Exception as error:
        return f"Error: {error}"
    else:
        return "Configuration applied successfully"


def _resolve_blacklist(blacklisted_nodes: list[str] | None) -> frozenset[str]:
    """Resolves an optional blacklist parameter to a frozenset suitable for GenICam configuration functions.

    Converts a list of node names to a frozenset when provided, or returns the library default blacklist when None.

    Args:
        blacklisted_nodes: A list of GenICam node names to exclude, or None to use the default blacklist.

    Returns:
        A frozenset of blacklisted node names.
    """
    return frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES


@contextlib.contextmanager
def _harvester_connection(camera_index: int) -> Generator[HarvestersCamera, None, None]:
    """Opens a temporary connection to a Harvesters camera and guarantees disconnection on exit.

    Creates a HarvestersCamera with system_id=0 (placeholder, since only node map access is needed) and connects it.
    The camera is always disconnected in the finally block, releasing the GenTL handle for other processes.

    Args:
        camera_index: The index of the Harvesters camera to connect to.

    Yields:
        The connected HarvestersCamera instance.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        yield camera
    finally:
        camera.disconnect()
