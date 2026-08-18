"""Provides MCP tools for reading, writing, dumping, and loading GenICam node configurations on Harvesters cameras."""

from pathlib import Path

from ..video import (
    DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration,
    read_genicam_node as read_node_info,
    format_genicam_node,
    harvester_connection,
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
        Detailed node information for a single node, a newline-separated summary of all nodes, or an error
        description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with harvester_connection(camera_index=camera_index) as camera:
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
        target = f"node '{node_name}'" if node_name else "the writable node list"
        return f"Error: unable to read {target} from camera {camera_index}. {_describe_genicam_error(error=error)}"


@mcp.tool()
def write_genicam_node_tool(camera_index: int, node_name: str, value: str) -> str:
    """Sets a GenICam node value on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type based on the node's type.

    Notes:
        The write is applied over a connection that closes when this tool returns, and the reported value is read back
        over that same connection. A node that reports ReadWrite access can still coerce the write to its increment or
        reject it outright, and a camera that does not retain node state across a device close serves the previous
        value to the next connection. Compare the reported value against the requested one, and treat a configuration
        as applied to a recording only once the acquisition runtime writes it.

    Args:
        camera_index: The index of the Harvesters camera to write to.
        node_name: The name of the GenICam node to write (e.g., "Width", "ExposureTime").
        value: The string value to write. Automatically converted to the node's native type.

    Returns:
        A confirmation naming the requested value and the value the node reports after the write, a notice that the
        write landed while the read-back failed, or an error description. Only the last carries the 'Error:' prefix,
        since a failed read-back leaves the write itself in place and a retry would apply it a second time.
    """
    try:
        with harvester_connection(camera_index=camera_index) as camera:
            # The camera's set_node_value method casts the string value to int, float, or bool by node type, keeping
            # enumeration and string nodes as raw strings.
            camera.set_node_value(name=node_name, value=value)

            # Reads the node back over the same connection, since a node that advertises ReadWrite access can still
            # coerce or drop the write, and reporting the requested value alone hides that from the caller. A failure
            # here follows a write that already landed, so it reports separately from a failure to write at all.
            try:
                observed = read_node_info(node_map=camera.node_map, name=node_name).value
            except Exception as error:
                return (
                    f"Node '{node_name}' written with {value}, but reading it back failed, so the value the camera "
                    f"holds is unconfirmed. {_describe_genicam_error(error=error)}"
                )
    except Exception as error:
        return (
            f"Error: unable to write node '{node_name}' on camera {camera_index}. "
            f"{_describe_genicam_error(error=error)}"
        )
    else:
        return f"Node '{node_name}' written with {value}. The camera reports {observed} for it on this connection."


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
        A confirmation with the number of node entries saved, counting one entry per selector combination for
        selector-addressed nodes, or an error description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with harvester_connection(camera_index=camera_index) as camera:
            # Reads every writable (ReadWrite) value node from the camera's GenICam node map and packages them into a
            # GenicamConfiguration object that stores the camera's model and serial number along with the name,
            # current value, and addressing selectors of each node entry.
            config = camera.get_configuration(blacklisted_nodes=blacklist)

            config.to_yaml(file_path=Path(output_file))
            return f"Configuration saved: {len(config.nodes)} nodes written to {output_file}"
    except Exception as error:
        return (
            f"Error: unable to dump the configuration of camera {camera_index} to {output_file}. "
            f"{_describe_genicam_error(error=error)}"
        )


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

    Notes:
        The configuration is applied over a connection that closes when this tool returns. A camera that does not
        retain node state across a device close serves its previous values to the next connection. State that must
        survive this tool belongs in a camera UserSet, and a recording relies on the acquisition runtime to apply the
        configuration it requires. Read the nodes back with read_genicam_node_tool to confirm what the camera holds.

    Args:
        camera_index: The index of the Harvesters camera to load the configuration onto.
        config_file: The absolute path to the YAML configuration file to load. Must be provided by the user.
        strict_identity: Determines whether to abort on camera identity mismatch instead of warning.
        blacklisted_nodes: A list of GenICam node names to silently skip during validation and write operations.
            When None, uses the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes
            vendor-specific nodes that report ReadWrite access but reject writes at the hardware level. Pass an
            empty list to disable blacklisting.

    Returns:
        A confirmation naming the camera and the number of node entries read from the configuration file, or an error
        description. The blacklist withholds some of those entries from the camera, so the count reports the file's
        contents rather than the writes performed.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    # Validates the config file path before opening a camera connection.
    path = Path(config_file)
    if not path.exists():
        return f"Error: File not found at {config_file}"

    try:
        with harvester_connection(camera_index=camera_index) as camera:
            # Deserializes the YAML configuration and applies each writable node value to the connected camera. When
            # strict_identity is True, the camera model and serial number must match the values stored in the YAML
            # file. Otherwise, a mismatch produces a warning and the write proceeds.
            config = GenicamConfiguration.from_yaml(file_path=path)
            camera.apply_configuration(config=config, strict_identity=strict_identity, blacklisted_nodes=blacklist)
    except Exception as error:
        return (
            f"Error: unable to apply the configuration stored in {config_file} to camera {camera_index}. "
            f"{_describe_genicam_error(error=error)}"
        )
    else:
        return f"Configuration applied to camera {camera_index}: {len(config.nodes)} node entries read from the file."


def _describe_genicam_error(error: Exception) -> str:
    """Renders an exception raised by the GenICam runtime as caller-facing text.

    Several GenICam exceptions, including the one an unknown node name raises, carry no message, so interpolating them
    directly yields text that names no cause. Naming the exception type keeps such a failure distinguishable from one
    the runtime does describe.

    Args:
        error: The exception raised while the camera connection was open.

    Returns:
        The exception's own message, or a description naming its type when it carries no message.
    """
    description = str(error).strip()
    if description:
        return description
    return f"The camera runtime raised {type(error).__name__} without a message."


def _resolve_blacklist(blacklisted_nodes: list[str] | None) -> frozenset[str]:
    """Resolves an optional blacklist parameter, substituting the default blacklist for None.

    Args:
        blacklisted_nodes: The GenICam node names to exclude, or None to take the default blacklist.

    Returns:
        The node names excluded from enumeration.
    """
    return frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES
