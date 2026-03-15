"""Provides data classes and helper functions for reading, writing, and managing GenICam camera configurations.

These utilities allow enumerating, inspecting, and modifying individual GenICam feature nodes, as well as
dumping and loading full camera configurations to and from YAML files.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any
from contextlib import suppress
from dataclasses import field, dataclass

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from genicam.genapi import NodeMap  # type: ignore[import-untyped]


class _NodeType(IntEnum):
    """Defines GenICam ``principal_interface_type`` codes."""

    INTEGER = 2
    """Integer-valued node."""
    BOOLEAN = 3
    """Boolean-valued node."""
    COMMAND = 4
    """Command (action trigger) node."""
    FLOAT = 5
    """Float-valued node."""
    STRING = 6
    """String-valued node."""
    REGISTER = 7
    """Raw register node."""
    CATEGORY = 8
    """Category (container) node."""
    ENUMERATION = 9
    """Enumeration-valued node."""
    ENUM_ENTRY = 10
    """Single entry within an Enumeration node."""
    PORT = 11
    """Port node."""


class _AccessMode(IntEnum):
    """Defines GenICam ``get_access_mode()`` codes."""

    NOT_IMPLEMENTED = 0
    """Node is not implemented by the device."""
    NOT_AVAILABLE = 1
    """Node is currently not available."""
    WRITE_ONLY = 2
    """Node can be written but not read."""
    READ_ONLY = 3
    """Node can be read but not written."""
    READ_WRITE = 4
    """Node can be both read and written."""


_VALUE_NODE_TYPES: frozenset[int] = frozenset(
    {_NodeType.INTEGER, _NodeType.BOOLEAN, _NodeType.FLOAT, _NodeType.STRING, _NodeType.ENUMERATION}
)
"""The GenICam node type codes that represent collectible leaf value nodes."""


@dataclass(frozen=True, slots=True)
class GenicamNodeInfo:
    """Stores the name, value, and unit of a single GenICam feature node."""

    name: str
    """The feature name of the node (e.g., "Width", "ExposureTime")."""
    value: int | float | str | bool
    """The current value of the node."""
    unit: str | None = None
    """The measurement unit for the node value (e.g., "us", "dB"), or None if no unit is defined."""


@dataclass
class GenicamConfiguration(YamlConfig):
    """Stores a complete GenICam camera configuration with camera identity metadata."""

    camera_model: str = ""
    """The model name of the camera that produced this configuration."""
    camera_serial_number: str = ""
    """The serial number of the camera that produced this configuration."""
    nodes: list[GenicamNodeInfo] = field(default_factory=list)
    """The list of ReadWrite GenICam nodes with their current values."""


def enumerate_genicam_nodes(node_map: NodeMap) -> list[str]:
    """Collects the names of all writable leaf value nodes by walking the GenICam category tree from the root.

    Notes:
        Uses an iterative stack-based traversal starting from ``node_map.Root``. Collects ReadWrite nodes of type
        Integer, Float, Enumeration, Boolean, and String, skipping all other nodes. All node accesses are wrapped
        in try/except to gracefully handle locked or unavailable nodes.

    Args:
        node_map: The GenICam node map object.

    Returns:
        A sorted list of unique feature node names for all discovered writable leaf value nodes.
    """
    names: list[str] = []
    visited: set[str] = set()

    # Seeds the stack with the root category node. The GenICam node map is a tree where Category nodes act as
    # containers and leaf nodes hold the actual feature values.
    stack: list[Any] = [node_map.Root]

    while stack:
        node = stack.pop()

        # Extracts the node name. Some nodes may be locked or unavailable, so access is guarded.
        try:
            name: str = node.node.name
        except Exception:  # noqa: S112
            continue

        # Skips already-visited nodes to avoid cycles in the category tree.
        if name in visited:
            continue
        visited.add(name)

        # Resolves the node's principal interface type to determine how to handle it.
        try:
            type_code = int(node.node.principal_interface_type)
        except Exception:  # noqa: S112
            continue

        # Descends into Category nodes by pushing their children onto the stack.
        if type_code == _NodeType.CATEGORY:
            with suppress(Exception):
                stack.extend(node.features)
            continue

        # Collects leaf value nodes (Integer, Float, Boolean, String, Enumeration) that are ReadWrite.
        if type_code in _VALUE_NODE_TYPES:
            with suppress(Exception):
                if int(node.node.get_access_mode()) == _AccessMode.READ_WRITE:
                    names.append(name)

    names.sort()
    return names


def read_genicam_node(node_map: NodeMap, name: str) -> GenicamNodeInfo:
    """Reads a single readable value node from the GenICam node map and returns its name, value, and unit.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to read (e.g., "Width", "ExposureTime").

    Returns:
        A ``GenicamNodeInfo`` instance containing the node's name, current value, and unit.

    Raises:
        AttributeError: If the named node does not exist on the node map.
        ValueError: If the node is not a readable value node.
    """
    # Accesses the named feature on the node map. Raises AttributeError if the node does not exist.
    feature = getattr(node_map, name)
    raw_node = feature.node

    # Rejects nodes that are not readable value nodes.
    type_code = int(raw_node.principal_interface_type)
    if type_code not in _VALUE_NODE_TYPES:
        message = (
            f"Unable to read GenICam node '{name}'. The node must be a value type (Integer, Float, Boolean, "
            f"String, or Enumeration), but got type code {type_code}."
        )
        console.error(message=message, error=ValueError)

    access_code = int(raw_node.get_access_mode())
    if access_code not in (_AccessMode.READ_WRITE, _AccessMode.READ_ONLY):
        message = (
            f"Unable to read GenICam node '{name}'. The node must have ReadWrite or ReadOnly access, "
            f"but got access code {access_code}."
        )
        console.error(message=message, error=ValueError)

    # Reads the current value. Guaranteed readable by the access mode guard.
    value: int | float | str | bool = feature.value

    # Attempts to read the unit string. Not all nodes define a unit, so failure is suppressed.
    unit: str | None = None
    with suppress(Exception):
        raw_unit = str(feature.node.unit)
        if raw_unit:
            unit = raw_unit

    return GenicamNodeInfo(name=name, value=value, unit=unit)


def format_genicam_node(node_map: NodeMap, name: str) -> str:
    """Reads a single readable GenICam feature node and returns a formatted string with its full metadata.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to read (e.g., "Width", "ExposureTime").

    Returns:
        A multi-line formatted string containing the node's name, type, value, access mode, description, numeric
        range, enumeration entries, and unit.

    Raises:
        AttributeError: If the named node does not exist on the node map.
        ValueError: If the node is not readable (must be ReadWrite or ReadOnly).
    """
    # Accesses the named feature and its underlying GenICam node descriptor.
    feature = getattr(node_map, name)
    raw_node = feature.node

    # Reads the integer type and access mode codes from the node descriptor.
    type_code = int(raw_node.principal_interface_type)
    access_code = int(raw_node.get_access_mode())

    # Rejects nodes that are not readable value nodes.
    if type_code not in _VALUE_NODE_TYPES:
        message = (
            f"Unable to format GenICam node '{name}'. The node must be a value type (Integer, Float, Boolean, "
            f"String, or Enumeration), but got type code {type_code}."
        )
        console.error(message=message, error=ValueError)

    if access_code not in (_AccessMode.READ_WRITE, _AccessMode.READ_ONLY):
        message = (
            f"Unable to format GenICam node '{name}'. The node must have ReadWrite or ReadOnly access, "
            f"but got access code {access_code}."
        )
        console.error(message=message, error=ValueError)

    # Resolves human-readable names. Both are guaranteed valid by the guards above.
    node_type = _NodeType(type_code).name
    access_mode = _AccessMode(access_code).name

    # Reads the current value. Guaranteed readable by the access mode guard.
    value_str = str(feature.value)

    # Reads the node description from the camera's GenICam XML descriptor.
    description = ""
    with suppress(Exception):
        description = str(raw_node.description)

    # Builds the base output with fields common to all node types.
    lines = [
        f"Node: {name}",
        f"  Type: {node_type}",
        f"  Value: {value_str}",
        f"  Access: {access_mode}",
        f"  Description: {description}",
    ]

    # Appends numeric range information for Integer and Float nodes.
    if type_code in (_NodeType.INTEGER, _NodeType.FLOAT):
        with suppress(Exception):
            lines.append(f"  Min: {feature.min}")
        with suppress(Exception):
            lines.append(f"  Max: {feature.max}")

    # Appends the step increment for Integer nodes.
    if type_code == _NodeType.INTEGER:
        with suppress(Exception):
            lines.append(f"  Increment: {feature.inc}")

    # Appends the list of valid entry names for Enumeration nodes.
    if type_code == _NodeType.ENUMERATION:
        with suppress(Exception):
            entry_names = [str(entry.node.name) for entry in feature.entries]
            lines.append(f"  Entries: {', '.join(entry_names)}")

    # Appends the measurement unit if the node defines one.
    with suppress(Exception):
        raw_unit = str(raw_node.unit)
        if raw_unit:
            lines.append(f"  Unit: {raw_unit}")

    return "\n".join(lines)


def write_genicam_node(node_map: NodeMap, name: str, value: str) -> None:
    """Sets the value of a single writable (ReadWrite) GenICam feature node.

    Accepts a string value and coerces it to the appropriate Python type (int, float, bool, or str) based on
    the node's ``principal_interface_type`` before writing.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of a writable node (e.g., "Width", "ExposureTime"). The target node must have
            ReadWrite access mode.
        value: The string representation of the value to write. Coerced to the node's native type automatically.

    Raises:
        ValueError: If the named node does not have ReadWrite access or the value cannot be coerced.
        RuntimeError: If the write operation fails.
    """
    feature = getattr(node_map, name)

    # Rejects nodes that are not writable.
    access_code = int(feature.node.get_access_mode())
    if access_code != _AccessMode.READ_WRITE:
        message = (
            f"Unable to write to GenICam node '{name}'. The node must have ReadWrite access, "
            f"but got access code {access_code}."
        )
        console.error(message=message, error=ValueError)

    # Coerces the string value to the node's native type.
    type_code = int(feature.node.principal_interface_type)
    typed_value: int | float | str | bool
    if type_code == _NodeType.INTEGER:
        typed_value = int(value)
    elif type_code == _NodeType.FLOAT:
        typed_value = float(value)
    elif type_code == _NodeType.BOOLEAN:
        typed_value = value.lower() in ("true", "1", "yes")
    else:
        typed_value = value

    try:
        feature.value = typed_value
    except Exception as e:
        message = f"Unable to write value '{typed_value}' to GenICam node '{name}': {e}"
        console.error(message=message, error=RuntimeError)


def apply_genicam_configuration(
    node_map: NodeMap,
    config: GenicamConfiguration,
    current_model: str,
    current_serial: str,
    *,
    strict: bool = False,
) -> None:
    """Applies the ReadWrite nodes from a ``GenicamConfiguration`` to the connected camera's node map.

    First validates that the camera identity matches and that all nodes in the configuration exist on the device
    and are writable. Then applies all node values. Aborts with an error if any validation or write step fails.

    Args:
        node_map: The GenICam node map object.
        config: The configuration instance containing ReadWrite nodes to apply.
        current_model: The model name of the currently connected camera.
        current_serial: The serial number of the currently connected camera.
        strict: Determines whether to abort on camera identity mismatch instead of warning.

    Raises:
        ValueError: If ``strict`` is True and a camera identity mismatch is detected, or if any node in the
            configuration is missing or not writable on the target device.
        RuntimeError: If any node write operation fails.
    """
    # Checks camera identity against the configuration metadata.
    mismatches: list[str] = []
    if config.camera_model != current_model:
        mismatches.append(f"model (config='{config.camera_model}', camera='{current_model}')")
    if config.camera_serial_number != current_serial:
        mismatches.append(f"serial (config='{config.camera_serial_number}', camera='{current_serial}')")

    if mismatches:
        mismatch_details = ", ".join(mismatches)
        message = f"Unable to apply GenICam configuration. Camera identity mismatch: {mismatch_details}."
        if strict:
            console.error(message=message, error=ValueError)
        else:
            console.echo(message=message, level=LogLevel.WARNING)

    # Validates that all configuration nodes exist on the device and are writable.
    for node_info in config.nodes:
        if not hasattr(node_map, node_info.name):
            message = (
                f"Unable to apply GenICam configuration. The node '{node_info.name}' does not exist on the "
                f"connected camera."
            )
            console.error(message=message, error=ValueError)

        access_code = int(getattr(node_map, node_info.name).node.get_access_mode())
        if access_code != _AccessMode.READ_WRITE:
            message = (
                f"Unable to apply GenICam configuration. The node '{node_info.name}' must have ReadWrite access, "
                f"but got access code {access_code}."
            )
            console.error(message=message, error=ValueError)

    # Applies all node values. Aborts on the first write failure.
    for node_info in config.nodes:
        try:
            getattr(node_map, node_info.name).value = node_info.value
        except Exception as e:
            message = f"Unable to apply GenICam configuration. Failed to write node '{node_info.name}': {e}"
            console.error(message=message, error=RuntimeError)
