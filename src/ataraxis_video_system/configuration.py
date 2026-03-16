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

DEFAULT_BLACKLISTED_NODES: frozenset[str] = frozenset({"CustomerIDKey", "CustomerValueKey", "TestPattern"})
"""Node names silently skipped during configuration enumeration and apply operations.

Some vendor-specific nodes report ReadWrite access but reject writes at the hardware level, causing spurious
errors. These nodes are excluded by default from all configuration operations. End users can override this set
via the ``blacklisted_nodes`` parameter on ``enumerate_genicam_nodes`` and ``apply_genicam_configuration``.
"""

_APPLY_PHASE_ORDER: tuple[tuple[str, ...], ...] = (
    # Phase 1 — Unlock: disables auto-controls and centering that lock dependent nodes.
    (
        "CenterX",
        "CenterY",
        "ExposureAuto",
        "GainAuto",
        "BalanceWhiteAuto",
        "BlackLevelAuto",
    ),
    # Phase 2 — Reset: zeroes offsets to maximize the available Width/Height range.
    (
        "OffsetX",
        "OffsetY",
    ),
    # Phase 3 — Format: pixel format, binning, decimation, and reversal change WidthMax/HeightMax.
    (
        "PixelFormat",
        "BinningHorizontal",
        "BinningVertical",
        "BinningHorizontalMode",
        "BinningVerticalMode",
        "DecimationHorizontal",
        "DecimationVertical",
        "DecimationHorizontalMode",
        "DecimationVerticalMode",
        "ReverseX",
        "ReverseY",
    ),
    # Phase 4 — Dimensions: sets Width/Height within the range established by phases 2-3.
    (
        "Width",
        "Height",
    ),
    # Phase 5 — Offsets: sets OffsetX/OffsetY now that Width/Height leave room for them.
    (
        "OffsetX",
        "OffsetY",
    ),
    # Phase 6 — Timing: exposure constrains max frame rate, so exposure is written first.
    (
        "ExposureMode",
        "AcquisitionFrameRateEnable",
        "ExposureTime",
        "AcquisitionFrameRate",
    ),
    # Phase 7 — Re-lock: re-enables auto-controls and centering if the target configuration uses them.
    (
        "CenterX",
        "CenterY",
        "ExposureAuto",
        "GainAuto",
        "BalanceWhiteAuto",
        "BlackLevelAuto",
    ),
)
"""SFNC-compliant node write ordering for ``apply_genicam_configuration``.

GenICam nodes have dynamic constraints defined by the SFNC standard (e.g., ``OffsetX.Max = SensorWidth - Width``).
Writing nodes in arbitrary order causes OutOfRangeException or AccessException errors. This tuple defines the
phases in which nodes must be written to satisfy all known dependency chains. Nodes that appear in multiple phases
(e.g., OffsetX in phases 2 and 5) are written with their reset value first, then their target value. Nodes not
listed in any phase are written after all phases complete.
"""

_PHASE_RESET_VALUES: dict[str, int | float | bool | str] = {
    "OffsetX": 0,
    "OffsetY": 0,
    "CenterX": False,
    "CenterY": False,
    "ExposureAuto": "Off",
    "GainAuto": "Off",
    "BalanceWhiteAuto": "Off",
    "BlackLevelAuto": "Off",
}
"""Reset values for nodes in the unlock and reset phases.

Nodes in phases 1-2 are written with these values (not their target values) to maximize the permissible range for
subsequent phases. Their target values are applied in later phases (phase 5 for offsets, phase 7 for auto-controls
and centering).
"""


@dataclass(frozen=True, slots=True)
class GenicamNodeInfo:
    """Stores the name and value of a single GenICam feature node."""

    name: str
    """The feature name of the node (e.g., "Width", "ExposureTime")."""
    value: int | float | str | bool
    """The current value of the node."""


@dataclass
class GenicamConfiguration(YamlConfig):
    """Stores a complete GenICam camera configuration with camera identity metadata."""

    camera_model: str = ""
    """The model name of the camera that produced this configuration."""
    camera_serial_number: str = ""
    """The serial number of the camera that produced this configuration."""
    nodes: list[GenicamNodeInfo] = field(default_factory=list)
    """The list of ReadWrite GenICam nodes with their current values."""


def enumerate_genicam_nodes(
    node_map: NodeMap,
    blacklisted_nodes: frozenset[str] = DEFAULT_BLACKLISTED_NODES,
) -> list[str]:
    """Collects the names of all writable leaf value nodes by walking the GenICam category tree from the root.

    Notes:
        Uses an iterative stack-based traversal starting from ``node_map.Root``. Collects ReadWrite nodes of type
        Integer, Float, Enumeration, Boolean, and String, skipping all other nodes. All node accesses are wrapped
        in try/except to gracefully handle locked or unavailable nodes. Nodes whose names appear in
        ``blacklisted_nodes`` are silently excluded.

    Args:
        node_map: The GenICam node map object.
        blacklisted_nodes: A set of node names to exclude from enumeration. Defaults to
            ``DEFAULT_BLACKLISTED_NODES``, which contains vendor-specific nodes known to report ReadWrite access
            but reject writes at the hardware level.

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
        # noinspection PyBroadException
        try:
            name: str = node.node.name
        except Exception:  # noqa: S112  # pragma: no cover
            continue

        # Skips already-visited nodes to avoid cycles in the category tree.
        if name in visited:
            continue
        visited.add(name)

        # Skips blacklisted nodes that are known to cause hardware-level write failures.
        if name in blacklisted_nodes:
            continue

        # Resolves the node's principal interface type to determine how to handle it.
        # noinspection PyBroadException
        try:
            type_code = int(node.node.principal_interface_type)
        except Exception:  # noqa: S112  # pragma: no cover
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
    """Reads a single readable value node from the GenICam node map and returns its name and current value.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to read (e.g., "Width", "ExposureTime").

    Returns:
        A ``GenicamNodeInfo`` instance containing the node's name and current value.

    Raises:
        AttributeError: If the named node does not exist on the node map.
        ValueError: If the node is not a readable value node.
    """
    # Accesses the named feature on the node map. Raises AttributeError if the node does not exist.
    feature = getattr(node_map, name)
    raw_node = feature.node

    # Rejects nodes that are not readable value nodes.
    type_code = int(raw_node.principal_interface_type)
    if type_code not in _VALUE_NODE_TYPES:  # pragma: no cover
        message = (
            f"Unable to read GenICam node '{name}'. The node must be a value type (Integer, Float, Boolean, "
            f"String, or Enumeration), but got type code {type_code}."
        )
        console.error(message=message, error=ValueError)

    access_code = int(raw_node.get_access_mode())
    if access_code not in (_AccessMode.READ_WRITE, _AccessMode.READ_ONLY):  # pragma: no cover
        message = (
            f"Unable to read GenICam node '{name}'. The node must have ReadWrite or ReadOnly access, "
            f"but got access code {access_code}."
        )
        console.error(message=message, error=ValueError)

    return GenicamNodeInfo(name=name, value=feature.value)


def format_genicam_node(node_map: NodeMap, name: str) -> str:
    """Reads a single readable GenICam feature node and returns a formatted string with its full metadata.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to read (e.g., "Width", "ExposureTime").

    Returns:
        A multi-line formatted string containing the node's name, type, value, access mode, description, numeric
        range, and enumeration entries.

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
    if type_code not in _VALUE_NODE_TYPES:  # pragma: no cover
        message = (
            f"Unable to format GenICam node '{name}'. The node must be a value type (Integer, Float, Boolean, "
            f"String, or Enumeration), but got type code {type_code}."
        )
        console.error(message=message, error=ValueError)

    if access_code not in (_AccessMode.READ_WRITE, _AccessMode.READ_ONLY):  # pragma: no cover
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
    if access_code != _AccessMode.READ_WRITE:  # pragma: no cover
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
    except Exception as e:  # pragma: no cover
        message = f"Unable to write value '{typed_value}' to GenICam node '{name}': {e}"
        console.error(message=message, error=RuntimeError)


def apply_genicam_configuration(
    node_map: NodeMap,
    config: GenicamConfiguration,
    current_model: str,
    current_serial: str,
    *,
    strict: bool = False,
    blacklisted_nodes: frozenset[str] = DEFAULT_BLACKLISTED_NODES,
) -> None:
    """Applies the ReadWrite nodes from a ``GenicamConfiguration`` to the connected camera's node map.

    First validates that the camera identity matches and that all non-blacklisted nodes in the configuration exist
    on the device and are writable. Then applies nodes in SFNC-compliant phase order to satisfy interdependent
    constraints (e.g., Width/OffsetX, GainAuto/Gain, binning/dimensions).

    Notes:
        GenICam SFNC defines dynamic constraints between nodes: ``OffsetX.Max = SensorWidth - Width``, auto-controls
        lock their manual counterparts, and binning changes ``WidthMax``/``HeightMax``. This function applies nodes
        in a fixed phase order defined by ``_APPLY_PHASE_ORDER`` to satisfy all known dependency chains. Phases 1-2
        write reset values (offsets to 0, auto-controls to Off) to unlock dependent nodes and maximize dimension
        ranges. Phases 3-7 write target values in dependency order. Remaining nodes not covered by any phase are
        written last.

    Args:
        node_map: The GenICam node map object.
        config: The configuration instance containing ReadWrite nodes to apply.
        current_model: The model name of the currently connected camera.
        current_serial: The serial number of the currently connected camera.
        strict: Determines whether to abort on camera identity mismatch instead of warning.
        blacklisted_nodes: A set of node names to silently skip during validation and write operations. Defaults to
            ``DEFAULT_BLACKLISTED_NODES``, which contains vendor-specific nodes known to report ReadWrite access
            but reject writes at the hardware level.

    Raises:
        ValueError: If ``strict`` is True and a camera identity mismatch is detected, or if any non-blacklisted
            node in the configuration is missing or not writable on the target device.
        RuntimeError: If any non-blacklisted node write operation fails.
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

    # Builds a lookup from node name to its target GenicamNodeInfo, filtering out blacklisted nodes.
    node_lookup: dict[str, GenicamNodeInfo] = {}
    for node_info in config.nodes:
        if node_info.name in blacklisted_nodes:
            continue
        node_lookup[node_info.name] = node_info

    # Validates that all nodes in the lookup exist on the device and are writable.
    for name in node_lookup:
        if not hasattr(node_map, name):
            message = (
                f"Unable to apply GenICam configuration. The node '{name}' does not exist on the connected camera."
            )
            console.error(message=message, error=ValueError)

        access_code = int(getattr(node_map, name).node.get_access_mode())
        if access_code != _AccessMode.READ_WRITE:  # pragma: no cover
            message = (
                f"Unable to apply GenICam configuration. The node '{name}' must have ReadWrite access, "
                f"but got access code {access_code}."
            )
            console.error(message=message, error=ValueError)

    # Tracks which nodes have been written so that remaining nodes can be applied after all phases.
    written: set[str] = set()

    # Applies nodes in SFNC-compliant phase order. Phases 1-2 use reset values from _PHASE_RESET_VALUES to unlock
    # dependent nodes and maximize dimension ranges. Phases 3+ use the target values from the configuration.
    reset_phases = frozenset(_APPLY_PHASE_ORDER[:2])
    for phase in _APPLY_PHASE_ORDER:
        use_reset_values = phase in reset_phases
        for name in phase:
            if name not in node_lookup:
                # Handles nodes that exist on the camera but are absent from the configuration (e.g., the
                # configuration was dumped from a camera without CenterX support). Reset-phase nodes are still
                # written with their safe defaults to unlock constraints.
                if use_reset_values and name in _PHASE_RESET_VALUES:
                    with suppress(Exception):
                        getattr(node_map, name).value = _PHASE_RESET_VALUES[name]
                continue

            value = (
                _PHASE_RESET_VALUES[name]
                if use_reset_values and name in _PHASE_RESET_VALUES
                else node_lookup[name].value
            )

            try:
                getattr(node_map, name).value = value
            except Exception as e:
                # Reset-phase failures are non-fatal — the node may not exist on this camera model.
                if use_reset_values:
                    continue
                message = f"Unable to apply GenICam configuration. Failed to write node '{name}': {e}"
                console.error(message=message, error=RuntimeError)

            # Only marks nodes as written when their target value was applied (not the reset value).
            if not use_reset_values:
                written.add(name)

    # Applies all remaining nodes that were not covered by any phase, in their original configuration order.
    for name, node_info in node_lookup.items():
        if name in written:
            continue

        try:
            getattr(node_map, name).value = node_info.value
        except Exception as e:  # pragma: no cover
            message = f"Unable to apply GenICam configuration. Failed to write node '{name}': {e}"
            console.error(message=message, error=RuntimeError)
