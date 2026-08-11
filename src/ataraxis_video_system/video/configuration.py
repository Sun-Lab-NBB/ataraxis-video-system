"""Provides data classes and helper functions for reading, writing, and managing GenICam camera configurations.

These utilities allow enumerating, inspecting, and modifying individual GenICam feature nodes, as well as
dumping and loading full camera configurations to and from YAML files.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any
from itertools import product
from contextlib import suppress
from dataclasses import field, dataclass

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from genicam.genapi import NodeMap


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


DEFAULT_BLACKLISTED_NODES: frozenset[str] = frozenset({"CustomerIDKey", "CustomerValueKey", "TestPattern"})
"""Node names silently skipped during configuration enumeration and apply operations.

Some vendor-specific nodes report ReadWrite access but reject writes at the hardware level, causing spurious
errors. These nodes are excluded by default from all configuration operations. End users can override this set
via the ``blacklisted_nodes`` parameter on ``enumerate_genicam_nodes``, ``read_genicam_nodes``, and
``apply_genicam_configuration``.
"""

_VALUE_NODE_TYPES: frozenset[int] = frozenset(
    {_NodeType.INTEGER, _NodeType.BOOLEAN, _NodeType.FLOAT, _NodeType.STRING, _NodeType.ENUMERATION}
)
"""The GenICam node type codes that represent collectible leaf value nodes."""

_APPLY_PHASE_ORDER: tuple[tuple[str, ...], ...] = (
    # Phase 1, Unlock: disables auto-controls and centering that lock dependent nodes.
    (
        "CenterX",
        "CenterY",
        "ExposureAuto",
        "GainAuto",
        "BalanceWhiteAuto",
        "BlackLevelAuto",
    ),
    # Phase 2, Reset: zeroes offsets to maximize the available Width/Height range.
    (
        "OffsetX",
        "OffsetY",
    ),
    # Phase 3, Format: pixel format, binning, decimation, and reversal change WidthMax/HeightMax.
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
    # Phase 4, Dimensions: sets Width/Height within the range established by phases 2-3.
    (
        "Width",
        "Height",
    ),
    # Phase 5, Offsets: sets OffsetX/OffsetY now that Width/Height leave room for them.
    (
        "OffsetX",
        "OffsetY",
    ),
    # Phase 6, Timing: exposure constrains max frame rate, so exposure is written first.
    (
        "ExposureMode",
        "AcquisitionFrameRateEnable",
        "ExposureTime",
        "AcquisitionFrameRate",
    ),
    # Phase 7, Analog: manual analog values, written while the auto-controls that own them are still disengaged.
    (
        "Gain",
        "BlackLevel",
        "BalanceRatio",
    ),
    # Phase 8, Re-lock: re-enables auto-controls and centering if the target configuration uses them.
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

_BOOLEAN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes"})
"""The literals a Boolean node write accepts as True, compared against the lowered input value."""

_BOOLEAN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no"})
"""The literals a Boolean node write accepts as False, compared against the lowered input value."""

_MAXIMUM_SELECTOR_COMBINATIONS: int = 64
"""The largest number of selector combinations enumerated for a single node.

A node addressed by several selectors expands into the product of their entries, so this ceiling bounds the read
and write cost on cameras that expose large selector sets.
"""

_RESET_PHASE_COUNT: int = 2
"""The number of leading phases in ``_APPLY_PHASE_ORDER`` that write reset values instead of target values.

Phases are identified by position rather than by content, because the later phases that restore the target values
repeat the node names of the leading phases that reset them.
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
subsequent phases. Their target values are applied in later phases (phase 5 for offsets, phase 8 for auto-controls
and centering).
"""


@dataclass(frozen=True, slots=True)
class GenicamNodeInfo:
    """Stores the name and value of a single GenICam feature node."""

    name: str
    """The feature name of the node (e.g., "Width", "ExposureTime")."""
    value: int | float | str | bool
    """The current value of the node."""
    selectors: dict[str, str | int] = field(default_factory=dict)
    """The selector node values that address the instance this entry describes, empty for an unselected node.

    SFNC multiplexes some features behind a selector, so a camera holds one ``BalanceRatio`` per
    ``BalanceRatioSelector`` entry rather than a single value. This mapping pins the instance the value belongs to,
    and it is applied to the camera before the value is read or written.
    """


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
        try:
            name: str = node.node.name
        except Exception:  # noqa: S112
            continue

        # Skips already-visited nodes to avoid cycles in the category tree.
        if name in visited:
            continue
        visited.add(name)

        # Skips blacklisted nodes that are known to cause hardware-level write failures.
        if name in blacklisted_nodes:
            continue

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


def read_genicam_nodes(
    node_map: NodeMap,
    blacklisted_nodes: frozenset[str] = DEFAULT_BLACKLISTED_NODES,
) -> list[GenicamNodeInfo]:
    """Reads every writable node of the connected camera, including each instance of a selector-addressed node.

    Notes:
        A node that SFNC multiplexes behind a selector holds one value per selector combination, so it contributes
        one entry per combination rather than a single entry. Reading those values moves the selectors, so the
        selector positions the camera started from are restored before this function returns.

    Args:
        node_map: The GenICam node map object.
        blacklisted_nodes: A set of node names to exclude from the result. Defaults to ``DEFAULT_BLACKLISTED_NODES``.

    Returns:
        A list of ``GenicamNodeInfo`` instances covering every writable node instance that could be read.
    """
    names = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=blacklisted_nodes)

    # Captures the position of every selector before any of them is moved, so that reading the configuration leaves
    # the camera exactly as it was found.
    selector_names = {selector for name in names for selector in _get_selecting_features(node_map=node_map, name=name)}
    original_positions: dict[str, str | int] = {}
    for selector_name in selector_names:
        with suppress(Exception):
            original_positions[selector_name] = getattr(node_map, selector_name).value

    nodes: list[GenicamNodeInfo] = []
    try:
        for name in names:
            # A selector is recorded at the position the camera was found in, because stepping the nodes it
            # addresses moves it away from that position before this loop reaches it.
            if name in original_positions:
                nodes.append(GenicamNodeInfo(name=name, value=original_positions[name], selectors={}))
                continue

            for selectors in _expand_selectors(node_map=node_map, name=name):
                with suppress(Exception):
                    _apply_selectors(node_map=node_map, selectors=selectors)
                    value = read_genicam_node(node_map=node_map, name=name).value
                    nodes.append(GenicamNodeInfo(name=name, value=value, selectors=dict(selectors)))
    finally:
        for selector_name, selector_value in original_positions.items():
            with suppress(Exception):
                getattr(node_map, selector_name).value = selector_value

    return nodes


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

    return GenicamNodeInfo(name=name, value=feature.value)


def format_genicam_node(node_map: NodeMap, name: str) -> str:
    """Reads a single readable GenICam feature node and returns a formatted string with its full metadata.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to read (e.g., "Width", "ExposureTime").

    Returns:
        A multi-line formatted string containing the node's name, type, value, access mode, description, numeric
        range, step increment (for Integer nodes), enumeration entries, and measurement unit (when defined).

    Raises:
        AttributeError: If the named node does not exist on the node map.
        ValueError: If the node is not a value type (Integer, Float, Boolean, String, or Enumeration), or is not
            readable (must be ReadWrite or ReadOnly).
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

    # Appends the measurement unit if the node defines one. GenICam exposes the unit on the feature rather than on
    # the INode descriptor, and only the numeric node types carry one at all.
    with suppress(Exception):
        raw_unit = str(feature.unit)
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
        AttributeError: If the named node does not exist on the node map.
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
        typed_value = _coerce_boolean(name=name, value=value)
    else:
        typed_value = value

    try:
        feature.value = typed_value
    except Exception as error:
        message = f"Unable to write value '{typed_value}' to GenICam node '{name}': {error}"
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
    on the device. Then applies nodes in SFNC-compliant phase order to satisfy interdependent constraints (e.g.,
    Width/OffsetX, GainAuto/Gain, binning/dimensions), validating that every node is writable once the unlock phase
    has run.

    Notes:
        GenICam SFNC defines dynamic constraints between nodes: ``OffsetX.Max = SensorWidth - Width``, auto-controls
        lock their manual counterparts, and binning changes ``WidthMax``/``HeightMax``. This function applies nodes
        in a fixed phase order defined by ``_APPLY_PHASE_ORDER`` to satisfy all known dependency chains. Phases 1-2
        write reset values (offsets to 0, auto-controls to Off) to unlock dependent nodes and maximize dimension
        ranges. Phases 3-8 write target values in dependency order. Remaining nodes not covered by any phase are
        written last.

        The writability check runs after phase 1 rather than before it, because an engaged auto-control holds its
        manual counterpart at ReadOnly until that phase disengages it.

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
        RuntimeError: If a non-blacklisted node write outside the two reset phases fails. Reset-phase write failures
            are non-fatal, since the later phases write the node's target value regardless.
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

    # Builds a lookup from node name to its target values, filtering out blacklisted nodes. A node addressed by a
    # selector contributes one entry per selector combination, so each name maps to a list.
    node_lookup: dict[str, list[GenicamNodeInfo]] = {}
    for node_info in config.nodes:
        if node_info.name in blacklisted_nodes:
            continue
        node_lookup.setdefault(node_info.name, []).append(node_info)

    # Validates that all nodes in the lookup exist on the device.
    for name in node_lookup:
        if not hasattr(node_map, name):
            message = (
                f"Unable to apply GenICam configuration. The node '{name}' does not exist on the connected camera."
            )
            console.error(message=message, error=ValueError)

    # Tracks which nodes have been written so that remaining nodes can be applied after all phases.
    written: set[str] = set()

    # Runs the unlock phase ahead of the writability check, because an engaged auto-control holds its manual
    # counterpart at ReadOnly until it is disengaged.
    _write_apply_phase(
        node_map=node_map,
        phase=_APPLY_PHASE_ORDER[0],
        node_lookup=node_lookup,
        written=written,
        blacklisted_nodes=blacklisted_nodes,
        use_reset_values=True,
    )

    # Validates that all nodes in the lookup are writable now that the auto-controls have been disengaged.
    for name in node_lookup:
        access_code = int(getattr(node_map, name).node.get_access_mode())
        if access_code != _AccessMode.READ_WRITE:
            message = (
                f"Unable to apply GenICam configuration. The node '{name}' must have ReadWrite access, "
                f"but got access code {access_code}."
            )
            console.error(message=message, error=ValueError)

    # Applies the remaining phases in SFNC-compliant order. Phase membership is resolved by position, because the
    # re-lock and offset phases repeat the node names of the unlock and reset phases they undo.
    for phase_index, phase in enumerate(_APPLY_PHASE_ORDER[1:], start=1):
        _write_apply_phase(
            node_map=node_map,
            phase=phase,
            node_lookup=node_lookup,
            written=written,
            blacklisted_nodes=blacklisted_nodes,
            use_reset_values=phase_index < _RESET_PHASE_COUNT,
        )

    # Applies all remaining nodes that were not covered by any phase, in their original configuration order.
    for name, node_infos in node_lookup.items():
        if name in written:
            continue

        for node_info in node_infos:
            try:
                _apply_selectors(node_map=node_map, selectors=node_info.selectors)
                getattr(node_map, name).value = node_info.value
            except Exception as error:
                message = f"Unable to apply GenICam configuration. Failed to write node '{name}': {error}"
                console.error(message=message, error=RuntimeError)


def _write_apply_phase(
    node_map: NodeMap,
    phase: tuple[str, ...],
    node_lookup: dict[str, list[GenicamNodeInfo]],
    written: set[str],
    blacklisted_nodes: frozenset[str],
    *,
    use_reset_values: bool,
) -> None:
    """Writes every node of a single ``apply_genicam_configuration`` phase to the connected camera.

    Args:
        node_map: The GenICam node map object.
        phase: The node names that belong to the phase, in the order they must be written.
        node_lookup: The target values of every non-blacklisted node, keyed by node name.
        written: The names of the nodes that have received their target value. Nodes this call writes with their
            target value are added to it.
        blacklisted_nodes: The node names this call writes nothing to, whatever phase it is running.
        use_reset_values: Determines whether nodes covered by ``_PHASE_RESET_VALUES`` receive their reset value
            instead of their target value.

    Raises:
        RuntimeError: If a node write outside a reset phase fails.
    """
    for name in phase:
        # A blacklisted node is skipped by every phase, including the reset phases. Filtering it out of node_lookup
        # alone leaves it matching the branch below that writes a safe default to a node the configuration omits,
        # which is exactly the hardware write the caller asked this function not to perform.
        if name in blacklisted_nodes:
            continue

        if name not in node_lookup:
            # Handles nodes that exist on the camera but are absent from the configuration (e.g., the
            # configuration was dumped from a camera without CenterX support). Reset-phase nodes are still
            # written with their safe defaults to unlock constraints.
            if use_reset_values and name in _PHASE_RESET_VALUES:
                with suppress(Exception):
                    getattr(node_map, name).value = _PHASE_RESET_VALUES[name]
            continue

        for node_info in node_lookup[name]:
            value = _PHASE_RESET_VALUES[name] if use_reset_values and name in _PHASE_RESET_VALUES else node_info.value

            try:
                _apply_selectors(node_map=node_map, selectors=node_info.selectors)
                getattr(node_map, name).value = value
            except Exception as error:
                # Reset-phase failures are non-fatal, as the reset value only has to unlock the constraints the
                # later phases write through. A camera that rejects it is written its target value regardless.
                if use_reset_values:
                    continue
                message = f"Unable to apply GenICam configuration. Failed to write node '{name}': {error}"
                console.error(message=message, error=RuntimeError)

        # Only marks nodes as written when their target value was applied (not the reset value).
        if not use_reset_values:
            written.add(name)


def _coerce_boolean(name: str, value: str) -> bool:
    """Coerces the string representation of a Boolean node's value into the boolean the node accepts.

    Notes:
        A literal outside both vocabularies is rejected rather than coerced. A bare membership test maps every
        unrecognized literal to False, so a caller asking for 'On' would silently disable the feature and be told the
        write succeeded.

    Args:
        name: The feature name of the node the value is written to.
        value: The string representation of the value to coerce.

    Returns:
        The coerced boolean value.

    Raises:
        ValueError: If the value names neither a true nor a false literal.
    """
    lowered = value.lower()

    if lowered in _BOOLEAN_TRUE_VALUES:
        return True

    if lowered in _BOOLEAN_FALSE_VALUES:
        return False

    accepted = ", ".join(sorted(_BOOLEAN_TRUE_VALUES | _BOOLEAN_FALSE_VALUES))
    message = (
        f"Unable to write to GenICam node '{name}'. The node is a Boolean, so the written value must be one of "
        f"{accepted}, but got '{value}'."
    )
    console.error(message=message, error=ValueError)

    # Satisfies ruff RET503. console.error() is NoReturn, so this line never executes.
    raise ValueError(message)  # pragma: no cover - console.error() is NoReturn, this satisfies ruff RET503.


def _get_selecting_features(node_map: NodeMap, name: str) -> list[str]:
    """Returns the names of the selector nodes that address the named node.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to inspect.

    Returns:
        A sorted list of selector node names, empty for a node no selector addresses.
    """
    try:
        feature = getattr(node_map, name)
        return sorted(str(selector.node.name) for selector in feature.node.selecting_features)
    except Exception:  # pragma: no cover - every supported genicam build exposes selecting_features.
        return []


def _get_selector_values(selector: Any) -> list[str | int]:
    """Collects every value the selector accepts that the connected camera implements.

    Args:
        selector: The selector feature object.

    Returns:
        The symbolic entries of an Enumeration selector or the permitted values of an Integer selector, empty when
        neither applies.
    """
    values: list[str | int] = []
    type_code = int(selector.node.principal_interface_type)

    if type_code == _NodeType.ENUMERATION:
        for entry in selector.entries:
            # Entries the camera model does not implement report an unreadable access mode and are skipped.
            with suppress(Exception):
                if int(entry.node.get_access_mode()) in (_AccessMode.READ_WRITE, _AccessMode.READ_ONLY):
                    values.append(str(entry.symbolic))
    elif type_code == _NodeType.INTEGER:
        with suppress(Exception):
            values.extend(range(int(selector.min), int(selector.max) + 1, max(int(selector.inc), 1)))

    return values


def _expand_selectors(node_map: NodeMap, name: str) -> list[dict[str, str | int]]:
    """Resolves every selector combination under which the named node holds a separate value.

    Notes:
        A node addressed by several selectors expands into the product of their values. The expansion is capped at
        ``_MAXIMUM_SELECTOR_COMBINATIONS`` entries, and the truncation is reported to the user.

    Args:
        node_map: The GenICam node map object.
        name: The feature name of the node to expand.

    Returns:
        A list of selector mappings, holding a single empty mapping for a node no selector addresses.
    """
    axes: list[list[tuple[str, str | int]]] = []
    for selector_name in _get_selecting_features(node_map=node_map, name=name):
        selector = getattr(node_map, selector_name, None)
        if selector is None:  # pragma: no cover - a selector the node map names always resolves on that node map.
            continue

        # A selector the camera does not allow writing cannot be stepped, so the node keeps its current instance. No
        # self-consistent camera reaches this state, since a camera that gates a selector also gates what it addresses.
        with suppress(Exception):
            if int(selector.node.get_access_mode()) != _AccessMode.READ_WRITE:  # pragma: no cover
                continue

        values = _get_selector_values(selector=selector)
        if values:
            axes.append([(selector_name, value) for value in values])

    if not axes:
        return [{}]

    combinations = [dict(combination) for combination in product(*axes)]
    if len(combinations) > _MAXIMUM_SELECTOR_COMBINATIONS:
        message = (
            f"The GenICam node '{name}' is addressed by {len(combinations)} selector combinations, which exceeds the "
            f"ceiling of {_MAXIMUM_SELECTOR_COMBINATIONS}. Only the first {_MAXIMUM_SELECTOR_COMBINATIONS} "
            f"combinations are covered."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        combinations = combinations[:_MAXIMUM_SELECTOR_COMBINATIONS]

    return combinations


def _apply_selectors(node_map: NodeMap, selectors: dict[str, str | int]) -> None:
    """Positions the selector nodes so that a selected feature addresses the intended instance.

    Args:
        node_map: The GenICam node map object.
        selectors: The selector values to write, keyed by selector node name.
    """
    for selector_name, selector_value in selectors.items():
        getattr(node_map, selector_name).value = selector_value
