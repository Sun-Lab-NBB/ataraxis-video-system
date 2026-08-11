"""Provides a synthetic GenICam node map that reproduces the SFNC feature interdependencies of real hardware.

The bundled GenTL Producer simulator exposes ten writable nodes whose only interdependency is the degenerate
single-entry EventSelector and EventNotification pair, so it cannot exercise the dependency chains that
``apply_genicam_configuration`` orders its write phases around. This module models those chains directly: binning
rescales the addressable sensor area, offsets and dimensions compete for that area. The auto-controls lock their
manual counterparts, exposure bounds the attainable frame rate, an enumeration selector addresses one value per
color channel, and an integer selector addresses one value per lookup table index. It also models the states that
make a camera refuse an operation: a gated node the camera reports as NotAvailable. The remaining two states are a
vendor node typed differently from the literal the apply phases reset it with, and a node whose descriptor cannot be
interrogated at all.
"""

from typing import Any
from dataclasses import field, dataclass

_INTEGER: int = 2
"""Stores the GenICam principal interface type code for Integer nodes."""

_BOOLEAN: int = 3
"""Stores the GenICam principal interface type code for Boolean nodes."""

_FLOAT: int = 5
"""Stores the GenICam principal interface type code for Float nodes."""

_STRING: int = 6
"""Stores the GenICam principal interface type code for String nodes."""

_CATEGORY: int = 8
"""Stores the GenICam principal interface type code for Category nodes."""

_ENUMERATION: int = 9
"""Stores the GenICam principal interface type code for Enumeration nodes."""

_NOT_AVAILABLE: int = 1
"""Stores the GenICam access mode code for nodes the camera implements but currently gates off."""

_READ_ONLY: int = 3
"""Stores the GenICam access mode code for ReadOnly nodes."""

_READ_WRITE: int = 4
"""Stores the GenICam access mode code for ReadWrite nodes."""

SENSOR_WIDTH: int = 1920
"""Stores the full addressable sensor width of the synthetic camera, in pixels."""

_SENSOR_HEIGHT: int = 1080
"""Stores the full addressable sensor height of the synthetic camera, in pixels."""

LUT_INDEX_MAXIMUM: int = 3
"""Stores the largest lookup table index the synthetic camera addresses unless a test asks for a deeper table."""

HOSTILE_NODE_NAME: str = "UnreadableFeature"
"""Stores the feature name of the hostile node whose type, rather than whose name, is unreadable."""

_LUT_VALUE_MAXIMUM: int = 4095
"""Stores the largest value a lookup table entry of the synthetic camera accepts."""

_LUT_VALUE_STEP: int = 10
"""Stores the spacing between the values the lookup table entries hold before a test writes its own.

The entries differ so that a configuration collapsing them into a single entry is detectable.
"""


@dataclass(slots=True)
class _NodeSpecification:
    """Defines the static description of a single synthetic GenICam node."""

    type_code: int
    """The GenICam principal interface type code of the node."""
    category: str
    """The name of the category node that owns this node."""
    description: str = ""
    """The human-readable description reported for the node."""
    unit: str = ""
    """The measurement unit reported for the node, or an empty string when the node defines none."""
    entries: tuple[str, ...] = ()
    """The valid symbolic entries of an Enumeration node."""
    increment: int = 1
    """The step increment reported for an Integer node."""
    read_only: bool = False
    """Determines whether the node is permanently ReadOnly regardless of the state of any other node."""
    extra_categories: tuple[str, ...] = ()
    """The names of the additional category nodes that reference this node without owning it.

    A GenICam category references its features by pointer, so a vendor grouping category lists features another
    category already owns. Such a node is reachable through more than one branch of the category tree.
    """


_NODE_SPECIFICATIONS: dict[str, _NodeSpecification] = {
    "DeviceModelName": _NodeSpecification(type_code=_STRING, category="DeviceControl", read_only=True),
    "BinningHorizontal": _NodeSpecification(
        type_code=_INTEGER, category="ImageFormatControl", description="Horizontal binning factor."
    ),
    "PixelFormat": _NodeSpecification(
        type_code=_ENUMERATION, category="ImageFormatControl", entries=("Mono8", "Mono12", "BGR8")
    ),
    "ReverseX": _NodeSpecification(type_code=_BOOLEAN, category="ImageFormatControl"),
    "Width": _NodeSpecification(
        type_code=_INTEGER, category="ImageFormatControl", description="Frame width.", unit="px", increment=4
    ),
    "Height": _NodeSpecification(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    "OffsetX": _NodeSpecification(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    "OffsetY": _NodeSpecification(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    # Centering is a vendor extension rather than an SFNC Boolean on every camera, so this camera types it as an
    # Enumeration and rejects the Boolean literal the apply phases reset it with.
    "CenterX": _NodeSpecification(type_code=_ENUMERATION, category="ImageFormatControl", entries=("Off", "On")),
    "LUTIndex": _NodeSpecification(type_code=_INTEGER, category="LUTControl"),
    "LUTValue": _NodeSpecification(type_code=_INTEGER, category="LUTControl"),
    "ExposureAuto": _NodeSpecification(
        type_code=_ENUMERATION, category="AcquisitionControl", entries=("Off", "Once", "Continuous")
    ),
    "ExposureTime": _NodeSpecification(
        type_code=_FLOAT,
        category="AcquisitionControl",
        description="Exposure duration.",
        unit="us",
        extra_categories=("QuickSetupControl",),
    ),
    "AcquisitionFrameRateEnable": _NodeSpecification(type_code=_BOOLEAN, category="AcquisitionControl"),
    "AcquisitionFrameRate": _NodeSpecification(type_code=_FLOAT, category="AcquisitionControl", unit="Hz"),
    "GainAuto": _NodeSpecification(
        type_code=_ENUMERATION, category="AnalogControl", entries=("Off", "Once", "Continuous")
    ),
    "Gain": _NodeSpecification(type_code=_FLOAT, category="AnalogControl", unit="dB"),
    "BalanceRatioSelector": _NodeSpecification(
        type_code=_ENUMERATION, category="AnalogControl", entries=("Red", "Green", "Blue")
    ),
    "BalanceRatio": _NodeSpecification(type_code=_FLOAT, category="AnalogControl", unit="dB"),
}
"""Describes every node the synthetic camera implements, keyed by feature name."""

_SELECTED_BY: dict[str, tuple[str, ...]] = {
    "BalanceRatio": ("BalanceRatioSelector",),
    "LUTValue": ("LUTIndex",),
}
"""Maps each selector-addressed node to the selector nodes that address it.

SFNC multiplexes these nodes, so the camera holds one value per selector position rather than a single value.
"""

_SELECTED_DEFAULTS: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {
    ("BalanceRatio", (("BalanceRatioSelector", "Red"),)): 1.0,
    ("BalanceRatio", (("BalanceRatioSelector", "Green"),)): 2.0,
    ("BalanceRatio", (("BalanceRatioSelector", "Blue"),)): 3.0,
}
"""Stores the distinct value each selector-addressed instance holds before a test applies its own starting state.

The three balance ratios differ so that a configuration collapsing them into one entry is detectable. The lookup
table instances are seeded by the constructor instead, because how many of them exist depends on the depth of the
table the camera implements.
"""

_CATEGORY_ORDER: tuple[str, ...] = (
    "DeviceControl",
    "ImageFormatControl",
    "LUTControl",
    "AcquisitionControl",
    "AnalogControl",
    "QuickSetupControl",
)
"""Orders the category nodes as they appear under the synthetic root node.

QuickSetupControl is a vendor convenience grouping that owns no node of its own and instead re-references nodes the
SFNC categories own, which is how a real node map exposes the same feature through two branches of the tree.
"""

_DEFAULT_VALUES: dict[str, Any] = {
    "DeviceModelName": "SyntheticCamera",
    "BinningHorizontal": 1,
    "PixelFormat": "Mono8",
    "ReverseX": False,
    "Width": SENSOR_WIDTH,
    "Height": _SENSOR_HEIGHT,
    "OffsetX": 0,
    "OffsetY": 0,
    "CenterX": "Off",
    "LUTIndex": 0,
    "ExposureAuto": "Off",
    "ExposureTime": 5000.0,
    "AcquisitionFrameRateEnable": True,
    "AcquisitionFrameRate": 30.0,
    "GainAuto": "Off",
    "Gain": 0.0,
    "BalanceRatioSelector": "Red",
}
"""Stores the value each node no selector addresses holds before a test applies its own starting state."""


class SyntheticNodeMapError(Exception):
    """Signals that a write violated a constraint the synthetic camera enforces."""


@dataclass(slots=True)
class _RawNode:
    """Exposes the descriptor half of a synthetic node, mirroring the ``feature.node`` accessor of GenICam.

    Notes:
        GenICam splits a node across two objects. This class stands in for the ``INode`` descriptor, which carries
        the name, type, access mode, and description. Value-related metadata, including the measurement unit, lives
        on the feature object instead, so this class deliberately omits it.
    """

    name: str
    """The feature name of the node."""
    principal_interface_type: int
    """The GenICam principal interface type code of the node."""
    description: str
    """The human-readable description of the node."""
    access_mode: int
    """The GenICam access mode code the node currently reports."""
    node_map: Any = None
    """The node map that owns this node, used to resolve the selector relationships."""

    def get_access_mode(self) -> int:
        """Returns the access mode code the node currently reports."""
        return self.access_mode

    def is_selector(self) -> bool:
        """Returns True if this node addresses the value of at least one other node."""
        return any(self.name in selectors for selectors in _SELECTED_BY.values())

    @property
    def selected_features(self) -> list[Any]:
        """Returns the features whose value this node addresses."""
        return [
            _Feature(node_map=self.node_map, name=name)
            for name, selectors in _SELECTED_BY.items()
            if self.name in selectors
        ]

    @property
    def selecting_features(self) -> list[Any]:
        """Returns the selector features that address the value of this node."""
        return [_Feature(node_map=self.node_map, name=name) for name in _SELECTED_BY.get(self.name, ())]


@dataclass(slots=True)
class _HostileRawNode:
    """Exposes a node descriptor that raises instead of reporting the metadata field the hostile mode names.

    Notes:
        A device that drops off the bus, or whose XML entry is malformed, makes the GenApi accessors raise for the
        nodes the enumeration walk has not reached yet. This class reproduces that state for a single node.
    """

    mode: str
    """The descriptor field that raises when it is read: 'name' or 'type'."""

    @property
    def name(self) -> str:
        """Returns the feature name of the node, or raises when the node models an unreadable name."""
        if self.mode == "name":
            message = "The name of this node is unreadable, as the device dropped off the bus."
            raise RuntimeError(message)
        return HOSTILE_NODE_NAME

    @property
    def principal_interface_type(self) -> int:
        """Returns the type code of the node, or raises when the node models an unreadable type."""
        if self.mode == "type":
            message = "The type of this node is unreadable, as the device dropped off the bus."
            raise RuntimeError(message)
        return _INTEGER


@dataclass(slots=True)
class _HostileFeature:
    """Exposes a node whose descriptor cannot be interrogated, mirroring a feature of a device that vanished."""

    mode: str
    """The descriptor field that raises when it is read: 'name' or 'type'."""

    @property
    def node(self) -> _HostileRawNode:
        """Returns the descriptor of the node."""
        return _HostileRawNode(mode=self.mode)


@dataclass(slots=True)
class _EnumerationEntry:
    """Exposes a single symbolic entry of a synthetic Enumeration node."""

    node: _RawNode
    """The descriptor of the entry."""
    symbolic: str
    """The symbolic name of the entry."""


@dataclass(slots=True)
class _Category:
    """Exposes a synthetic category node and the features it contains."""

    node: _RawNode
    """The descriptor of the category."""
    features: list[Any] = field(default_factory=list)
    """The child nodes the category contains."""


class SyntheticNodeMap:
    """Emulates a GenICam node map whose features constrain one another the way SFNC defines.

    Binning rescales the addressable sensor area, ``Width`` and ``OffsetX`` compete for that area, the auto-controls
    lock their manual counterparts to ReadOnly while they are engaged, and ``ExposureTime`` bounds the attainable
    ``AcquisitionFrameRate``. Writes that violate a constraint raise ``SyntheticNodeMapError``, mirroring the
    exceptions a camera raises when nodes are written out of order.

    Args:
        overrides: The node values to apply on top of the defaults, establishing the starting state of the camera.
        hostile_modes: The descriptor fields that raise when the enumeration walk reads them. Each entry adds one
            node to the root category whose name ('name') or type ('type') cannot be interrogated.
        lut_index_maximum: The largest lookup table index the camera addresses, which sets the depth of the table.

    Attributes:
        values: The current value of every node no selector addresses, keyed by feature name.
        selected_values: The current value of every selector-addressed node instance, keyed by the node name paired
            with the selector positions that address it.
        writes: The feature name of every successful write, in the order the writes occurred.
        lut_index_maximum: The largest lookup table index the camera addresses.
    """

    def __init__(
        self,
        overrides: dict[str, Any] | None = None,
        hostile_modes: tuple[str, ...] = (),
        lut_index_maximum: int = LUT_INDEX_MAXIMUM,
    ) -> None:
        self.values: dict[str, Any] = dict(_DEFAULT_VALUES)
        self.selected_values: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = dict(_SELECTED_DEFAULTS)
        self.writes: list[str] = []
        self.lut_index_maximum = lut_index_maximum
        self._hostile_modes = hostile_modes

        # Seeds one lookup table entry per index the table implements, as the depth of the table is not fixed.
        for index in range(lut_index_maximum + 1):
            self.selected_values[("LUTValue", (("LUTIndex", index),))] = index * _LUT_VALUE_STEP

        # Seeds the starting state without constraint checking so that a test can describe any camera state directly.
        if overrides is not None:
            self.values.update(overrides)

    def __getattr__(self, name: str) -> Any:
        """Resolves a feature or category node by name, mirroring GenICam node map attribute access."""
        if name == "Root":
            return self._build_root()
        if name in _NODE_SPECIFICATIONS:
            return _Feature(node_map=self, name=name)
        message = f"The synthetic node map does not implement the node '{name}'."
        raise AttributeError(message)

    def instance_key(self, name: str) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Builds the storage key that identifies the instance of the named node the selectors currently address."""
        selectors = _SELECTED_BY.get(name, ())
        return (name, tuple((selector, self.values[selector]) for selector in sorted(selectors)))

    def read(self, name: str) -> Any:
        """Returns the value of the instance of the named node the selectors currently address."""
        if name in _SELECTED_BY:
            return self.selected_values[self.instance_key(name=name)]
        return self.values[name]

    def access_mode(self, name: str) -> int:
        """Returns the access mode code the named node currently reports.

        Notes:
            The manual counterpart of an engaged auto-control reports ReadOnly, which is how a camera prevents the
            manual value from being written while the auto-control owns it. A node the camera gates off entirely
            reports NotAvailable instead, which is how a camera hides a feature its current mode does not provide.
        """
        if _NODE_SPECIFICATIONS[name].read_only:
            return _READ_ONLY
        if name == "Gain" and self.values["GainAuto"] != "Off":
            return _READ_ONLY
        if name == "ExposureTime" and self.values["ExposureAuto"] != "Off":
            return _READ_ONLY
        if name == "AcquisitionFrameRate" and not self.values["AcquisitionFrameRateEnable"]:
            return _NOT_AVAILABLE
        return _READ_WRITE

    def bounds(self, name: str) -> tuple[int | float, int | float]:
        """Returns the lower and upper bound the named node currently accepts."""
        binning = self.values["BinningHorizontal"]
        addressable_width = SENSOR_WIDTH // binning

        if name == "Width":
            return 8, addressable_width - self.values["OffsetX"]
        if name == "OffsetX":
            return 0, addressable_width - self.values["Width"]
        if name == "Height":
            return 8, _SENSOR_HEIGHT - self.values["OffsetY"]
        if name == "OffsetY":
            return 0, _SENSOR_HEIGHT - self.values["Height"]
        if name == "BinningHorizontal":
            return 1, 4
        if name == "LUTIndex":
            return 0, self.lut_index_maximum
        if name == "LUTValue":
            return 0, _LUT_VALUE_MAXIMUM
        if name == "ExposureTime":
            return 10.0, 1000000.0
        if name == "AcquisitionFrameRate":
            # Exposure bounds the attainable rate, which is why exposure is written first within the timing phase.
            return 1.0, 1000000.0 / self.values["ExposureTime"]
        if name == "Gain":
            return 0.0, 48.0
        if name == "BalanceRatio":
            return 0.0, 8.0
        return 0, 0

    def write(self, name: str, value: Any) -> None:
        """Writes a value to the named node after enforcing every constraint the node participates in.

        Raises:
            SyntheticNodeMapError: If the node is not currently writable, the value falls outside the bounds the node
                accepts, or the value is not a valid entry of an Enumeration node.
        """
        access_mode = self.access_mode(name=name)
        if access_mode != _READ_WRITE:
            message = f"The node '{name}' is not writable, as it currently reports access mode {access_mode}."
            raise SyntheticNodeMapError(message)

        specification = _NODE_SPECIFICATIONS[name]
        if specification.type_code == _ENUMERATION and value not in specification.entries:
            message = f"The value '{value}' is not a valid entry of the node '{name}'."
            raise SyntheticNodeMapError(message)

        if specification.type_code in (_INTEGER, _FLOAT):
            minimum, maximum = self.bounds(name=name)
            if not minimum <= value <= maximum:
                message = f"The value '{value}' falls outside the range [{minimum}, {maximum}] of the node '{name}'."
                raise SyntheticNodeMapError(message)

        # A selector-addressed node stores one value per selector position, so the write lands on the instance the
        # selectors currently point at rather than on a single shared slot.
        if name in _SELECTED_BY:
            self.selected_values[self.instance_key(name=name)] = value
        else:
            self.values[name] = value
        self.writes.append(name)

        # Reducing the addressable area shrinks the frame to fit, matching how cameras clamp dimensions on a binning
        # change rather than rejecting the binning write.
        if name == "BinningHorizontal":
            addressable_width = SENSOR_WIDTH // value
            self.values["OffsetX"] = min(self.values["OffsetX"], addressable_width - 8)
            self.values["Width"] = min(self.values["Width"], addressable_width - self.values["OffsetX"])

    def _build_root(self) -> _Category:
        """Assembles the category tree that node enumeration walks."""
        categories: dict[str, _Category] = {}
        for category_name in _CATEGORY_ORDER:
            categories[category_name] = _Category(
                node=_RawNode(
                    name=category_name,
                    principal_interface_type=_CATEGORY,
                    description="",
                    access_mode=_READ_ONLY,
                    node_map=self,
                )
            )

        for node_name, specification in _NODE_SPECIFICATIONS.items():
            categories[specification.category].features.append(_Feature(node_map=self, name=node_name))

            # A category that references a node another category owns puts the same node on two branches of the
            # tree, which is what makes the walk reach it twice.
            for category_name in specification.extra_categories:
                categories[category_name].features.append(_Feature(node_map=self, name=node_name))

        features: list[Any] = [categories[category_name] for category_name in _CATEGORY_ORDER]
        features.extend(_HostileFeature(mode=mode) for mode in self._hostile_modes)

        return _Category(
            node=_RawNode(
                name="Root",
                principal_interface_type=_CATEGORY,
                description="",
                access_mode=_READ_ONLY,
                node_map=self,
            ),
            features=features,
        )


class _Feature:
    """Exposes the value half of a synthetic node, mirroring the feature accessor of a GenICam node map."""

    def __init__(self, node_map: "SyntheticNodeMap", name: str) -> None:
        self._node_map = node_map
        self._name = name

    @property
    def node(self) -> _RawNode:
        """Returns the descriptor of the node."""
        specification = _NODE_SPECIFICATIONS[self._name]
        return _RawNode(
            name=self._name,
            principal_interface_type=specification.type_code,
            description=specification.description,
            access_mode=self._node_map.access_mode(name=self._name),
            node_map=self._node_map,
        )

    @property
    def value(self) -> Any:
        """Returns the current value of the node."""
        return self._node_map.read(name=self._name)

    @value.setter
    def value(self, new_value: Any) -> None:
        self._node_map.write(name=self._name, value=new_value)

    @property
    def unit(self) -> str:
        """Returns the measurement unit of the node, or an empty string when the node defines none."""
        return _NODE_SPECIFICATIONS[self._name].unit

    @property
    def min(self) -> int | float:
        """Returns the lower bound the node currently accepts."""
        return self._node_map.bounds(name=self._name)[0]

    @property
    def max(self) -> int | float:
        """Returns the upper bound the node currently accepts."""
        return self._node_map.bounds(name=self._name)[1]

    @property
    def inc(self) -> int:
        """Returns the step increment of the node."""
        return _NODE_SPECIFICATIONS[self._name].increment

    @property
    def entries(self) -> list[_EnumerationEntry]:
        """Returns the symbolic entries of an Enumeration node."""
        return [
            _EnumerationEntry(
                node=_RawNode(
                    name=entry,
                    principal_interface_type=_ENUMERATION,
                    description="",
                    access_mode=_READ_WRITE,
                    node_map=self._node_map,
                ),
                symbolic=entry,
            )
            for entry in _NODE_SPECIFICATIONS[self._name].entries
        ]
