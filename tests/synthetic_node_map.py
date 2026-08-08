"""Provides a synthetic GenICam node map that reproduces the SFNC feature interdependencies of real hardware.

The bundled GenTL Producer simulator exposes ten mutually independent writable nodes, so it cannot exercise the
dependency chains that ``apply_genicam_configuration`` orders its write phases around. This module models those
chains directly: binning rescales the addressable sensor area, offsets and dimensions compete for that area, the
auto-controls lock their manual counterparts, and exposure bounds the attainable frame rate.
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

_READ_ONLY: int = 3
"""Stores the GenICam access mode code for ReadOnly nodes."""

_READ_WRITE: int = 4
"""Stores the GenICam access mode code for ReadWrite nodes."""

SENSOR_WIDTH: int = 1920
"""Stores the full addressable sensor width of the synthetic camera, in pixels."""

SENSOR_HEIGHT: int = 1080
"""Stores the full addressable sensor height of the synthetic camera, in pixels."""


@dataclass(slots=True)
class _NodeSpec:
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


_NODE_SPECS: dict[str, _NodeSpec] = {
    "DeviceModelName": _NodeSpec(type_code=_STRING, category="DeviceControl", read_only=True),
    "BinningHorizontal": _NodeSpec(
        type_code=_INTEGER, category="ImageFormatControl", description="Horizontal binning factor."
    ),
    "PixelFormat": _NodeSpec(
        type_code=_ENUMERATION, category="ImageFormatControl", entries=("Mono8", "Mono12", "BGR8")
    ),
    "ReverseX": _NodeSpec(type_code=_BOOLEAN, category="ImageFormatControl"),
    "Width": _NodeSpec(
        type_code=_INTEGER, category="ImageFormatControl", description="Frame width.", unit="px", increment=4
    ),
    "Height": _NodeSpec(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    "OffsetX": _NodeSpec(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    "OffsetY": _NodeSpec(type_code=_INTEGER, category="ImageFormatControl", increment=4),
    "ExposureAuto": _NodeSpec(
        type_code=_ENUMERATION, category="AcquisitionControl", entries=("Off", "Once", "Continuous")
    ),
    "ExposureTime": _NodeSpec(
        type_code=_FLOAT, category="AcquisitionControl", description="Exposure duration.", unit="us"
    ),
    "AcquisitionFrameRate": _NodeSpec(type_code=_FLOAT, category="AcquisitionControl", unit="Hz"),
    "GainAuto": _NodeSpec(type_code=_ENUMERATION, category="AnalogControl", entries=("Off", "Once", "Continuous")),
    "Gain": _NodeSpec(type_code=_FLOAT, category="AnalogControl", unit="dB"),
}
"""Describes every node the synthetic camera implements, keyed by feature name."""

_CATEGORY_ORDER: tuple[str, ...] = ("DeviceControl", "ImageFormatControl", "AcquisitionControl", "AnalogControl")
"""Orders the category nodes as they appear under the synthetic root node."""

_DEFAULT_VALUES: dict[str, Any] = {
    "DeviceModelName": "SyntheticCamera",
    "BinningHorizontal": 1,
    "PixelFormat": "Mono8",
    "ReverseX": False,
    "Width": SENSOR_WIDTH,
    "Height": SENSOR_HEIGHT,
    "OffsetX": 0,
    "OffsetY": 0,
    "ExposureAuto": "Off",
    "ExposureTime": 5000.0,
    "AcquisitionFrameRate": 30.0,
    "GainAuto": "Off",
    "Gain": 0.0,
}
"""Stores the value each node holds before a test applies its own starting state."""


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

    def get_access_mode(self) -> int:
        """Returns the access mode code the node currently reports."""
        return self.access_mode


@dataclass(slots=True)
class _EnumEntry:
    """Exposes a single symbolic entry of a synthetic Enumeration node."""

    node: _RawNode
    """The descriptor of the entry."""
    symbolic: str
    """The symbolic name of the entry."""


class _Feature:
    """Exposes the value half of a synthetic node, mirroring the feature accessor of a GenICam node map."""

    def __init__(self, node_map: "SyntheticNodeMap", name: str) -> None:
        self._node_map = node_map
        self._name = name

    @property
    def node(self) -> _RawNode:
        """Returns the descriptor of the node."""
        specification = _NODE_SPECS[self._name]
        return _RawNode(
            name=self._name,
            principal_interface_type=specification.type_code,
            description=specification.description,
            access_mode=self._node_map.access_mode(name=self._name),
        )

    @property
    def value(self) -> Any:
        """Returns the current value of the node."""
        return self._node_map.values[self._name]

    @value.setter
    def value(self, new_value: Any) -> None:
        self._node_map.write(name=self._name, value=new_value)

    @property
    def unit(self) -> str:
        """Returns the measurement unit of the node, or an empty string when the node defines none."""
        return _NODE_SPECS[self._name].unit

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
        return _NODE_SPECS[self._name].increment

    @property
    def entries(self) -> list[_EnumEntry]:
        """Returns the symbolic entries of an Enumeration node."""
        return [
            _EnumEntry(
                node=_RawNode(
                    name=entry,
                    principal_interface_type=_ENUMERATION,
                    description="",
                    access_mode=_READ_WRITE,
                ),
                symbolic=entry,
            )
            for entry in _NODE_SPECS[self._name].entries
        ]


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

    Attributes:
        values: The current value of every node, keyed by feature name.
        writes: The feature name of every successful write, in the order the writes occurred.
    """

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.values: dict[str, Any] = dict(_DEFAULT_VALUES)
        self.writes: list[str] = []

        # Seeds the starting state without constraint checking so that a test can describe any camera state directly.
        if overrides is not None:
            self.values.update(overrides)

    def __getattr__(self, name: str) -> Any:
        """Resolves a feature or category node by name, mirroring GenICam node map attribute access."""
        if name == "Root":
            return self._build_root()
        if name in _NODE_SPECS:
            return _Feature(node_map=self, name=name)
        message = f"The synthetic node map does not implement the node '{name}'."
        raise AttributeError(message)

    def access_mode(self, name: str) -> int:
        """Returns the access mode code the named node currently reports.

        Notes:
            The manual counterpart of an engaged auto-control reports ReadOnly, which is how a camera prevents the
            manual value from being written while the auto-control owns it.
        """
        if _NODE_SPECS[name].read_only:
            return _READ_ONLY
        if name == "Gain" and self.values["GainAuto"] != "Off":
            return _READ_ONLY
        if name == "ExposureTime" and self.values["ExposureAuto"] != "Off":
            return _READ_ONLY
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
            return 8, SENSOR_HEIGHT - self.values["OffsetY"]
        if name == "OffsetY":
            return 0, SENSOR_HEIGHT - self.values["Height"]
        if name == "BinningHorizontal":
            return 1, 4
        if name == "ExposureTime":
            return 10.0, 1000000.0
        if name == "AcquisitionFrameRate":
            # Exposure bounds the attainable rate, which is why exposure is written first within the timing phase.
            return 1.0, 1000000.0 / self.values["ExposureTime"]
        if name == "Gain":
            return 0.0, 48.0
        return 0, 0

    def write(self, name: str, value: Any) -> None:
        """Writes a value to the named node after enforcing every constraint the node participates in.

        Raises:
            SyntheticNodeMapError: If the node is currently ReadOnly, the value falls outside the bounds the node
                accepts, or the value is not a valid entry of an Enumeration node.
        """
        if self.access_mode(name=name) != _READ_WRITE:
            message = f"The node '{name}' is ReadOnly while its auto-control is engaged."
            raise SyntheticNodeMapError(message)

        specification = _NODE_SPECS[name]
        if specification.type_code == _ENUMERATION and value not in specification.entries:
            message = f"The value '{value}' is not a valid entry of the node '{name}'."
            raise SyntheticNodeMapError(message)

        if specification.type_code in (_INTEGER, _FLOAT):
            minimum, maximum = self.bounds(name=name)
            if not minimum <= value <= maximum:
                message = f"The value '{value}' falls outside the range [{minimum}, {maximum}] of the node '{name}'."
                raise SyntheticNodeMapError(message)

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
                )
            )

        for node_name, specification in _NODE_SPECS.items():
            categories[specification.category].features.append(_Feature(node_map=self, name=node_name))

        return _Category(
            node=_RawNode(
                name="Root",
                principal_interface_type=_CATEGORY,
                description="",
                access_mode=_READ_ONLY,
            ),
            features=[categories[category_name] for category_name in _CATEGORY_ORDER],
        )
