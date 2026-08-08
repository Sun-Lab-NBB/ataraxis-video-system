"""Contains tests for classes and functions provided by the configuration.py module."""

import pytest
from synthetic_node_map import SENSOR_WIDTH, SyntheticNodeMap, SyntheticNodeMapError
from ataraxis_base_utilities import error_format

from ataraxis_video_system import GenicamNodeInfo, GenicamConfiguration
from ataraxis_video_system.video.camera import HarvestersCamera
from ataraxis_video_system.video.configuration import (
    read_genicam_node,
    read_genicam_nodes,
    write_genicam_node,
    format_genicam_node,
    enumerate_genicam_nodes,
    apply_genicam_configuration,
)

_SIMULATED_MODEL: str = "TLSimuMono"
"""Stores the model name the bundled GenTL Producer simulator reports for its monochrome devices."""

_SIMULATED_SERIAL: str = "SN_InterfaceA_0"
"""Stores the serial number the bundled GenTL Producer simulator reports for the first monochrome device."""

_SIMULATED_WRITABLE_NODES: int = 10
"""Stores the number of writable leaf nodes the bundled GenTL Producer simulator exposes."""

_SYNTHETIC_IDENTITY: dict[str, str] = {"current_model": "SyntheticCamera", "current_serial": "SN_SYNTHETIC"}
"""Stores the identity arguments that match the configurations the synthetic node map tests build."""


def _synthetic_configuration(nodes: dict[str, object]) -> GenicamConfiguration:
    """Builds a GenicamConfiguration carrying the synthetic camera identity and the supplied node values."""
    return GenicamConfiguration(
        camera_model=_SYNTHETIC_IDENTITY["current_model"],
        camera_serial_number=_SYNTHETIC_IDENTITY["current_serial"],
        nodes=[GenicamNodeInfo(name=name, value=value) for name, value in nodes.items()],
    )


def _write_nodes_in_order(node_map: SyntheticNodeMap, values: dict[str, object], names: list[str]) -> None:
    """Writes the named nodes to the node map in the order the caller supplies."""
    for name in names:
        getattr(node_map, name).value = values[name]


def test_genicam_node_info_creation() -> None:
    """Verifies creation of GenicamNodeInfo instances."""
    node = GenicamNodeInfo(name="Width", value=200)
    assert node.name == "Width"
    assert node.value == 200

    node_float = GenicamNodeInfo(name="Gain", value=1.5)
    assert node_float.name == "Gain"
    assert node_float.value == 1.5


def test_genicam_node_info_types() -> None:
    """Verifies that GenicamNodeInfo correctly stores all supported value types."""
    assert GenicamNodeInfo(name="IntNode", value=42).value == 42
    assert GenicamNodeInfo(name="FloatNode", value=3.14).value == 3.14
    assert GenicamNodeInfo(name="BoolNode", value=True).value
    assert GenicamNodeInfo(name="StrNode", value="Mono8").value == "Mono8"


def test_genicam_configuration_yaml_roundtrip(tmp_path) -> None:
    """Verifies GenicamConfiguration serialization and deserialization via YAML."""
    nodes = [
        GenicamNodeInfo(name="Width", value=200),
        GenicamNodeInfo(name="Height", value=200),
        GenicamNodeInfo(name="Gain", value=2.5),
        GenicamNodeInfo(name="ReverseX", value=False),
        GenicamNodeInfo(name="PixelFormat", value="Mono8"),
    ]
    config = GenicamConfiguration(
        camera_model="TestCamera",
        camera_serial_number="SN12345",
        nodes=nodes,
    )

    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(file_path=yaml_path)

    loaded = GenicamConfiguration.from_yaml(file_path=yaml_path)
    assert loaded.camera_model == "TestCamera"
    assert loaded.camera_serial_number == "SN12345"
    assert len(loaded.nodes) == 5
    assert loaded.nodes[0].name == "Width"
    assert loaded.nodes[0].value == 200
    assert loaded.nodes[1].name == "Height"
    assert loaded.nodes[1].value == 200
    assert loaded.nodes[2].value == 2.5
    assert not loaded.nodes[3].value
    assert loaded.nodes[4].value == "Mono8"


def test_genicam_configuration_empty_yaml_roundtrip(tmp_path) -> None:
    """Verifies YAML roundtrip for a GenicamConfiguration with no nodes."""
    config = GenicamConfiguration(
        camera_model="EmptyCamera",
        camera_serial_number="SN00000",
        nodes=[],
    )

    yaml_path = tmp_path / "empty_config.yaml"
    config.to_yaml(file_path=yaml_path)

    loaded = GenicamConfiguration.from_yaml(file_path=yaml_path)
    assert loaded.camera_model == "EmptyCamera"
    assert loaded.camera_serial_number == "SN00000"
    assert loaded.nodes == []


def test_node_map_error_not_connected() -> None:
    """Verifies that accessing node_map on a disconnected HarvestersCamera raises ConnectionError."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    message = (
        f"Unable to access the node map for VideoSystem with id {camera._system_id}. The camera is not "
        f"connected. Call the connect() method first."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.node_map


def test_model_serial_not_connected() -> None:
    """Verifies that model and serial_number return empty strings when the camera is not connected."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    assert camera.model == ""
    assert camera.serial_number == ""


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_model_serial_connected() -> None:
    """Verifies that model and serial_number report the identity of the connected device."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        assert camera.model == _SIMULATED_MODEL
        assert camera.serial_number == _SIMULATED_SERIAL
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_get_node_info() -> None:
    """Verifies that get_node_info returns a GenicamNodeInfo for a standard GenICam node."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        info = camera.get_node_info(name="Width")
        assert isinstance(info, GenicamNodeInfo)
        assert info.name == "Width"
        assert isinstance(info.value, int)
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_get_node_description() -> None:
    """Verifies that get_node_description returns a formatted multi-line string."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        description = camera.get_node_description(name="Width")
        assert isinstance(description, str)
        assert "Node: Width" in description
        assert "Type: INTEGER" in description
        assert "Value:" in description
        assert "Access: READ_WRITE" in description
        assert "Min:" in description
        assert "Max:" in description
        assert "Increment:" in description
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_set_node_value() -> None:
    """Verifies that set_node_value writes a value to a writable GenICam node."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        camera.set_node_value(name="Width", value="256")
        assert camera.get_node_info(name="Width").value == 256
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_get_configuration() -> None:
    """Verifies that get_configuration returns a GenicamConfiguration with populated nodes."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        assert isinstance(config, GenicamConfiguration)
        assert config.camera_model == _SIMULATED_MODEL
        assert config.camera_serial_number == _SIMULATED_SERIAL
        assert len(config.nodes) == _SIMULATED_WRITABLE_NODES
        assert all(isinstance(node, GenicamNodeInfo) for node in config.nodes)
        assert {node.name for node in config.nodes} >= {"Width", "Height", "PixelFormat"}
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_apply_configuration() -> None:
    """Verifies that apply_configuration restores every node value captured by get_configuration."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        original_width = camera.get_node_info(name="Width").value

        # Moves the camera away from the captured state so that the restore has something to undo.
        camera.set_node_value(name="Width", value="128")
        assert camera.get_node_info(name="Width").value == 128

        camera.apply_configuration(config=config, strict_identity=True)
        assert camera.get_node_info(name="Width").value == original_width
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_apply_configuration_strict_mismatch() -> None:
    """Verifies that apply_configuration raises ValueError on identity mismatch in strict mode."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        # Overwrites the model to simulate a mismatch.
        config.camera_model = "WrongModel"
        with pytest.raises(ValueError, match="Camera identity mismatch"):
            camera.apply_configuration(config=config, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_apply_configuration_missing_node() -> None:
    """Verifies that apply_configuration raises ValueError when a node does not exist on the camera."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # Creates a configuration with matching identity but a non-existent node.
        config = GenicamConfiguration(
            camera_model=camera.model,
            camera_serial_number=camera.serial_number,
            nodes=[GenicamNodeInfo(name="NonExistentFakeNode12345", value=42)],
        )
        with pytest.raises(ValueError, match="does not exist"):
            camera.apply_configuration(config=config, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_configuration_yaml_roundtrip(tmp_path) -> None:
    """Verifies that a live camera configuration can be serialized to YAML and deserialized back."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        yaml_path = tmp_path / "camera_config.yaml"
        config.to_yaml(file_path=yaml_path)

        loaded = GenicamConfiguration.from_yaml(file_path=yaml_path)
        assert loaded.camera_model == config.camera_model
        assert loaded.camera_serial_number == config.camera_serial_number
        assert len(loaded.nodes) == len(config.nodes)

        # A configuration that survived a YAML roundtrip still applies to the camera it came from.
        camera.apply_configuration(config=loaded, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_format_genicam_node_enumeration() -> None:
    """Verifies that format_genicam_node includes entry names for Enumeration nodes."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        description = format_genicam_node(node_map=camera.node_map, name="PixelFormat")
        assert "Node: PixelFormat" in description
        assert "Type: ENUMERATION" in description
        assert "Entries:" in description
        assert "Mono8" in description
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_write_genicam_node_boolean() -> None:
    """Verifies that write_genicam_node correctly coerces string values to bool for Boolean nodes."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        node_map = camera.node_map
        write_genicam_node(node_map=node_map, name="ChunkModeActive", value="true")
        assert read_genicam_node(node_map=node_map, name="ChunkModeActive").value

        write_genicam_node(node_map=node_map, name="ChunkModeActive", value="false")
        assert not read_genicam_node(node_map=node_map, name="ChunkModeActive").value
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_write_genicam_node_enum_string() -> None:
    """Verifies that write_genicam_node correctly handles string values for Enumeration nodes."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        node_map = camera.node_map
        write_genicam_node(node_map=node_map, name="PixelFormat", value="Mono12")
        assert read_genicam_node(node_map=node_map, name="PixelFormat").value == "Mono12"

        write_genicam_node(node_map=node_map, name="PixelFormat", value="Mono8")
        assert read_genicam_node(node_map=node_map, name="PixelFormat").value == "Mono8"
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_apply_configuration_non_strict_mismatch() -> None:
    """Verifies that apply_configuration warns but proceeds when identity mismatches in non-strict mode."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        # Overwrites both model and serial number to trigger the mismatch warning.
        config.camera_model = "WrongModel"
        config.camera_serial_number = "WrongSerial"
        # Non-strict mode should warn but not raise.
        camera.apply_configuration(config=config, strict_identity=False)
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_apply_configuration_blacklisted_nodes() -> None:
    """Verifies that apply_configuration skips blacklisted nodes during application."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        camera.set_node_value(name="Width", value="128")

        # Blacklisting Width excludes it from both validation and the write pass, so it keeps its current value.
        camera.apply_configuration(config=config, strict_identity=True, blacklisted_nodes=frozenset({"Width"}))
        assert camera.get_node_info(name="Width").value == 128
    finally:
        camera.disconnect()


def test_enumerate_genicam_nodes_synthetic() -> None:
    """Verifies that node enumeration collects writable value nodes and skips locked and ReadOnly ones."""
    node_map = SyntheticNodeMap()
    names = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=frozenset())

    # DeviceModelName is permanently ReadOnly, so it never appears among the writable nodes.
    assert "DeviceModelName" not in names
    assert names == sorted(names)
    assert {"Width", "Height", "OffsetX", "OffsetY", "Gain", "GainAuto"} <= set(names)

    # Engaging an auto-control locks its manual counterpart, which drops out of the writable set.
    locked_map = SyntheticNodeMap(overrides={"GainAuto": "Continuous", "ExposureAuto": "Continuous"})
    locked_names = enumerate_genicam_nodes(node_map=locked_map, blacklisted_nodes=frozenset())
    assert "Gain" not in locked_names
    assert "ExposureTime" not in locked_names

    # Blacklisted nodes are excluded regardless of their access mode.
    filtered = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=frozenset({"Width"}))
    assert "Width" not in filtered


def test_format_genicam_node_synthetic_unit_and_range() -> None:
    """Verifies that format_genicam_node reports the measurement unit, numeric range, and type-specific fields."""
    node_map = SyntheticNodeMap()

    integer_description = format_genicam_node(node_map=node_map, name="Width")
    assert "Node: Width" in integer_description
    assert "Type: INTEGER" in integer_description
    assert "Description: Frame width." in integer_description
    assert f"Max: {SENSOR_WIDTH}" in integer_description
    assert "Increment: 4" in integer_description
    assert "Unit: px" in integer_description

    float_description = format_genicam_node(node_map=node_map, name="ExposureTime")
    assert "Type: FLOAT" in float_description
    assert "Min:" in float_description
    assert "Unit: us" in float_description
    # Increment is an Integer-only field, so it is absent from a Float node description.
    assert "Increment:" not in float_description

    # A node that declares no unit omits the field entirely rather than reporting an empty one.
    assert "Unit:" not in format_genicam_node(node_map=node_map, name="ReverseX")


def test_write_genicam_node_synthetic_float() -> None:
    """Verifies that write_genicam_node coerces string values to float for Float nodes."""
    node_map = SyntheticNodeMap()
    write_genicam_node(node_map=node_map, name="ExposureTime", value="12500.5")
    assert node_map.values["ExposureTime"] == 12500.5
    assert isinstance(node_map.values["ExposureTime"], float)


def test_apply_configuration_phase_ordering() -> None:
    """Verifies that apply_configuration satisfies the SFNC dimension and offset dependency chain."""
    target = {
        "BinningHorizontal": 2,
        "PixelFormat": "Mono12",
        "ReverseX": True,
        "Width": 900,
        "Height": 500,
        "OffsetX": 60,
        "OffsetY": 100,
        "ExposureAuto": "Off",
        "ExposureTime": 20000.0,
        "AcquisitionFrameRate": 40.0,
        "GainAuto": "Off",
        "Gain": 12.5,
    }

    # Starts from a full-sensor, unbinned state, so every offset and dimension constraint has to be renegotiated.
    node_map = SyntheticNodeMap(
        overrides={
            "BinningHorizontal": 1,
            "Width": SENSOR_WIDTH,
            "OffsetX": 0,
            "ExposureTime": 5000.0,
            "AcquisitionFrameRate": 100.0,
        }
    )

    apply_genicam_configuration(
        node_map=node_map,
        config=_synthetic_configuration(nodes=target),
        strict=True,
        blacklisted_nodes=frozenset(),
        **_SYNTHETIC_IDENTITY,
    )

    for name, value in target.items():
        assert node_map.values[name] == value, f"node '{name}' was not restored to its target value"

    # A node covered by both a reset phase and a later restore phase is written exactly twice, once with its reset
    # value and once with its target value, rather than being deferred to the pass that handles unphased nodes.
    writes = node_map.writes
    assert writes.count("OffsetX") == 2
    assert writes.count("OffsetY") == 2
    assert writes.count("GainAuto") == 2

    # The offsets take their target value in the phase that follows the dimensions and precedes the timing phase.
    last_offset_write = len(writes) - 1 - writes[::-1].index("OffsetX")
    assert writes.index("Width") < last_offset_write < writes.index("ExposureTime")


def test_apply_configuration_naive_order_fails() -> None:
    """Verifies that the phase ordering is load-bearing by writing the same nodes in an unordered sequence."""
    target = {
        "AcquisitionFrameRate": 40.0,
        "BinningHorizontal": 2,
        "ExposureTime": 20000.0,
        "Height": 500,
        "OffsetX": 60,
        "OffsetY": 100,
        "Width": 900,
    }

    node_map = SyntheticNodeMap(
        overrides={"BinningHorizontal": 1, "Width": SENSOR_WIDTH, "OffsetX": 0, "ExposureTime": 5000.0}
    )

    # Writing the same values in sorted order violates the offset constraint, which is what the phases exist to avoid.
    with pytest.raises(SyntheticNodeMapError):
        _write_nodes_in_order(node_map=node_map, values=target, names=sorted(target))


def test_apply_configuration_resets_offsets_absent_from_configuration() -> None:
    """Verifies that the reset phase zeroes offsets the configuration omits so that dimensions gain their full range."""
    # The device holds a large horizontal offset, which caps Width well below the target until the offset is zeroed.
    node_map = SyntheticNodeMap(overrides={"BinningHorizontal": 1, "OffsetX": 1000, "Width": 900})
    assert node_map.OffsetX.value == 1000
    assert node_map.Width.max == SENSOR_WIDTH - 1000

    apply_genicam_configuration(
        node_map=node_map,
        config=_synthetic_configuration(nodes={"Width": 1800}),
        strict=True,
        blacklisted_nodes=frozenset(),
        **_SYNTHETIC_IDENTITY,
    )

    assert node_map.values["Width"] == 1800
    assert node_map.values["OffsetX"] == 0


def test_apply_configuration_unlocks_auto_controls() -> None:
    """Verifies that apply_configuration writes a manual node the device holds ReadOnly under an auto-control."""
    node_map = SyntheticNodeMap(overrides={"ExposureAuto": "Continuous", "ExposureTime": 5000.0})

    # The manual node stays locked until the unlock phase disengages the auto-control that owns it.
    with pytest.raises(SyntheticNodeMapError):
        node_map.ExposureTime.value = 20000.0

    apply_genicam_configuration(
        node_map=node_map,
        config=_synthetic_configuration(nodes={"ExposureAuto": "Continuous", "ExposureTime": 20000.0}),
        strict=True,
        blacklisted_nodes=frozenset(),
        **_SYNTHETIC_IDENTITY,
    )

    assert node_map.values["ExposureTime"] == 20000.0
    assert node_map.values["ExposureAuto"] == "Continuous"


def test_apply_configuration_writes_manual_analog_before_relock() -> None:
    """Verifies that a manual analog value is written before the auto-control that owns it is re-engaged."""
    node_map = SyntheticNodeMap(overrides={"GainAuto": "Off", "Gain": 0.0})

    # Enumeration emits node names in sorted order, so a dumped configuration presents Gain ahead of GainAuto.
    apply_genicam_configuration(
        node_map=node_map,
        config=_synthetic_configuration(nodes={"Gain": 12.5, "GainAuto": "Continuous"}),
        strict=True,
        blacklisted_nodes=frozenset(),
        **_SYNTHETIC_IDENTITY,
    )

    assert node_map.values["Gain"] == 12.5
    assert node_map.values["GainAuto"] == "Continuous"

    # The manual value lands in its own phase rather than in the trailing pass that runs after the re-lock phase.
    writes = node_map.writes
    assert writes.index("Gain") < len(writes) - 1 - writes[::-1].index("GainAuto")


def test_read_genicam_nodes_expands_selectors() -> None:
    """Verifies that reading a configuration captures every instance of a selector-addressed node."""
    node_map = SyntheticNodeMap()
    nodes = read_genicam_nodes(node_map=node_map, blacklisted_nodes=frozenset())

    ratios = {node.selectors["BalanceRatioSelector"]: node.value for node in nodes if node.name == "BalanceRatio"}
    assert ratios == {"Red": 1.0, "Green": 2.0, "Blue": 3.0}

    # The selector itself is recorded at the position the camera was found in, not at the position stepping the
    # instances left it in.
    selector_entries = [node for node in nodes if node.name == "BalanceRatioSelector"]
    assert len(selector_entries) == 1
    assert selector_entries[0].value == "Red"
    assert selector_entries[0].selectors == {}

    # Reading the configuration leaves the selector exactly where it was found.
    assert node_map.values["BalanceRatioSelector"] == "Red"

    # A node no selector addresses contributes a single entry carrying no selector context.
    width_entries = [node for node in nodes if node.name == "Width"]
    assert len(width_entries) == 1
    assert width_entries[0].selectors == {}


def test_apply_configuration_restores_every_selected_instance() -> None:
    """Verifies that applying a configuration writes each selector-addressed instance to its own captured value."""
    node_map = SyntheticNodeMap()
    config = GenicamConfiguration(
        camera_model=_SYNTHETIC_IDENTITY["current_model"],
        camera_serial_number=_SYNTHETIC_IDENTITY["current_serial"],
        nodes=read_genicam_nodes(node_map=node_map, blacklisted_nodes=frozenset()),
    )

    # Collapses every channel onto one value, which is the state a selector-blind restore would leave behind.
    for channel in ("Red", "Green", "Blue"):
        node_map.values["BalanceRatioSelector"] = channel
        node_map.write(name="BalanceRatio", value=7.0)
    node_map.values["BalanceRatioSelector"] = "Green"

    apply_genicam_configuration(
        node_map=node_map,
        config=config,
        strict=True,
        blacklisted_nodes=frozenset(),
        **_SYNTHETIC_IDENTITY,
    )

    restored = {key[1][0][1]: value for key, value in node_map.selected_values.items()}
    assert restored == {"Red": 1.0, "Green": 2.0, "Blue": 3.0}
    assert node_map.values["BalanceRatioSelector"] == "Red"


def test_apply_configuration_rejects_permanently_read_only_node() -> None:
    """Verifies that apply_configuration rejects a node the device reports as ReadOnly after the unlock phase."""
    node_map = SyntheticNodeMap()

    # DeviceModelName is ReadOnly regardless of any other node, so no phase can make it writable.
    with pytest.raises(ValueError, match="must have ReadWrite access"):
        apply_genicam_configuration(
            node_map=node_map,
            config=_synthetic_configuration(nodes={"DeviceModelName": "Renamed"}),
            strict=True,
            blacklisted_nodes=frozenset(),
            **_SYNTHETIC_IDENTITY,
        )

    assert node_map.values["DeviceModelName"] == "SyntheticCamera"


def test_apply_configuration_write_failure() -> None:
    """Verifies that apply_configuration raises RuntimeError when a target node write is rejected."""
    node_map = SyntheticNodeMap()

    # The requested width exceeds the addressable sensor area, so the device rejects the write.
    with pytest.raises(RuntimeError, match="Failed to write node"):
        apply_genicam_configuration(
            node_map=node_map,
            config=_synthetic_configuration(nodes={"Width": SENSOR_WIDTH * 2}),
            strict=True,
            blacklisted_nodes=frozenset(),
            **_SYNTHETIC_IDENTITY,
        )


@pytest.mark.xdist_group(name="group2")
def test_format_genicam_node_with_unit_hardware(has_harvesters) -> None:
    """Verifies that format_genicam_node reports the range of a real SFNC Float node."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GenICam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # ExposureTime is a standard SFNC Float node with a unit, which the bundled simulator does not implement.
        description = format_genicam_node(node_map=camera.node_map, name="ExposureTime")
        assert "Node: ExposureTime" in description
        assert "Min:" in description
        assert "Max:" in description
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_write_genicam_node_float_hardware(has_harvesters) -> None:
    """Verifies that write_genicam_node coerces string values to float against real hardware."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GenICam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # Reads the current ExposureTime value and writes it back unchanged.
        original = read_genicam_node(node_map=camera.node_map, name="ExposureTime")
        write_genicam_node(node_map=camera.node_map, name="ExposureTime", value=str(original.value))
        restored = read_genicam_node(node_map=camera.node_map, name="ExposureTime")
        assert restored.value == original.value
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_configuration_roundtrip_hardware(has_harvesters, tmp_path) -> None:
    """Verifies the full dump, serialize, reload, and apply cycle against real GenICam hardware."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GenICam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # Real hardware carries the interdependent SFNC nodes, such as offsets and auto-controls, that drive the
        # phase ordering inside apply_genicam_configuration.
        config = camera.get_configuration()
        assert config.nodes

        yaml_path = tmp_path / "hardware_config.yaml"
        config.to_yaml(file_path=yaml_path)
        loaded = GenicamConfiguration.from_yaml(file_path=yaml_path)

        camera.apply_configuration(config=loaded, strict_identity=True)
    finally:
        camera.disconnect()
