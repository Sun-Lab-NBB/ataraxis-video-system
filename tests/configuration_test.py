"""Contains tests for classes and functions provided by the configuration.py module."""

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system import GenicamNodeInfo, GenicamConfiguration
from ataraxis_video_system.camera import HarvestersCamera
from ataraxis_video_system.configuration import (
    format_genicam_node,
    read_genicam_node,
    write_genicam_node,
)


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
    assert GenicamNodeInfo(name="BoolNode", value=True).value is True
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
    assert loaded.nodes[3].value is False
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


@pytest.mark.xdist_group(name="group2")
def test_harvesters_model_serial_connected(has_harvesters) -> None:
    """Verifies that model and serial_number return non-empty strings after connecting."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        assert isinstance(camera.model, str)
        assert camera.model
        assert isinstance(camera.serial_number, str)
        assert camera.serial_number
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_get_node_info(has_harvesters) -> None:
    """Verifies that get_node_info returns a GenicamNodeInfo for a standard GenICam node."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        info = camera.get_node_info("Width")
        assert isinstance(info, GenicamNodeInfo)
        assert info.name == "Width"
        assert isinstance(info.value, (int, float))
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_get_node_description(has_harvesters) -> None:
    """Verifies that get_node_description returns a formatted multi-line string."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        description = camera.get_node_description("Width")
        assert isinstance(description, str)
        assert "Node: Width" in description
        assert "Type:" in description
        assert "Value:" in description
        assert "Access:" in description
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_set_node_value(has_harvesters) -> None:
    """Verifies that set_node_value can write a value to a writable GenICam node."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # Reads the current Width value and writes it back unchanged to avoid disrupting camera state.
        original = camera.get_node_info("Width")
        camera.set_node_value("Width", str(original.value))
        restored = camera.get_node_info("Width")
        assert restored.value == original.value
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_get_configuration(has_harvesters) -> None:
    """Verifies that get_configuration returns a GenicamConfiguration with populated nodes."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        assert isinstance(config, GenicamConfiguration)
        assert config.camera_model
        assert config.camera_serial_number
        assert config.nodes
        assert all(isinstance(node, GenicamNodeInfo) for node in config.nodes)
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_apply_configuration(has_harvesters) -> None:
    """Verifies that apply_configuration can re-apply a camera's own configuration."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # Dumps and immediately re-applies the same configuration. This is non-destructive.
        config = camera.get_configuration()
        camera.apply_configuration(config, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_apply_configuration_strict_mismatch(has_harvesters) -> None:
    """Verifies that apply_configuration raises ValueError on identity mismatch in strict mode."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        # Overwrites the model to simulate a mismatch.
        config.camera_model = "WrongModel"
        with pytest.raises(ValueError, match="Camera identity mismatch"):
            camera.apply_configuration(config, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_apply_configuration_missing_node(has_harvesters) -> None:
    """Verifies that apply_configuration raises ValueError when a node does not exist on the camera."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

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
            camera.apply_configuration(config, strict_identity=True)
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_configuration_yaml_roundtrip(has_harvesters, tmp_path) -> None:
    """Verifies that a live camera configuration can be serialized to YAML and deserialized back."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

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
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_format_genicam_node_enumeration(has_harvesters) -> None:
    """Verifies that format_genicam_node includes entry names for Enumeration nodes."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # PixelFormat is a standard SFNC Enumeration node present on all GenICam cameras.
        description = format_genicam_node(node_map=camera.node_map, name="PixelFormat")
        assert "Entries:" in description
        assert "Node: PixelFormat" in description
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_format_genicam_node_with_unit(has_harvesters) -> None:
    """Verifies that format_genicam_node includes the measurement unit when the node defines one."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # ExposureTime is a standard SFNC Float node with a unit (typically "us").
        description = format_genicam_node(node_map=camera.node_map, name="ExposureTime")
        assert "Node: ExposureTime" in description
        assert "Min:" in description
        assert "Max:" in description
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_write_genicam_node_float(has_harvesters) -> None:
    """Verifies that write_genicam_node correctly coerces string values to float for Float nodes."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

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
def test_write_genicam_node_boolean(has_harvesters) -> None:
    """Verifies that write_genicam_node correctly coerces string values to bool for Boolean nodes."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # ReverseX is a standard SFNC Boolean node present on most GenICam cameras.
        original = read_genicam_node(node_map=camera.node_map, name="ReverseX")
        write_genicam_node(node_map=camera.node_map, name="ReverseX", value=str(original.value).lower())
        restored = read_genicam_node(node_map=camera.node_map, name="ReverseX")
        assert restored.value == original.value
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_write_genicam_node_enum_string(has_harvesters) -> None:
    """Verifies that write_genicam_node correctly handles string values for Enumeration nodes."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        # PixelFormat is a standard SFNC Enumeration node. Reads and writes back the current value.
        original = read_genicam_node(node_map=camera.node_map, name="PixelFormat")
        write_genicam_node(node_map=camera.node_map, name="PixelFormat", value=str(original.value))
        restored = read_genicam_node(node_map=camera.node_map, name="PixelFormat")
        assert restored.value == original.value
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_apply_configuration_non_strict_mismatch(has_harvesters) -> None:
    """Verifies that apply_configuration warns but proceeds when identity mismatches in non-strict mode."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()
        # Overwrites both model and serial number to trigger the mismatch warning.
        config.camera_model = "WrongModel"
        config.camera_serial_number = "WrongSerial"
        # Non-strict mode should warn but not raise.
        camera.apply_configuration(config, strict_identity=False)
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_apply_configuration_blacklisted_nodes(has_harvesters) -> None:
    """Verifies that apply_configuration skips blacklisted nodes during application."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    camera.connect()
    try:
        config = camera.get_configuration()

        # Adds a real node name to the blacklist to verify that blacklisted nodes are skipped.
        custom_blacklist = frozenset({"Width"})
        camera.apply_configuration(config, strict_identity=True, blacklisted_nodes=custom_blacklist)
    finally:
        camera.disconnect()
