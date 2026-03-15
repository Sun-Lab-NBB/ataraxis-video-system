"""Contains tests for classes and functions provided by the configuration.py module."""

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system import (
    CameraInterfaces,
    GenicamNodeInfo,
    GenicamConfiguration,
    discover_camera_ids,
)
from ataraxis_video_system.camera import HarvestersCamera


@pytest.fixture(scope="session")
def has_harvesters():
    """Checks for Harvesters camera availability in the test environment."""
    try:
        all_cameras = discover_camera_ids()
        return any(cam.interface == CameraInterfaces.HARVESTERS for cam in all_cameras)
    except Exception:
        return False


def test_genicam_node_info_creation() -> None:
    """Verifies creation of GenicamNodeInfo instances with and without a unit."""
    node = GenicamNodeInfo(name="Width", value=1920, unit="px")
    assert node.name == "Width"
    assert node.value == 1920
    assert node.unit == "px"

    node_no_unit = GenicamNodeInfo(name="Gain", value=1.5)
    assert node_no_unit.name == "Gain"
    assert node_no_unit.value == 1.5
    assert node_no_unit.unit is None


def test_genicam_node_info_types() -> None:
    """Verifies that GenicamNodeInfo correctly stores all supported value types."""
    assert GenicamNodeInfo(name="IntNode", value=42).value == 42
    assert GenicamNodeInfo(name="FloatNode", value=3.14).value == 3.14
    assert GenicamNodeInfo(name="BoolNode", value=True).value is True
    assert GenicamNodeInfo(name="StrNode", value="Mono8").value == "Mono8"


def test_genicam_configuration_yaml_roundtrip(tmp_path) -> None:
    """Verifies GenicamConfiguration serialization and deserialization via YAML."""
    nodes = [
        GenicamNodeInfo(name="Width", value=1920, unit="px"),
        GenicamNodeInfo(name="Height", value=1080),
        GenicamNodeInfo(name="Gain", value=2.5, unit="dB"),
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
    assert loaded.nodes[0].value == 1920
    assert loaded.nodes[0].unit == "px"
    assert loaded.nodes[1].name == "Height"
    assert loaded.nodes[1].value == 1080
    assert loaded.nodes[1].unit is None
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
        assert len(camera.model) > 0
        assert isinstance(camera.serial_number, str)
        assert len(camera.serial_number) > 0
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
        assert len(config.camera_model) > 0
        assert len(config.camera_serial_number) > 0
        assert len(config.nodes) > 0
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
