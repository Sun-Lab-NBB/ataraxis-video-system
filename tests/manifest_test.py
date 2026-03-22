"""Contains tests for classes and functions provided by the manifest.py module."""

from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData
from ataraxis_video_system.manifest import write_camera_manifest


def test_camera_source_data_creation() -> None:
    """Verifies creation of CameraSourceData instances with explicit field values."""
    source = CameraSourceData(id=112, name="face_camera")
    assert source.id == 112
    assert source.name == "face_camera"


def test_camera_source_data_defaults() -> None:
    """Verifies that CameraSourceData uses correct default values."""
    source = CameraSourceData()
    assert source.id == 0
    assert source.name == ""


def test_camera_manifest_yaml_roundtrip(tmp_path) -> None:
    """Verifies CameraManifest serialization and deserialization via YAML."""
    sources = [
        CameraSourceData(id=51, name="face_camera"),
        CameraSourceData(id=62, name="body_camera"),
    ]
    manifest = CameraManifest(sources=sources)

    yaml_path = tmp_path / "camera_manifest.yaml"
    manifest.to_yaml(file_path=yaml_path)

    loaded = CameraManifest.from_yaml(file_path=yaml_path)
    assert len(loaded.sources) == 2
    assert loaded.sources[0].id == 51
    assert loaded.sources[0].name == "face_camera"
    assert loaded.sources[1].id == 62
    assert loaded.sources[1].name == "body_camera"


def test_camera_manifest_empty_yaml_roundtrip(tmp_path) -> None:
    """Verifies YAML roundtrip for a CameraManifest with no sources."""
    manifest = CameraManifest(sources=[])

    yaml_path = tmp_path / "empty_manifest.yaml"
    manifest.to_yaml(file_path=yaml_path)

    loaded = CameraManifest.from_yaml(file_path=yaml_path)
    assert loaded.sources == []


def test_write_camera_manifest_new(tmp_path) -> None:
    """Verifies that write_camera_manifest creates a new manifest file."""
    write_camera_manifest(log_directory=tmp_path, source_id=111, name="test_camera")

    manifest_path = tmp_path / CAMERA_MANIFEST_FILENAME
    assert manifest_path.exists()

    loaded = CameraManifest.from_yaml(file_path=manifest_path)
    assert len(loaded.sources) == 1
    assert loaded.sources[0].id == 111
    assert loaded.sources[0].name == "test_camera"


def test_write_camera_manifest_append(tmp_path) -> None:
    """Verifies that write_camera_manifest appends to an existing manifest."""
    write_camera_manifest(log_directory=tmp_path, source_id=51, name="face_camera")
    write_camera_manifest(log_directory=tmp_path, source_id=62, name="body_camera")

    manifest_path = tmp_path / CAMERA_MANIFEST_FILENAME
    loaded = CameraManifest.from_yaml(file_path=manifest_path)
    assert len(loaded.sources) == 2
    assert loaded.sources[0].id == 51
    assert loaded.sources[0].name == "face_camera"
    assert loaded.sources[1].id == 62
    assert loaded.sources[1].name == "body_camera"
