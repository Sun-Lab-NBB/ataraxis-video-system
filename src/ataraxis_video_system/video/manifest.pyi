from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_data_structures import YamlConfig

CAMERA_MANIFEST_FILENAME: str
_MANIFEST_LOCK_TIMEOUT: float

@dataclass(frozen=True, slots=True)
class CameraSourceData:
    id: int = ...
    name: str = ...

@dataclass
class CameraManifest(YamlConfig):
    sources: list[CameraSourceData] = field(default_factory=list)

def write_camera_manifest(log_directory: Path, source_id: int, name: str) -> None: ...
