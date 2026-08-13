from enum import StrEnum
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

CAMERA_EXTRACTION_JOB_NAME: str

class OutputLayout(StrEnum):
    DIRECTORY_NAME = "camera_timestamps"
    TRACKER_FILENAME = "camera_processing_tracker.yaml"
    FILE_PREFIX = "camera_"
    TIMESTAMPS_INFIX = "_timestamps"
    FILE_SUFFIX = ".feather"

@dataclass(frozen=True, slots=True)
class JobDescriptor:
    log_directory: Path
    archive_path: Path
    output_directory: Path
    tracker_path: Path
    job_name: str
    job_id: str
    source_id: str
    core_weight: int
    @classmethod
    def for_archive(
        cls,
        archive_path: Path,
        output_directory: Path,
        tracker_path: Path,
        source_id: str,
        log_directory: Path | None = None,
        core_weight: int = 1,
    ) -> JobDescriptor: ...
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> JobDescriptor: ...
    @property
    def dispatch_key(self) -> tuple[str, str]: ...
    def to_mapping(self) -> dict[str, str | int]: ...

@dataclass(frozen=True, slots=True)
class JobSizing:
    memory_mb: int
    message_count: int
    archive_bytes: int
    modeled: bool

def generate_job_ids(source_ids: Sequence[str]) -> dict[str, str]: ...
def resolve_output_directory(output_directory: Path) -> Path: ...
def resolve_tracker_path(output_directory: Path) -> Path: ...
def resolve_timestamps_path(output_directory: Path, source_id: str) -> Path: ...
