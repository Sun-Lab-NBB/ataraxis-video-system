from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray as NDArray
from ataraxis_data_structures import ProcessingTracker

from .manifest import (
    CAMERA_MANIFEST_FILENAME as CAMERA_MANIFEST_FILENAME,
    CameraManifest as CameraManifest,
)

LOG_ARCHIVE_SUFFIX: str
TRACKER_FILENAME: str
CAMERA_TIMESTAMPS_DIRECTORY: str
PARALLEL_PROCESSING_THRESHOLD: int
TIMESTAMP_JOB_NAME: str

def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]: ...
def find_log_archive(log_directory: Path, source_id: str) -> Path: ...
def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None = None,
    log_ids: list[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None: ...
def extract_logged_camera_timestamps(
    log_path: Path, n_workers: int = -1, *, display_progress: bool = True, executor: ProcessPoolExecutor | None = None
) -> NDArray[np.uint64]: ...
def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]]) -> None: ...
def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None: ...
def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]: ...
def generate_job_ids(source_ids: list[str]) -> dict[str, str]: ...
def _process_frame_message_batch(log_path: Path, keys: list[str], onset_us: np.uint64) -> NDArray[np.uint64]: ...
