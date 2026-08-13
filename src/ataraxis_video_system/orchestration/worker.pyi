from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from ataraxis_data_structures import ProcessingTracker

from .jobs import (
    CAMERA_EXTRACTION_JOB_NAME as CAMERA_EXTRACTION_JOB_NAME,
    JobDescriptor as JobDescriptor,
    resolve_timestamps_path as resolve_timestamps_path,
)
from ..video import (
    ExtractedDataColumns as ExtractedDataColumns,
    extract_logged_camera_timestamps as extract_logged_camera_timestamps,
)

def run_extraction_job(job: JobDescriptor) -> None: ...
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
