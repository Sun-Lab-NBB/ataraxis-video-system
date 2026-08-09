"""Provides the job identity constants, the batch job descriptor, and the manifest-derived job discovery shared by
every consumer that schedules camera timestamp extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingTracker,
    index_marker_files,
    discover_marker_files,
)

from ..video import CAMERA_MANIFEST_FILENAME, CameraManifest

if TYPE_CHECKING:
    from pathlib import Path

TIMESTAMP_JOB_NAME: str = "camera_timestamp_extraction"
"""The job name under which camera timestamp extraction is registered in a ProcessingTracker.

Notes:
    The value is hashed into every persisted job identifier, so changing the string invalidates every identifier a
    tracker already holds and every identifier a scheduler derived independently.
"""

TRACKER_FILENAME: str = "camera_processing_tracker.yaml"
"""The name of the processing tracker file the pipeline places in its output directory."""

CAMERA_TIMESTAMPS_DIRECTORY: str = "camera_timestamps"
"""The name of the subdirectory the pipeline creates under its output path for tracker and feather files."""

CAMERA_TIMESTAMPS_PREFIX: str = "camera_"
"""The prefix of the feather file each extraction job writes."""

CAMERA_TIMESTAMPS_SUFFIX: str = "_timestamps.feather"
"""The suffix of the feather file each extraction job writes."""

FRAME_TIME_COLUMN: str = "frame_time_us"
"""The name of the feather column holding the frame acquisition timestamps, in microseconds since the UTC epoch."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single timestamp extraction job queued for batch execution.

    Notes:
        The core and memory weights are resolved from the job's own archive before dispatch, so admission weighs each
        job against the budgets at the size that archive actually demands.
    """

    log_directory: Path
    """The path to the DataLogger output directory whose tree holds the log archive."""
    output_directory: Path
    """The path to the directory the extracted feather file is written to."""
    tracker_path: Path
    """The path to the ProcessingTracker file that records this job's outcome."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    source_id: str
    """The identifier of the camera source whose archive this job reads."""
    core_weight: int = 1
    """The cores this job occupies while it runs."""
    memory_mb: int = 0
    """The memory this job occupies while it runs, estimated from the archive it reads."""
    archive_path: Path | None = None
    """The path to the archive this job reads, resolved while the job is sized, or None when it did not resolve."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite tracker path and job identifier pair that identifies this job across the batch."""
        return str(self.tracker_path), self.job_id


def discover_camera_jobs(log_directory: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the timestamp extraction job universe and the subset backed by an archive on disk.

    Notes:
        The universe is a manifest fingerprint rather than an invocation fingerprint, so every invocation aligns a
        tracker against the same set and no invocation resets the jobs it did not request. The camera manifest also
        gates the discovery, which keeps archives written by other libraries out of the resolved set.

    Args:
        log_directory: The root directory whose tree is searched for the camera manifest and the log archives.

    Returns:
        The full job universe the manifest defines and the subset whose archives resolve to exactly one file, each as
        a list of job name and source identifier pairs.

    Raises:
        FileNotFoundError: If the log directory does not exist, is not a directory, or holds no camera manifest.
        OSError: If any directory beneath the log directory cannot be read.
        ValueError: If the camera manifest registers no sources.
    """
    if not log_directory.is_dir():
        message = (
            f"Unable to discover camera timestamp extraction jobs in '{log_directory}'. The path does not exist or "
            f"is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    candidates = discover_marker_files(directory=log_directory, marker_name=CAMERA_MANIFEST_FILENAME)
    if not candidates:
        message = (
            f"Unable to discover camera timestamp extraction jobs in '{log_directory}'. No "
            f"{CAMERA_MANIFEST_FILENAME} was found. A camera manifest is required to identify which log archives "
            f"were produced by ataraxis-video-system."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = CameraManifest.from_yaml(file_path=candidates[0])
    source_ids = sorted({str(source.id) for source in manifest.sources})

    if not source_ids:
        message = (
            f"Unable to discover camera timestamp extraction jobs in '{log_directory}'. The "
            f"{CAMERA_MANIFEST_FILENAME} at '{candidates[0]}' contains no source entries."
        )
        console.error(message=message, error=ValueError)

    universe = [(TIMESTAMP_JOB_NAME, source_id) for source_id in source_ids]

    # Indexes every source's archive in one pass, since the archive names are known once the manifest resolves. A
    # source whose name resolves to several archives spans several loggers, which is ambiguous rather than redundant,
    # so it is left out of the possible set alongside the sources holding no archive at all.
    archives = index_marker_files(
        directory=log_directory,
        marker_names=[f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids],
    )
    possible = [
        (TIMESTAMP_JOB_NAME, source_id)
        for source_id in source_ids
        if len(archives[f"{source_id}{LOG_ARCHIVE_SUFFIX}"]) == 1
    ]

    return universe, possible


def generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates the processing job identifier of every requested camera source.

    Args:
        source_ids: The camera source identifiers to generate job identifiers for.

    Returns:
        The generated hexadecimal job identifier of each source, keyed by that source identifier.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def resolve_camera_timestamps_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the path of the feather file holding the target source's extracted timestamps.

    Args:
        output_directory: The directory the extraction job writes its output into.
        source_id: The identifier of the camera source whose output path is resolved.

    Returns:
        The path to the source's timestamp feather file.
    """
    return output_directory / f"{CAMERA_TIMESTAMPS_PREFIX}{source_id}{CAMERA_TIMESTAMPS_SUFFIX}"
