"""Provides the job identity constants, the output layout names and resolvers, and the descriptor and sizing records
every consumer that schedules camera timestamp extraction exchanges.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import fields, dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CAMERA_EXTRACTION_JOB_NAME: str = "camera_timestamp_extraction"
"""The job name under which camera timestamp extraction is registered in a ProcessingTracker.

Notes:
    The value is hashed into every persisted job identifier, so changing the string invalidates every identifier a
    tracker already holds and every identifier a scheduler derived independently.
"""


class OutputLayout(StrEnum):
    """Defines the filesystem names an extraction job writes its tracker and its output files under."""

    DIRECTORY_NAME = "camera_timestamps"
    """The subdirectory created under a caller's output path for the tracker and the extracted files."""
    TRACKER_FILENAME = "camera_processing_tracker.yaml"
    """The processing tracker file recording the outcome of every job writing to one directory."""
    FILE_PREFIX = "camera_"
    """The prefix of every output file an extraction job writes."""
    TIMESTAMPS_INFIX = "_timestamps"
    """The infix marking an output file as holding frame acquisition timestamps."""
    FILE_SUFFIX = ".feather"
    """The filename suffix of every output (Arrow IPC) file an extraction job writes."""


@dataclass(frozen=True, slots=True)
class JobDescriptor:
    """Describes one camera timestamp extraction job, addressed by the single log archive it reads.

    Notes:
        Every field is a path, a string, or an integer, so an instance pickles into a spawned worker and crosses a
        scheduler boundary or a tool payload unchanged.

        The archive path is resolved rather than optional, so a dispatched job never searches the tree.

        The figures a sizing pass produces live in the paired JobSizing record, which a worker never sees.
    """

    log_directory: Path
    """The path to the DataLogger output directory whose tree holds the log archive."""
    archive_path: Path
    """The path to the .npz log archive this job reads."""
    output_directory: Path
    """The path to the directory this job writes its output file into."""
    tracker_path: Path
    """The path to the ProcessingTracker file that records this job's outcome."""
    job_name: str
    """The tracker job name this job is registered under."""
    job_id: str
    """The unique hexadecimal identifier of this job in its tracker."""
    source_id: str
    """The identifier of the camera source whose archive this job reads."""
    core_weight: int
    """The cores this job occupies while it runs, which is the width of the extraction pool its body opens when it
    holds more than one core.
    """

    @classmethod
    def for_archive(
        cls,
        archive_path: Path,
        output_directory: Path,
        tracker_path: Path,
        source_id: str,
        log_directory: Path | None = None,
        core_weight: int = 1,
    ) -> JobDescriptor:
        """Builds a descriptor for one archive an external scheduler has already resolved.

        Notes:
            Derives the job identifier as this library's own preparation does, so one built here addresses the same
            tracker entry.

        Args:
            archive_path: The path to the .npz log archive the job reads.
            output_directory: The path to the directory the job writes its output file into.
            tracker_path: The path to the ProcessingTracker file that records the job's outcome.
            source_id: The identifier of the camera source whose archive the job reads.
            log_directory: The path to the DataLogger output directory holding the archive. Leaving this unset uses
                the archive's own parent directory.
            core_weight: The cores the job occupies while it runs.

        Returns:
            The built descriptor.
        """
        return cls(
            log_directory=log_directory if log_directory is not None else archive_path.parent,
            archive_path=archive_path,
            output_directory=output_directory,
            tracker_path=tracker_path,
            job_name=CAMERA_EXTRACTION_JOB_NAME,
            job_id=ProcessingTracker.generate_job_id(job_name=CAMERA_EXTRACTION_JOB_NAME, specifier=source_id),
            source_id=source_id,
            core_weight=core_weight,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> JobDescriptor:
        """Reconstructs a descriptor from the mapping a caller received across a tool boundary.

        Args:
            mapping: The mapping to read, carrying every field name to_mapping writes.

        Returns:
            The reconstructed descriptor.

        Raises:
            ValueError: If a required key is absent, or if a value cannot be read as the type its field
                declares.
        """
        field_names = tuple(field.name for field in fields(cls))
        missing_keys = [name for name in field_names if name not in mapping]

        if missing_keys:
            message = (
                f"Unable to read a camera timestamp extraction job descriptor from the supplied mapping. The "
                f"following required keys are absent: {', '.join(sorted(missing_keys))}. A descriptor mapping "
                f"carries every key the descriptor writes: {', '.join(field_names)}."
            )
            console.error(message=message, error=ValueError)

        try:
            return cls(
                log_directory=Path(mapping["log_directory"]),
                archive_path=Path(mapping["archive_path"]),
                output_directory=Path(mapping["output_directory"]),
                tracker_path=Path(mapping["tracker_path"]),
                job_name=str(mapping["job_name"]),
                job_id=str(mapping["job_id"]),
                source_id=str(mapping["source_id"]),
                core_weight=int(mapping["core_weight"]),
            )
        except (TypeError, ValueError) as error:
            message = (
                f"Unable to read a camera timestamp extraction job descriptor from the supplied mapping. One of its "
                f"values cannot be read as the type its field declares: {error}."
            )
            console.error(message=message, error=ValueError)

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the tracker path and job identifier pair that identifies this job across the batch."""
        return str(self.tracker_path), self.job_id

    def to_mapping(self) -> dict[str, str | int]:
        """Renders this descriptor as the flat mapping the interface layer exchanges.

        Notes:
            Every value is a string or an integer, so the mapping reconstructs through from_mapping without loss.

        Returns:
            The descriptor's fields keyed by their field names, with every path rendered as a string.
        """
        return {
            "log_directory": str(self.log_directory),
            "archive_path": str(self.archive_path),
            "output_directory": str(self.output_directory),
            "tracker_path": str(self.tracker_path),
            "job_name": self.job_name,
            "job_id": self.job_id,
            "source_id": self.source_id,
            "core_weight": self.core_weight,
        }


@dataclass(frozen=True, slots=True)
class JobSizing:
    """Describes the resource figures one sizing pass resolved for a single extraction job."""

    memory_mb: int
    """The memory the job occupies while it runs, estimated from the archive it reads."""
    message_count: int
    """The data messages the archive holds, as the sizing pass read them."""
    archive_bytes: int
    """The size of the archive on disk, in bytes, as the sizing pass read it."""
    modeled: bool
    """Determines whether the figures follow from the archive's own properties rather than from the job baseline."""


def generate_job_ids(source_ids: Sequence[str]) -> dict[str, str]:
    """Generates the processing job identifier of every requested camera source.

    Args:
        source_ids: The camera source identifiers to generate job identifiers for.

    Returns:
        The generated hexadecimal job identifier of each source, keyed by that source identifier.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=CAMERA_EXTRACTION_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def resolve_output_directory(output_directory: Path) -> Path:
    """Resolves the subdirectory the extraction output and its tracker are written into.

    Args:
        output_directory: The root output directory the caller nominated.

    Returns:
        The path to the library's own subdirectory under the nominated root.
    """
    return output_directory / OutputLayout.DIRECTORY_NAME


def resolve_tracker_path(output_directory: Path) -> Path:
    """Resolves the path of the processing tracker recording the outcome of every job writing to a directory.

    Args:
        output_directory: The directory the extraction jobs write their output into.

    Returns:
        The path to the tracker file.
    """
    return output_directory / OutputLayout.TRACKER_FILENAME


def resolve_timestamps_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the path of the file holding the target source's extracted timestamps.

    Args:
        output_directory: The directory the extraction jobs write their output into.
        source_id: The identifier of the camera source whose output path is resolved.

    Returns:
        The path to the source's timestamp file.
    """
    filename = f"{OutputLayout.FILE_PREFIX}{source_id}{OutputLayout.TIMESTAMPS_INFIX}{OutputLayout.FILE_SUFFIX}"
    return output_directory / filename
