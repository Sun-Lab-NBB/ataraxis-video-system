"""Provides the manifest-derived job resolution every consumer shares, the preparation that turns a resolved universe
into dispatchable job descriptors, and the archive pass that sizes one prepared job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace, dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingTracker,
    index_marker_files,
    discover_marker_files,
)

from .jobs import (
    CAMERA_EXTRACTION_JOB_NAME,
    JobSizing,
    JobDescriptor,
    generate_job_ids,
    resolve_tracker_path,
    resolve_output_directory,
)
from ..video import CAMERA_MANIFEST_FILENAME, CameraManifest
from .allocation import (
    CAMERA_EXTRACTION_JOB_CORES,
    ArchiveFootprint,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_archive_footprint,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class JobSource:
    """Describes one camera source the manifest registers and the log archive it produced."""

    source_id: str
    """The identifier of the source, as it appears in every job specifier and every archive filename."""
    name: str
    """The colloquial name the manifest records for the source."""
    archive_path: Path | None
    """The path to the source's log archive, or None when the tree holds no single archive for it."""


@dataclass(frozen=True, slots=True)
class JobUniverse:
    """Describes every extraction job one log directory's manifest defines and the subset its archives back."""

    log_directory: Path
    """The root directory the resolution searched."""
    manifest_path: Path | None
    """The path to the single camera manifest the directory holds, or None when the tree holds none."""
    sources: tuple[JobSource, ...]
    """Every source the manifest registers, in ascending source identifier order."""
    universe: tuple[tuple[str, str], ...]
    """Every job the manifest defines, as job name and source identifier pairs.

    Notes:
        This is a manifest fingerprint rather than an invocation fingerprint, so every invocation aligns a tracker
        against the same set and no invocation resets the jobs it did not request.
    """
    possible: tuple[tuple[str, str], ...]
    """The subset of the universe whose archive resolved to exactly one file under the log directory."""

    @property
    def archives(self) -> dict[str, Path]:
        """Returns the resolved archive of each source that has one, keyed by that source identifier."""
        return {source.source_id: source.archive_path for source in self.sources if source.archive_path is not None}


@dataclass(frozen=True, slots=True)
class JobSet:
    """Describes the dispatchable extraction jobs one invocation prepared for one log directory."""

    log_directory: Path
    """The root directory holding the manifest and the log archives."""
    output_directory: Path
    """The subdirectory the preparation created, which holds the tracker and every output file."""
    tracker_path: Path
    """The path to the ProcessingTracker file recording every job in this set."""
    universe: tuple[tuple[str, str], ...]
    """Every job the manifest defines, which is the set the tracker is aligned against."""
    jobs: tuple[JobDescriptor, ...]
    """Every dispatchable job this set holds, in ascending source identifier order."""
    skipped_sources: tuple[tuple[str, str], ...]
    """Each source that yielded no job, paired with the reason. Always empty under strict sourcing, where a source
    that cannot be prepared raises instead."""


def resolve_jobs(log_directory: Path) -> JobUniverse:
    """Resolves the extraction job universe of one log directory and the subset its archives back.

    Notes:
        Reads the manifest and indexes the archive filenames, decoding no message and writing nothing, so a caller
        enumerates a directory's jobs without launching or materializing anything. Two tree walks serve any number
        of sources, one for the manifest and one indexing every archive name the manifest implies.

        One recording writes one VideoSystem to one DataLogger, so a tree holding several manifests spans several
        recordings and is rejected rather than resolved against the first manifest found. A tree holding no manifest
        holds no camera jobs, and yields an empty universe rather than an error.

    Args:
        log_directory: The root directory whose tree is searched for the camera manifest and the log archives.

    Returns:
        The resolved job universe.

    Raises:
        FileNotFoundError: If the log directory does not exist or is not a directory.
        ValueError: If the tree holds more than one camera manifest, or if a manifest registers no sources.
        OSError: If any directory beneath the log directory cannot be read.
        YAMLError: If the camera manifest does not hold a well-formed YAML document.
        MissingValueError: If the camera manifest omits a field the CameraManifest class requires.
    """
    if not log_directory.is_dir():
        message = (
            f"Unable to resolve camera timestamp extraction jobs in '{log_directory}'. The path does not exist or is "
            f"not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    candidates = discover_marker_files(directory=log_directory, marker_name=CAMERA_MANIFEST_FILENAME)

    # A tree holding no camera manifest holds no camera jobs, which is an answer rather than a failure. A caller
    # walking many recordings reads the empty universe and moves on, while a caller that asked for work to be done
    # raises on the empty result itself.
    if not candidates:
        return JobUniverse(
            log_directory=log_directory,
            manifest_path=None,
            sources=(),
            universe=(),
            possible=(),
        )

    if len(candidates) > 1:
        # Joins the paths into the message text, because interpolating the list renders each path through its repr,
        # which doubles every separator of a Windows path and reports one the reader is unable to use as printed.
        reported_candidates = ", ".join(f"'{candidate}'" for candidate in candidates)
        message = (
            f"Unable to resolve camera timestamp extraction jobs in '{log_directory}'. The directory tree holds "
            f"{len(candidates)} {CAMERA_MANIFEST_FILENAME} files, which means it spans several recordings or several "
            f"DataLogger instances: {reported_candidates}. One recording writes one VideoSystem to one logger, so "
            f"exactly one manifest is supported per invocation. Pass the individual DataLogger output directory of "
            f"each recording instead."
        )
        console.error(message=message, error=ValueError)

    manifest_path = candidates[0]
    manifest = CameraManifest.from_yaml(file_path=manifest_path)
    entries = {str(source.id): source.name for source in manifest.sources}

    if not entries:
        message = (
            f"Unable to resolve camera timestamp extraction jobs in '{log_directory}'. The "
            f"{CAMERA_MANIFEST_FILENAME} at '{manifest_path}' contains no source entries."
        )
        console.error(message=message, error=ValueError)

    source_ids = sorted(entries)

    # Indexes every source's archive in one pass, since the archive names are known once the manifest resolves. A
    # source whose name resolves to several archives spans several loggers, which is ambiguous rather than redundant,
    # so it is left unresolved alongside the sources holding no archive at all.
    archives = index_marker_files(
        directory=log_directory,
        marker_names=[f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids],
    )
    matches = {source_id: archives[f"{source_id}{LOG_ARCHIVE_SUFFIX}"] for source_id in source_ids}

    sources = tuple(
        JobSource(
            source_id=source_id,
            name=entries[source_id],
            archive_path=matches[source_id][0] if len(matches[source_id]) == 1 else None,
        )
        for source_id in source_ids
    )

    return JobUniverse(
        log_directory=log_directory,
        manifest_path=manifest_path,
        sources=sources,
        universe=tuple((CAMERA_EXTRACTION_JOB_NAME, source_id) for source_id in source_ids),
        possible=tuple(
            (CAMERA_EXTRACTION_JOB_NAME, source.source_id) for source in sources if source.archive_path is not None
        ),
    )


def prepare_jobs(
    log_directory: Path,
    output_directory: Path,
    source_ids: Sequence[str] | None = None,
    job_id: str | None = None,
    *,
    strict_sources: bool = True,
) -> JobSet:
    """Resolves and registers the camera timestamp extraction jobs of one log directory.

    Notes:
        Materializes the output subdirectory and aligns the tracker against the manifest universe, which is every
        write this call performs outside a job's own output. The prepared job list lives in the returned set rather
        than on disk.

        Reads no archive. Every job carries the declared allocation as its width, which the sizing pass replaces with
        the width the job's own archive resolves to.

        The tracker is aligned against the whole manifest universe whichever subset this call prepares, so several
        invocations naming different jobs share one tracker without resetting each other's recorded outcomes.

        A tree holding no manifest is rejected whatever the sourcing mode, because the absent manifest is a property
        of the directory rather than of any one requested source.

    Args:
        log_directory: The root directory whose tree holds the camera manifest and the log archives.
        output_directory: The root output directory. The library's own subdirectory is created under it.
        source_ids: The sources to prepare jobs for, or None to prepare every source the manifest registers. The
            argument is ignored when a job identifier selects the work.
        job_id: The hexadecimal identifier of the single job to prepare. Leaving this unset prepares every requested
            source.
        strict_sources: Determines whether a source that cannot be prepared stops the call. When set, a requested
            source the manifest does not register, or one whose archive does not resolve to exactly one file, raises.
            When unset, such a source is recorded in the returned set's skipped sources with its reason.

    Returns:
        The prepared job set.

    Raises:
        FileNotFoundError: If the log directory does not exist, if the log directory's tree holds no camera
            manifest, or if a requested source's archive is absent under strict sourcing.
        ValueError: If the tree holds more than one camera manifest, if a manifest registers no sources, if a
            requested source or job identifier is not registered, or if the resolved archives span several
            directories.
        OSError: If any directory beneath the log directory cannot be read.
        YAMLError: If the camera manifest does not hold a well-formed YAML document.
        MissingValueError: If the camera manifest omits a field the CameraManifest class requires.
        TimeoutError: If the tracker's .lock file cannot be acquired within the timeout period.
    """
    universe = resolve_jobs(log_directory=log_directory)

    # Reported before the requested sources are weighed, so the directory itself carries the reason.
    if universe.manifest_path is None:
        message = (
            f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. Its tree holds no "
            f"{CAMERA_MANIFEST_FILENAME}, so no source in it is registered and no requested source can be prepared. "
            f"The archives beneath it were not produced by ataraxis-video-system, or the recording was logged "
            f"without a manifest."
        )
        console.error(message=message, error=FileNotFoundError)

    registered_ids = [source.source_id for source in universe.sources]
    archives = universe.archives

    if job_id is not None:
        # An identifier names one job, so the requested set is the source whose identifier matches it. Resolving it
        # against the manifest universe rather than against the archives on disk keeps a missing sibling archive
        # from hiding the job that was actually named.
        identifiers = generate_job_ids(source_ids=registered_ids)
        matched = [source_id for source_id, candidate in identifiers.items() if candidate == job_id]

        if not matched:
            message = (
                f"Unable to prepare the camera timestamp extraction job '{job_id}' in '{log_directory}'. The camera "
                f"manifest at '{universe.manifest_path}' defines no job with that identifier. Registered source IDs: "
                f"{', '.join(registered_ids)}."
            )
            console.error(message=message, error=ValueError)

        requested_ids = matched
    else:
        # Collapses a repeated identifier, since one source addresses one archive and one tracker entry. Two
        # descriptors for one source share a dispatch key, which leaves the batch engine tracking one running job
        # while two workers extract the same archive and write the same output file.
        requested_ids = sorted(set(source_ids)) if source_ids else registered_ids

    unregistered_ids = [source_id for source_id in requested_ids if source_id not in registered_ids]
    skipped: list[tuple[str, str]] = []

    if unregistered_ids:
        message = (
            f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The following requested "
            f"source IDs are not registered in the {CAMERA_MANIFEST_FILENAME} at '{universe.manifest_path}': "
            f"{', '.join(unregistered_ids)}. The corresponding log archives were not produced by "
            f"ataraxis-video-system. Registered source IDs: {', '.join(registered_ids)}."
        )
        if strict_sources:
            console.error(message=message, error=ValueError)
        skipped.extend(
            (source_id, "The source is not registered in the camera manifest.") for source_id in unregistered_ids
        )

    unresolved_ids = [
        source_id for source_id in requested_ids if source_id in registered_ids and source_id not in archives
    ]

    if unresolved_ids:
        message = (
            f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The log archives of the "
            f"following requested source IDs are absent or resolve to more than one file: "
            f"{', '.join(unresolved_ids)}."
        )
        if strict_sources:
            console.error(message=message, error=FileNotFoundError)
        skipped.extend(
            (source_id, "The source's log archive is absent or resolves to more than one file.")
            for source_id in unresolved_ids
        )

    prepared_ids = [source_id for source_id in requested_ids if source_id in archives]
    parent_directories = {archives[source_id].parent for source_id in prepared_ids}

    if len(parent_directories) > 1:
        # Joins the paths into the message text, because interpolating the list renders each path through its repr,
        # which doubles every separator of a Windows path and reports one the reader is unable to use as printed.
        sorted_directories = sorted(str(parent) for parent in parent_directories)
        reported_directories = ", ".join(f"'{directory}'" for directory in sorted_directories)
        message = (
            f"Unable to prepare camera timestamp extraction jobs in '{log_directory}'. The resolved log archives sit "
            f"in {len(parent_directories)} different directories: {reported_directories}. Archives in separate "
            f"directories were written by separate DataLogger instances, and one recording writes one logger, so this "
            f"tree holds more than one recording. Each DataLogger output directory must be prepared and processed on "
            f"its own invocation."
        )
        console.error(message=message, error=ValueError)

    resolved_output = resolve_output_directory(output_directory=output_directory)
    tracker_path = resolve_tracker_path(output_directory=resolved_output)

    # Creates the output layout only once a job is going to be written into it, so a lenient request that prepared
    # nothing leaves the caller's output path as it found it.
    if prepared_ids:
        resolved_output.mkdir(parents=True, exist_ok=True)

    jobs = tuple(
        JobDescriptor.for_archive(
            archive_path=archives[source_id],
            output_directory=resolved_output,
            tracker_path=tracker_path,
            source_id=source_id,
            log_directory=log_directory,
            core_weight=CAMERA_EXTRACTION_JOB_CORES,
        )
        for source_id in prepared_ids
    )

    # Aligns the tracker with the prepared subset while detecting foreign entries against the full manifest universe,
    # so an invocation covering part of a recording leaves its siblings' recorded outcomes untouched. A lenient
    # request that prepared nothing registers nothing, since a tracker names at least one job and the caller receives
    # the reasons through the skipped sources instead.
    if prepared_ids:
        ProcessingTracker(file_path=tracker_path).align_jobs(
            jobs=[(CAMERA_EXTRACTION_JOB_NAME, source_id) for source_id in prepared_ids],
            universe=list(universe.universe),
        )

    return JobSet(
        log_directory=log_directory,
        output_directory=resolved_output,
        tracker_path=tracker_path,
        universe=universe.universe,
        jobs=jobs,
        skipped_sources=tuple(sorted(skipped)),
    )


def size_job(job: JobDescriptor) -> tuple[JobDescriptor, JobSizing, ArchiveFootprint]:
    """Sizes one prepared job from the archive it reads.

    Notes:
        Reads the archive's zip directory and its file metadata alone, decoding no message.

    Args:
        job: The prepared job to size.

    Returns:
        The job carrying its resolved width, the resources the sizing produced, and the archive footprint they
        follow from, in that order.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run.
    """
    footprint = resolve_archive_footprint(archive_path=job.archive_path)
    core_weight = resolve_job_workers(footprint=footprint)

    return (
        replace(job, core_weight=core_weight),
        JobSizing(cores=core_weight, memory_mb=estimate_job_memory_mb(footprint=footprint, cores=core_weight)),
        footprint,
    )
