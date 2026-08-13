"""Provides the declared core allocation of the camera timestamp extraction stage, the archive-derived footprint model
that sizes each job's cores and memory from the input it will read, and the batch-wide budget resolvers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import psutil
from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import read_archive_message_count

if TYPE_CHECKING:
    from pathlib import Path

_RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when a core budget auto-resolves. The value reaches
``resolve_worker_count``, which applies it only to a non-positive budget and honors an explicit budget up to the
logical core count."""

CAMERA_EXTRACTION_JOB_CORES: int = 8
"""The core allocation every parallel camera timestamp extraction job receives.

Notes:
    Every worker opens the archive itself, so the fixed cost per worker holds as workers are added and the speedup
    flattens well before the core count. Doubling the allocation past this width returns a few percent of a job's
    runtime, and the curve turns over by twenty four workers, where opening the pool costs more than splitting the
    archive saves.

    The allocation also sets how many jobs a batch runs at once, because a core budget admits one job per allocation.
    A wider allocation therefore buys a single recording a little latency at the cost of the concurrency a batch of
    recordings needs, and this width is the point where a batch of ten still fits one admission wave on a host that
    reserves a hundred and twenty six cores for it.
"""

_PARALLEL_EXTRACTION_THRESHOLD: int = 35_000
"""The number of data messages an archive has to hold before an extraction job opens a pool to read it.

Notes:
    Opening a pool costs one spawned child per core, and every child re-imports the package before it reads a message.
    Below this count the archive is decoded before that fixed cost is repaid, so the parallel path finishes behind the
    sequential one. The count is the message load at which the two paths finish together, so it moves with the
    declared core allocation, which sets how many children the pool has to spawn.

    This threshold governs whether a pool is opened at all, which is a different decision from the message batching
    ``PARALLEL_PROCESSING_THRESHOLD`` governs inside the archive reader.
"""

SPAWNED_CHILD_MEMORY_MB: int = 208
"""The resident memory one spawned child holds before it touches any data, covering the interpreter and the package's
import graph. The term is charged once for a job's body and once more for each child of the extraction pool it opens,
so a job holding a single core and therefore carrying no pool is charged once.

Notes:
    The figure is resident memory rather than proportional memory. Roughly two fifths of it is pages the readers
    share, so a batch running many jobs at once holds less than the sum this term charges it.

    A child that only reads holds the import graph alone, while a job body also writes its output file and carries
    the pinned thread pool the write opens. This term covers the wider of the two, so one constant sizes both.
"""

_MEMORY_ESTIMATE_TOLERANCE: float = 1.15
"""The margin applied to every memory estimate before it is reported. It covers the working sets the model does not
enumerate and the variation between archives of the same size. The penalty for understating is asymmetric, since a
batch that overcommits its host swaps or is killed outright, so estimates round up."""

_POOL_MEMORY_RESERVATION_DIVISOR: int = 2
"""The share of the memory budget the shared pool's warmed job bodies may claim, expressed as a divisor. The
remainder covers the work those bodies perform."""

_ARCHIVE_DIRECTORY_RATIO: float = 2.66
"""The resident memory a log archive reader holds per byte of archive on disk. Reading an archive builds one
directory entry per logged message, which dominates the decoded payload itself.

Notes:
    Every reader holds its own copy of the directory, so this term scales with the reader count while the spawned child
    baseline it accompanies is roughly two fifths shared.
"""

_MEMORY_BUDGET_FRACTION: float = 0.85
"""The share of the host's physical memory an auto-resolved batch budget claims, leaving the remainder to the host."""

_MINIMUM_MEMORY_BUDGET_MB: int = 1024
"""The floor an auto-resolved memory budget is held to, so a small host still admits one job at a time."""

_MEMORY_ROUNDING_QUANTUM_MB: int = 256
"""The multiple every reportable estimate is rounded up to.

Notes:
    The quantum is roughly the memory one spawned child holds, so the smallest job shape is charged one quantum rather
    than the four a whole-gigabyte quantum charged it. A coarser quantum charges the two shapes the stage emits at
    nearly the same figure, which would let a batch of sequential jobs reserve the memory a batch of pooled ones needs.
"""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""


@dataclass(frozen=True, slots=True)
class ArchiveFootprint:
    """Describes the on-disk properties of one log archive that size the job reading it."""

    message_count: int
    """The number of data messages the archive holds."""
    archive_bytes: int
    """The size of the archive file on disk."""
    modeled: bool
    """Determines whether the figures were read from the archive rather than falling back to the job baseline."""


def resolve_archive_footprint(archive_path: Path) -> ArchiveFootprint:
    """Reads the properties of the target log archive that size the job reading it.

    Notes:
        Decodes no message and loads no payload. An archive that cannot be read yields an unmodeled footprint, which
        every consumer treats as a floor rather than as a measurement.

    Args:
        archive_path: The path to the .npz log archive to read.

    Returns:
        The footprint describing the archive.
    """
    try:
        message_count = read_archive_message_count(archive_path=archive_path)
        archive_bytes = archive_path.stat().st_size
    except Exception:
        return ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)

    return ArchiveFootprint(message_count=message_count, archive_bytes=archive_bytes, modeled=True)


def resolve_job_workers(footprint: ArchiveFootprint) -> int:
    """Resolves the cores one extraction job receives, from the archive it reads.

    Notes:
        The stage offers no width between the two it emits, because the speedup between one core and the declared
        allocation is smooth enough that a narrower pool costs a job time without returning a core the batch can
        place elsewhere.

    Args:
        footprint: The footprint of the archive this job reads.

    Returns:
        The cores this job receives, which is one or the declared allocation.
    """
    if footprint.message_count < _PARALLEL_EXTRACTION_THRESHOLD:
        return 1

    return CAMERA_EXTRACTION_JOB_CORES


def estimate_job_memory_mb(footprint: ArchiveFootprint, cores: int) -> int:
    """Estimates the memory one extraction job holds at its allocated core count.

    Notes:
        A job holding more than one core splits its archive across an extraction pool, and every child of that pool
        opens the archive itself. The archive's message directory is therefore held once per core, and the body is
        charged one more reader's worth for the working set it holds alongside its pool. A job holding a single core
        takes the sequential path, which opens no pool and holds the body's reader alone.

        An unmodeled footprint falls back to the spawned child baseline, which is a floor to plan around rather than
        a measurement.

    Args:
        footprint: The footprint of the archive this job reads.
        cores: The cores this job holds, which is how many extraction pool children it opens, or none when it is one.

    Returns:
        The memory this job holds, in megabytes, carrying the estimate tolerance and rounded up to the reporting
        quantum.
    """
    if not footprint.modeled:
        return _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB)

    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)

    if cores == 1:
        return _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB + per_reader)

    readers = cores + 1
    return _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB * readers + per_reader * readers)


def size_archive_job(archive_path: Path) -> tuple[int, int, bool]:
    """Resolves the cores and the memory one extraction job receives, from the archive it reads.

    Notes:
        Reads the archive once and answers both halves of the sizing model from that read, so a scheduler planning
        this stage reproduces neither the width rule nor the memory model.

    Args:
        archive_path: The path to the .npz log archive the job reads.

    Returns:
        The cores the job receives, the memory it holds in megabytes, and whether both figures follow from the
        archive itself rather than from the spawned child baseline.
    """
    footprint = resolve_archive_footprint(archive_path=archive_path)
    cores = resolve_job_workers(footprint=footprint)
    return cores, estimate_job_memory_mb(footprint=footprint, cores=cores), footprint.modeled


def resolve_core_budget(requested_budget: int) -> int:
    """Resolves the cores a batch may commit across all of its concurrently running jobs.

    Args:
        requested_budget: The cores the caller requested. A non-positive value auto-resolves to every available core
            minus the reserved host cores.

    Returns:
        The cores the batch may commit, always at least one.
    """
    return resolve_worker_count(requested_workers=requested_budget, reserved_cores=_RESERVED_CORES)


def resolve_memory_budget_mb(requested_budget_mb: int) -> int:
    """Resolves the memory a batch may commit across all of its concurrently running jobs.

    Args:
        requested_budget_mb: The memory the caller requested, in megabytes. A non-positive value auto-resolves to a
            share of the host's physical memory.

    Returns:
        The memory the batch may commit, in megabytes. A positive request is returned verbatim, while an
        auto-resolved budget is held to at least the auto-resolution floor.
    """
    if requested_budget_mb > 0:
        return requested_budget_mb

    return max(_MINIMUM_MEMORY_BUDGET_MB, int(resolve_host_memory_mb() * _MEMORY_BUDGET_FRACTION))


def resolve_pool_size(job_count: int, core_budget: int, memory_budget_mb: int) -> int:
    """Resolves the job slots one batch's shared pool opens.

    Notes:
        A slot holds a job rather than a core, so the count covers the widest running set admission can produce.
        Every slot is warmed at creation and holds a spawned child's baseline memory for the whole session, so the
        count is held to the bodies half the memory budget can hold.

    Args:
        job_count: The jobs the batch holds.
        core_budget: The cores the batch may commit across all concurrently running jobs.
        memory_budget_mb: The memory the batch may commit across all concurrently running jobs.

    Returns:
        The job slots the shared pool opens, always at least one.
    """
    affordable_bodies = (memory_budget_mb // _POOL_MEMORY_RESERVATION_DIVISOR) // SPAWNED_CHILD_MEMORY_MB
    return max(1, min(job_count, core_budget, affordable_bodies))


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory, in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return int(byte_count / _BYTES_PER_MEGABYTE) + 1 if byte_count > 0 else 0


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the estimate tolerance to a modeled memory figure and rounds it up to the reporting quantum.

    Notes:
        Rounding to a fixed quantum here keeps the figure an admission decision weighs identical to the figure a
        caller reports, which is what lets a planned batch and a running one be compared directly.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance.
    """
    reportable = int(memory_mb * _MEMORY_ESTIMATE_TOLERANCE) + 1
    return math.ceil(reportable / _MEMORY_ROUNDING_QUANTUM_MB) * _MEMORY_ROUNDING_QUANTUM_MB
