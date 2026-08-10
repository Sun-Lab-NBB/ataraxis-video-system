"""Provides the failure kinds the orchestration layer reports and the single error type that carries them."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from functools import partial

if TYPE_CHECKING:
    from collections.abc import Callable


class OrchestrationErrors(StrEnum):
    """Defines every failure the orchestration layer reports for an input or a runtime condition it cannot process."""

    MISSING_LOG_MANIFEST = "missing_log_manifest"
    """The target log directory does not exist or holds no camera manifest."""
    AMBIGUOUS_LOG_DIRECTORY = "ambiguous_log_directory"
    """The target log directory tree holds more than one camera manifest, so it spans several recordings."""
    EMPTY_LOG_MANIFEST = "empty_log_manifest"
    """The camera manifest registers no sources."""
    UNKNOWN_JOB_SOURCE = "unknown_job_source"
    """A requested source identifier is absent from the camera manifest."""
    UNRESOLVED_ARCHIVE = "unresolved_archive"
    """A requested source's log archive is absent or resolves to more than one file."""
    SPLIT_LOGGER_OUTPUT = "split_logger_output"
    """The resolved log archives do not share one parent directory, so they span several DataLogger instances."""
    MALFORMED_JOB_DESCRIPTOR = "malformed_job_descriptor"
    """A mapping handed to the job descriptor reader lacks a required key or carries an unreadable value."""
    UNKNOWN_JOB_ID = "unknown_job_id"
    """A job identifier was requested from a sized job set that does not hold it."""
    JOB_EXCEEDS_HOST_MEMORY = "job_exceeds_host_memory"
    """One job's memory estimate passes the host's total physical memory, so it cannot complete anywhere."""

    def as_error(self) -> Callable[..., OrchestrationError]:
        """Returns the callable that raises this failure kind with a caller's message.

        Notes:
            Binds the kind to the error type, matching the single-argument callable the console's error reporting
            invokes.

        Returns:
            The callable that builds the error carrying this kind.
        """
        return partial(OrchestrationError, kind=self)


class OrchestrationError(RuntimeError):
    """Reports one failure the orchestration layer cannot process, identified by its kind.

    Notes:
        The kind survives pickling, so a failure raised in a spawned worker reaches the submitting process intact.

    Args:
        message: The description of the failure, naming the action, the constraint, and the offending value.
        kind: The failure this error reports.

    Attributes:
        kind: The failure this error reports.
    """

    def __init__(self, message: str, kind: OrchestrationErrors = OrchestrationErrors.MISSING_LOG_MANIFEST) -> None:
        super().__init__(message)
        self.kind = kind

    def __reduce__(self) -> tuple[Any, ...]:
        """Returns the class and arguments that rebuild this error, so its kind survives a process boundary."""
        return self.__class__, (str(self), self.kind)
