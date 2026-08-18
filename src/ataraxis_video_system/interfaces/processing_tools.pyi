from typing import Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..video import ExtractedDataColumns as ExtractedDataColumns
from .responses import (
    page_fields as page_fields,
    project_item as project_item,
    resolve_page as resolve_page,
    item_breakdown as item_breakdown,
    reject_unknown as reject_unknown,
    resolve_detail_limit as resolve_detail_limit,
)
from .mcp_instance import mcp as mcp
from ..orchestration import (
    JobSizing as JobSizing,
    OutputLayout as OutputLayout,
    JobDescriptor as JobDescriptor,
    ArchiveFootprint as ArchiveFootprint,
    JobExecutionState as JobExecutionState,
    size_job as size_job,
    prepare_jobs as prepare_jobs,
    resolve_pool_size as resolve_pool_size,
    session_is_active as session_is_active,
    get_execution_state as get_execution_state,
    resolve_core_budget as resolve_core_budget,
    resolve_job_workers as resolve_job_workers,
    group_jobs_by_tracker as group_jobs_by_tracker,
    estimate_job_memory_mb as estimate_job_memory_mb,
    start_execution_session as start_execution_session,
    finish_execution_session as finish_execution_session,
    resolve_memory_budget_mb as resolve_memory_budget_mb,
)

_REQUIRED_JOB_KEYS: tuple[str, ...]
_OVERVIEW_AXES: tuple[str, ...]
_OVERVIEW_SEMI_DETAIL_FIELDS: tuple[str, ...]
_OVERVIEW_DETAIL_FIELDS: tuple[str, ...]
_AUTO_DROP_THRESHOLD_MULTIPLIER: float
type TimestampArray = NDArray[np.uint64] | NDArray[np.int64]

def prepare_log_processing_batch_tool(
    log_directories: list[str], source_ids: list[str], output_directories: list[str]
) -> dict[str, Any]: ...
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, Any]], *, core_budget: int = -1, memory_budget_mb: int = -1
) -> dict[str, Any]: ...
def get_log_processing_status_tool() -> dict[str, Any]: ...
def get_log_processing_timing_tool() -> dict[str, Any]: ...
def cancel_log_processing_tool() -> dict[str, Any]: ...
def reset_log_processing_jobs_tool(tracker_path: str, source_ids: list[str] | None = None) -> dict[str, Any]: ...
def get_batch_status_overview_tool(
    root_directory: str,
    statuses: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def analyze_camera_frame_statistics_tool(
    feather_files: list[str], drop_threshold_us: int = 0, max_drop_locations: int = 50
) -> dict[str, Any]: ...
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]: ...
def _read_tracker_status(tracker_path: Path) -> dict[str, Any]: ...
def _analyze_single_feather(feather_file: str, drop_threshold_us: int, max_drop_locations: int) -> dict[str, Any]: ...
def _read_feather_timestamps(feather_file: str) -> tuple[TimestampArray | None, str | None]: ...
def _compute_frame_statistics(
    timestamps: TimestampArray, drop_threshold_us: int, max_drop_locations: int
) -> dict[str, Any]: ...
def _estimate_dropped_frames(
    intervals_us: NDArray[np.int64], gap_indices: NDArray[np.intp], expected_interval: float
) -> NDArray[np.int64]: ...
def _clean_single_output(output_directory: str) -> dict[str, Any]: ...
