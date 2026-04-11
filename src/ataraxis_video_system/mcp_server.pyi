from typing import Any, Literal
from pathlib import Path
from threading import Lock, Thread
import contextlib
from dataclasses import field, dataclass
from collections.abc import Generator

from mcp.server.fastmcp import FastMCP
from ataraxis_data_structures import DataLogger

from .saver import (
    VideoEncoders as VideoEncoders,
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
    check_gpu_availability as check_gpu_availability,
    check_ffmpeg_availability as check_ffmpeg_availability,
)
from .camera import (
    CameraInterfaces as CameraInterfaces,
    HarvestersCamera as HarvestersCamera,
    add_cti_file as add_cti_file,
    check_cti_file as check_cti_file,
    discover_camera_ids as discover_camera_ids,
)
from .manifest import (
    CAMERA_MANIFEST_FILENAME as CAMERA_MANIFEST_FILENAME,
    CameraManifest as CameraManifest,
    write_camera_manifest as write_camera_manifest,
)
from .video_system import (
    MAXIMUM_QUANTIZATION_VALUE as MAXIMUM_QUANTIZATION_VALUE,
    VideoSystem as VideoSystem,
)
from .configuration import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration as GenicamConfiguration,
    format_genicam_node as format_genicam_node,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
)
from .log_processing import (
    TRACKER_FILENAME as TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    TIMESTAMP_JOB_NAME as TIMESTAMP_JOB_NAME,
    CAMERA_TIMESTAMPS_DIRECTORY as CAMERA_TIMESTAMPS_DIRECTORY,
    PARALLEL_PROCESSING_THRESHOLD as PARALLEL_PROCESSING_THRESHOLD,
    execute_job as execute_job,
    prepare_tracker as prepare_tracker,
    find_log_archive as find_log_archive,
    generate_job_ids as generate_job_ids,
    resolve_recording_roots as resolve_recording_roots,
)

mcp: FastMCP
_WORKER_SCALING_FACTOR: int
_WORKER_MULTIPLE: int
_RESERVED_CORES: int
_FEATHER_PREFIX: str
_FEATHER_SUFFIX: str
_active_session: VideoSystem | None
_active_logger: DataLogger | None
_session_info: dict[str, Any] | None

@dataclass(slots=True)
class _PendingJob:
    log_directory: Path
    output_directory: Path
    tracker_path: Path
    job_id: str
    source_id: str
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

@dataclass(slots=True)
class _ActiveGroup:
    source_id: str
    jobs: list[_PendingJob]
    workers: int
    thread: Thread

@dataclass(slots=True)
class _JobExecutionState:
    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    pending_queue: list[_PendingJob] = field(default_factory=list)
    active_groups: list[_ActiveGroup] = field(default_factory=list)
    job_message_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    worker_budget: int = ...
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...
    canceled: bool = ...

_job_execution_state: _JobExecutionState | None

def list_cameras() -> str: ...
def get_cti_status() -> str: ...
def set_cti_file(file_path: str) -> str: ...
def check_runtime_requirements() -> str: ...
def start_video_session(
    output_directory: str,
    interface: str = "opencv",
    camera_index: int = 0,
    width: int = 600,
    height: int = 400,
    frame_rate: int = 30,
    gpu_index: int = -1,
    display_frame_rate: int | None = 25,
    *,
    monochrome: bool = False,
    video_encoder: str = "H264",
    encoder_speed_preset: int = 3,
    output_pixel_format: str = "yuv420p",
    quantization_parameter: int = 15,
) -> str: ...
def stop_video_session() -> dict[str, Any]: ...
def assemble_log_archives_tool(
    log_directory: str, *, remove_sources: bool = True, verify_integrity: bool = False
) -> dict[str, Any]: ...
def start_frame_saving() -> str: ...
def stop_frame_saving() -> str: ...
def get_session_status() -> dict[str, Any]: ...
def validate_video_file_tool(video_file: str) -> dict[str, Any]: ...
def read_genicam_node(
    camera_index: int = 0, node_name: str = "", blacklisted_nodes: list[str] | None = None
) -> str: ...
def write_genicam_node(camera_index: int, node_name: str, value: str) -> str: ...
def dump_genicam_config(camera_index: int, output_file: str, blacklisted_nodes: list[str] | None = None) -> str: ...
def load_genicam_config(
    camera_index: int, config_file: str, *, strict_identity: bool = False, blacklisted_nodes: list[str] | None = None
) -> str: ...
def read_camera_manifest_tool(manifest_path: str) -> dict[str, Any]: ...
def write_camera_manifest_tool(log_directory: str, source_id: int, name: str) -> dict[str, Any]: ...
def discover_camera_data_tool(root_directory: str) -> dict[str, Any]: ...
def prepare_log_processing_batch_tool(
    log_directories: list[str], source_ids: list[str], output_directories: list[str]
) -> dict[str, Any]: ...
def execute_log_processing_jobs_tool(jobs: list[dict[str, str]], *, worker_budget: int = -1) -> dict[str, Any]: ...
def get_log_processing_status_tool() -> dict[str, Any]: ...
def get_log_processing_timing_tool() -> dict[str, Any]: ...
def cancel_log_processing_tool() -> dict[str, Any]: ...
def reset_log_processing_jobs_tool(tracker_path: str, source_ids: list[str] | None = None) -> dict[str, Any]: ...
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]: ...
def analyze_camera_frame_statistics_tool(
    feather_files: list[str], drop_threshold_us: int = 0, max_drop_locations: int = 50
) -> dict[str, Any]: ...
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]: ...
def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
def _resolve_blacklist(blacklisted_nodes: list[str] | None) -> frozenset[str]: ...
@contextlib.contextmanager
def _harvester_connection(camera_index: int) -> Generator[HarvestersCamera, None, None]: ...
def _scan_archive_source_ids(directory: Path) -> list[str]: ...
def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]: ...
def _match_video_file(
    all_video_files: tuple[Path, ...], log_directory: Path, source_id: int, name: str
) -> str | None: ...
def _find_feather_file(timestamps_dirs: tuple[Path, ...], source_id: int) -> Path | None: ...
def _derive_tracker_status(summary: dict[str, Any]) -> str: ...
def _group_jobs_by_tracker(state: _JobExecutionState) -> dict[Path, list[_PendingJob]]: ...
def _read_tracker_status(tracker_path: Path) -> dict[str, Any]: ...
def _analyze_single_feather(feather_file: str, drop_threshold_us: int, max_drop_locations: int) -> dict[str, Any]: ...
def _clean_single_output(output_directory: str) -> dict[str, Any]: ...
def _probe_archive_message_count(job: _PendingJob) -> int: ...
def _compute_sqrt_minimum(message_count: int) -> int: ...
def _group_worker(jobs: list[_PendingJob], workers: int, state: _JobExecutionState) -> None: ...
def _job_execution_manager() -> None: ...
