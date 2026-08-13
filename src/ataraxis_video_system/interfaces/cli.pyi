from typing import Literal
from pathlib import Path
from dataclasses import dataclass

import click
from _typeshed import Incomplete

from ..video import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GENICAM_UNAVAILABLE_REASON as GENICAM_UNAVAILABLE_REASON,
    VideoSystem as VideoSystem,
    CameraInterfaces as CameraInterfaces,
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
    GenicamConfiguration as GenicamConfiguration,
    add_cti_file as add_cti_file,
    check_cti_file as check_cti_file,
    read_genicam_node as read_genicam_node,
    discover_camera_ids as discover_camera_ids,
    format_genicam_node as format_genicam_node,
    harvester_connection as harvester_connection,
    check_gpu_availability as check_gpu_availability,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
    check_ffmpeg_availability as check_ffmpeg_availability,
    genicam_runtime_available as genicam_runtime_available,
)
from ..orchestration import run_log_processing_pipeline as run_log_processing_pipeline

_CONTEXT_SETTINGS: dict[str, int]

@dataclass(frozen=True, slots=True)
class _SharedConfigurationParameters:
    camera_index: int | None
    blacklisted_nodes: frozenset[str]
    def require_camera_index(self) -> int: ...

_pass_shared_parameters: Incomplete

def axvs_cli() -> None: ...
def cti_group() -> None: ...
def set_cti_file(file_path: Path) -> None: ...
def check_cti_status() -> None: ...
def check_group() -> None: ...
def check_devices() -> None: ...
def check_compatibility() -> None: ...
def live_run(
    interface: str,
    camera_index: int,
    gpu_index: int,
    output_directory: Path,
    width: int,
    height: int,
    frame_rate: int,
    *,
    monochrome: bool,
) -> None: ...
def process_log_archives(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None,
    specifier: tuple[str, ...],
    *,
    workers: int,
    no_progress: bool,
) -> None: ...
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None: ...
@click.pass_context
def configure_group(
    context: click.Context, camera_index: int | None, blacklisted_node: tuple[str, ...], *, no_blacklist: bool
) -> None: ...
@_pass_shared_parameters
def read_genicam_configuration(shared: _SharedConfigurationParameters, node_name: str) -> None: ...
@_pass_shared_parameters
def write_genicam_configuration(shared: _SharedConfigurationParameters, node_name: str, value: str) -> None: ...
@_pass_shared_parameters
def dump_genicam_configuration(shared: _SharedConfigurationParameters, output_file: Path) -> None: ...
@_pass_shared_parameters
def load_genicam_configuration(shared: _SharedConfigurationParameters, config_file: Path, *, strict: bool) -> None: ...
