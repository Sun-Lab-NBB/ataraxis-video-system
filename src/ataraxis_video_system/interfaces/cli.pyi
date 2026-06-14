from typing import Literal
from pathlib import Path

import click

from ..video import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    VideoSystem as VideoSystem,
    CameraInterfaces as CameraInterfaces,
    HarvestersCamera as HarvestersCamera,
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
    GenicamConfiguration as GenicamConfiguration,
    add_cti_file as add_cti_file,
    check_cti_file as check_cti_file,
    read_genicam_node as read_genicam_node,
    discover_camera_ids as discover_camera_ids,
    format_genicam_node as format_genicam_node,
    check_gpu_availability as check_gpu_availability,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
    check_ffmpeg_availability as check_ffmpeg_availability,
    run_log_processing_pipeline as run_log_processing_pipeline,
)

CONTEXT_SETTINGS: dict[str, int]

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
def process(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None,
    log_id: tuple[str, ...],
    *,
    workers: int,
    progress: bool,
) -> None: ...
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None: ...
@click.pass_context
def configure_group(context: click.Context, blacklisted_node: tuple[str, ...], *, no_blacklist: bool) -> None: ...
@click.pass_context
def configuration_read(context: click.Context, camera_index: int, node_name: str) -> None: ...
def configuration_write(camera_index: int, node_name: str, value: str) -> None: ...
@click.pass_context
def configuration_dump(context: click.Context, camera_index: int, output_file: Path) -> None: ...
@click.pass_context
def configuration_load(context: click.Context, camera_index: int, config_file: Path, *, strict: bool) -> None: ...
