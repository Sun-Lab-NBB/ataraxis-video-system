from typing import Literal
from pathlib import Path

import click

from .saver import (
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
from .video_system import VideoSystem as VideoSystem
from .configuration import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration as GenicamConfiguration,
    read_genicam_node as read_genicam_node,
    format_genicam_node as format_genicam_node,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
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
