"""Provides the Command Line Interface (CLI) installed into the Python environment together with the library."""

from typing import Literal
from pathlib import Path

import click
import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, assemble_log_archives

from ..video import (
    DEFAULT_BLACKLISTED_NODES,
    GENICAM_UNAVAILABLE_REASON,
    VideoSystem,
    CameraInterfaces,
    OutputPixelFormats,
    EncoderSpeedPresets,
    GenicamConfiguration,
    add_cti_file,
    check_cti_file,
    read_genicam_node,
    discover_camera_ids,
    format_genicam_node,
    harvester_connection,
    check_gpu_availability,
    enumerate_genicam_nodes,
    check_ffmpeg_availability,
    genicam_runtime_available,
)
from .mcp_server import run_server as run_mcp
from ..orchestration import run_log_processing_pipeline

console.enable()

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("axvs", context_settings=_CONTEXT_SETTINGS)
def axvs_cli() -> None:
    """Serves as the entry-point for interfacing with all interactive components of the ataraxis-video-system (AXVS)
    library.
    """


@axvs_cli.group("cti")
def cti_group() -> None:
    """Allows working with the GenTL Producer interface (.cti) files."""


@cti_group.command("set")
@click.option(
    "-f",
    "--file-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help=(
        "The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the "
        "file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, is also acceptable. "
        "See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details."
    ),
)
def set_cti_file(file_path: Path) -> None:
    """Configures the library to use the input CTI file for all future runtimes involving GenICam cameras.

    This library relies on the Harvesters library to interface with GenICam-compatible cameras. In turn, the Harvesters
    library requires the GenTL Producer interface (.cti) file to discover and interface with compatible cameras. This
    command must be called at least once before calling all other CLIs and APIs that rely on the Harvesters library.
    """
    add_cti_file(cti_path=file_path)

    # Notifies the user that the CTI file has been successfully set.
    console.echo(message=f"AXVS CTI file: Set to {file_path}.", level=LogLevel.SUCCESS)


@cti_group.command("check")
def check_cti_status() -> None:
    """Checks whether the library is configured with a valid GenTL Producer interface (.cti) file.

    This command verifies if a .cti file has been configured and whether it is still valid. The Harvesters camera
    interface requires the GenTL Producer interface (.cti) file to discover and interface with GenICam-compatible
    cameras. Use this command to verify the configuration status before attempting to use the Harvesters interface.
    """
    # Reports the unsupported platform first, since check_cti_file() is unable to validate a Producer without the
    # runtime that loads it and would otherwise blame the configuration for a platform limitation.
    if not genicam_runtime_available():
        console.echo(
            message=f"AXVS CTI file: Unable to check. {GENICAM_UNAVAILABLE_REASON}",
            level=LogLevel.ERROR,
        )
        return

    cti_path = check_cti_file()

    if cti_path is not None:
        console.echo(message=f"AXVS CTI file: Configured and valid. Path: {cti_path}", level=LogLevel.SUCCESS)
    else:
        console.echo(
            message=(
                "AXVS CTI file: Not configured or invalid. Use the 'axvs cti set -f <path>' command to configure the "
                "library to use a GenTL Producer interface (.cti) file."
            ),
            level=LogLevel.ERROR,
        )


@axvs_cli.group("check")
def check_group() -> None:
    """Allows discovering compatible camera devices and verifying host-system compatibility."""


@check_group.command("devices")
def check_devices() -> None:
    """Discovers all cameras compatible with the library and prints their identification information.

    This command is primarily intended to be used during the initial system configuration to determine the positional
    indices of each camera in the list of all cameras discoverable by each supported interface. The discovered indices
    can then be used to initialize the VideoSystem instances to interface with the discovered cameras.
    """
    # Notifies the user that discovery is in progress, as probing camera devices may take several seconds.
    console.echo(message="Scanning for available camera devices, this may take a moment...", level=LogLevel.INFO)

    all_cameras = discover_camera_ids()

    # Separates cameras by interface for display purposes.
    opencv_cameras = [camera for camera in all_cameras if camera.interface == CameraInterfaces.OPENCV]
    harvesters_cameras = [camera for camera in all_cameras if camera.interface == CameraInterfaces.HARVESTERS]

    # Displays OpenCV camera information.
    if not opencv_cameras:
        console.echo(message="No OpenCV-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        console.echo(
            message=(
                "Warning! Currently, it is impossible to resolve camera models or serial numbers through the "
                "OpenCV interface. It is recommended to check each discovered OpenCV camera via the 'axvs run' "
                "CLI command to precisely map the discovered camera indices to specific camera hardware."
            ),
            level=LogLevel.WARNING,
        )
        console.echo(message="Available OpenCV cameras:", level=LogLevel.SUCCESS)
        for number, camera_data in enumerate(opencv_cameras, start=1):
            console.echo(
                message=(
                    f"OpenCV camera {number}: index={camera_data.camera_index}, "
                    f"frame_height={camera_data.frame_height} pixels, frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )

    # Displays Harvesters camera information.
    if not genicam_runtime_available():
        console.echo(
            message=f"Harvesters camera discovery skipped. {GENICAM_UNAVAILABLE_REASON}",
            level=LogLevel.WARNING,
        )
    elif not harvesters_cameras:
        console.echo(message="No Harvesters-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        # Note, Harvesters interface supports identifying the camera's model and serial number, which makes it easy to
        # map discovered indices to physical hardware.
        console.echo(message="Available Harvesters cameras:", level=LogLevel.SUCCESS)
        for number, camera_data in enumerate(harvesters_cameras, start=1):
            console.echo(
                message=(
                    f"Harvesters camera {number}: index={camera_data.camera_index}, model={camera_data.model}, "
                    f"serial_code={camera_data.serial_number}, frame_height={camera_data.frame_height} pixels, "
                    f"frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )


@check_group.command("compatibility")
def check_compatibility() -> None:
    """Checks whether the host system meets the requirements for CPU and (optionally) GPU video encoding.

    This command allows checking whether the local system is set up correctly to support saving acquired camera frames
    as videos. As a minimum, this requires that the system has the FFMPEG library installed and available on the
    system's Path. Additionally, to support GPU (hardware) encoding, the system must have an Nvidia GPU. Note, the
    presence of the GPU is evaluated by calling the 'nvidia-smi' command, so it must also be installed on the local
    system alongside the GPU for the check to work as expected.
    """
    if not check_ffmpeg_availability():
        console.echo(
            message="Video saving requirements: Not met. Unable to access the FFMPEG library.", level=LogLevel.ERROR
        )
    elif not check_gpu_availability():
        console.echo(
            message=(
                "Video saving requirements: Partially met. The local system supports CPU video encoding via the "
                "FFMPEG library, but does not have an Nvidia GPU for GPU encoding."
            ),
            level=LogLevel.WARNING,
        )
    else:
        console.echo(
            message="Video saving requirements: Fully met. The system supports both CPU and GPU video encoding.",
            level=LogLevel.SUCCESS,
        )


@axvs_cli.command("run")
@click.option(
    "-i",
    "--interface",
    type=click.Choice(["mock", "harvesters", "opencv"]),
    default="mock",
    show_default=True,
    help="The camera interface to use for interacting with the camera hardware. It is recommended to use the "
    "'harvesters' interface for all GenICam-compatible cameras and the 'opencv' interface for all other cameras.",
)
@click.option(
    "-c",
    "--camera-index",
    type=int,
    default=0,
    show_default=True,
    help="The index of the target camera in the list of all cameras discoverable through the chosen interface. This "
    "option allows selecting the desired camera if multiple are available on the host-system.",
)
@click.option(
    "-g",
    "--gpu-index",
    type=int,
    default=-1,
    show_default=True,
    help="The index of the GPU device to use for video encoding. Setting this option to a value below zero (default) "
    "forces the VideoSystem to use the CPU for encoding the videos. Note; GPU encoding currently requires an "
    "Nvidia GPU that supports hardware video encoding.",
)
@click.option(
    "-o",
    "--output-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    help="The path to the output directory where to save the acquired camera frames as an .mp4 video file.",
)
@click.option(
    "-m",
    "--monochrome",
    is_flag=True,
    default=False,
    show_default=True,
    help="Determines whether the camera records frames in monochrome (grayscale) or colored spectrum. Applies to the "
    "'opencv' and 'mock' interfaces only, as the 'harvesters' interface takes the color mode from the camera's own "
    "configuration.",
)
@click.option(
    "-w",
    "--width",
    type=int,
    default=600,
    show_default=True,
    help="The width of the camera frames to acquire, in pixels.",
)
@click.option(
    "-h",
    "--height",
    type=int,
    default=400,
    show_default=True,
    help="The height of the camera frames to acquire, in pixels.",
)
@click.option(
    "-f",
    "--frame-rate",
    type=int,
    default=30,
    show_default=True,
    help="The rate at which to acquire the frames, in frames per second.",
)
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
) -> None:
    """Creates a VideoSystem instance using the input parameters and starts an interactive imaging session.

    This command allows testing various components of the VideoSystem by running an interactive session controlled via
    the terminal. Primarily, this CLI is designed to help with the initial identification and calibration of VideoSystem
    instances and does not support the full range of features offered through the VideoSystem class API.
    """
    # Initializes and starts the DataLogger instance.
    logger = DataLogger(output_directory=output_directory, instance_name="axvs_live_run")
    logger.start()

    # Uses command arguments to resolve VideoSystem configuration parameters.
    if interface == "mock":
        camera_interface = CameraInterfaces.MOCK
    elif interface == "harvesters":
        camera_interface = CameraInterfaces.HARVESTERS
    else:
        camera_interface = CameraInterfaces.OPENCV

    video_system = VideoSystem(
        system_id=np.uint8(111),
        data_logger=logger,
        name="live_camera",
        output_directory=output_directory,
        camera_interface=camera_interface,
        camera_index=camera_index,
        frame_width=width,
        frame_height=height,
        frame_rate=frame_rate,
        display_frame_rate=25,  # Statically sets the display rate to 25 fps.
        color=not monochrome,
        gpu=gpu_index,
        video_encoder="H264",  # Older H264 codec for compatibility with older hardware.
        encoder_speed_preset=EncoderSpeedPresets.FAST,  # Faster encoding speed for compatibility with older hardware.
        output_pixel_format=OutputPixelFormats.YUV420,  # Half-width chroma coding.
        quantization_parameter=15,  # Statically sets the H264 quantization parameter to 15.
    )

    video_system.start()
    console.echo(message="Live VideoSystem: initialized and started (spawned child processes).", level=LogLevel.INFO)

    # Ensures that manual control instructions are only shown once.
    show_instructions: bool = True

    try:
        # Uses terminal input to control the video system.
        while video_system.started:
            if show_instructions:
                message = (
                    "Enter 'q' to terminate system's runtime. Enter 'w' to start saving camera frames. "
                    "Enter 's' to stop saving camera frames. Note, after termination, the system may stay alive for "
                    "up to 600 seconds to finish saving buffered frame data."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
                show_instructions = False

            key = input("\nEnter command key:")
            if key.lower() == "q":
                message = "Terminating the VideoSystem..."
                console.echo(message=message)
                video_system.stop()
                logger.stop()
            elif key.lower() == "w":
                message = "VideoSystem's camera frame saving: Started."
                console.echo(message=message)
                video_system.start_frame_saving()
            elif key.lower() == "s":
                message = "VideoSystem's camera frame saving: Stopped."
                console.echo(message=message)
                video_system.stop_frame_saving()
            else:
                message = (
                    f"Unknown input key {key.lower()} encountered while interacting with the VideoSystem. Use 'q' to "
                    f"terminate the runtime, 'w' to start saving frames, and 's' to stop saving frames."
                )
                console.echo(message=message, level=LogLevel.WARNING)
    finally:
        # Finalizes the session on every exit path, including the Abort that Click raises when the operator interrupts
        # the prompt above. Both stop calls are idempotent, so the 'q' branch does not conflict with them, and the
        # archive assembly is the step a later 'axvs process' invocation depends on.
        video_system.stop()
        logger.stop()
        console.echo(
            message=(
                f"VideoSystem: Terminated. Saved frames (if any) are available from the {output_directory} directory."
            ),
            level=LogLevel.SUCCESS,
        )
        assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=True)


@axvs_cli.command("process")
@click.option(
    "-ld",
    "--log-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    help="The path to the root directory to search for .npz log archives. Searched recursively.",
)
@click.option(
    "-od",
    "--output-directory",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="The root path under which processed output files are written. A camera_timestamps/ subdirectory is created "
    "automatically beneath it and holds the processing tracker and every output file.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help="The canonical hexadecimal identifier of the single job to run. If provided, runs only the matching job, "
    "which is the target an external scheduler names when it dispatches one unit of work.",
)
@click.option(
    "-s",
    "--specifier",
    type=str,
    multiple=True,
    help="Camera source ID to process. Repeat to specify multiple IDs. If not provided, resolves all source IDs from "
    "the camera_manifest.yaml file in the log directory. Ignored when a job ID selects the work.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="The ceiling on the worker processes any single job receives. Every job runs at this ceiling, which is "
    "itself bounded by the stage's own core cap, except that an archive below the parallel processing threshold is "
    "processed sequentially. Set to a value below 1 (default -1) to resolve the ceiling from the host.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether to display progress bars during timestamp extraction.",
)
def process(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None,
    specifier: tuple[str, ...],
    *,
    workers: int,
    progress: bool,
) -> None:
    """Processes the VideoSystem log archives of one recording to extract frame timestamps.

    Functions as the entry point for processing the data stored in the .npz log archives generated by VideoSystem
    instances during runtime. Targets a single recording and runs its archives one at a time. Each specified source
    ID must be registered in the recording's camera manifest and correspond to exactly one archive. Passing a job ID
    runs that single job alone, which is how an external scheduler dispatches one unit of work. Use the MCP server to
    orchestrate batches spanning many recordings.
    """
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        job_id=job_id,
        source_ids=list(specifier) if specifier else None,
        workers=workers,
        display_progress=progress,
    )


@axvs_cli.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    show_default=True,
    help="The transport protocol to use for MCP communication. Use 'stdio' for standard input/output communication "
    "(default, recommended for Claude Desktop integration) or 'streamable-http' for HTTP-based communication.",
)
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None:
    """Starts the Model Context Protocol (MCP) server for agentic interaction with the library.

    The MCP server exposes camera discovery, CTI file management, video session control, GenICam configuration,
    camera manifest management, and log processing functionality through the MCP protocol, enabling AI agents to
    programmatically interact with the library.
    """
    # The stdio transport carries the JSON-RPC message stream over stdout, which is also where the console writes
    # every message up to the WARNING level. Silencing the console keeps library output out of that stream, as a
    # single logged line renders the message it interleaves with unparsable for the connected client.
    if transport == "stdio":
        console.disable()
    else:
        console.echo(message=f"Starting AXVS MCP server with {transport} transport...", level=LogLevel.INFO)

    run_mcp(transport=transport)


@axvs_cli.group("configure")
@click.option(
    "-b",
    "--blacklisted-node",
    type=str,
    multiple=True,
    default=sorted(DEFAULT_BLACKLISTED_NODES),
    show_default=True,
    help="GenICam node name to exclude from the read, dump, and load operations. Repeat to specify multiple nodes. "
    "Some vendor-specific nodes report ReadWrite access but reject writes at the hardware level. Modify this list to "
    "match your camera hardware. An explicitly named node passed to 'configure write' is always written. Use "
    "--no-blacklist to disable all blacklisting.",
)
@click.option(
    "--no-blacklist",
    is_flag=True,
    default=False,
    help="Disables all node blacklisting. When set, all ReadWrite nodes are included in the read, dump, and load "
    "operations regardless of the --blacklisted-node values.",
)
@click.pass_context
def configure_group(context: click.Context, blacklisted_node: tuple[str, ...], *, no_blacklist: bool) -> None:
    """Allows working with the configuration of the GenTL- (Harvesters)-compatible cameras."""
    context.ensure_object(dict)
    context.obj["blacklisted_nodes"] = frozenset() if no_blacklist else frozenset(blacklisted_node)


@configure_group.command("read")
@click.option(
    "-c",
    "--camera-index",
    type=int,
    default=0,
    show_default=True,
    help="The index of the Harvesters camera to read the configuration from.",
)
@click.option(
    "-n",
    "--node-name",
    type=str,
    default="",
    help="The name of a specific GenICam node to read. If omitted, lists every writable (ReadWrite) node that is not "
    "blacklisted.",
)
@click.pass_context
def configuration_read(context: click.Context, camera_index: int, node_name: str) -> None:
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, displays detailed information about that specific node. Otherwise, lists every
    writable (ReadWrite) node that is not blacklisted, with its current value.
    """
    blacklist: frozenset[str] = context.obj["blacklisted_nodes"]

    with harvester_connection(camera_index=camera_index) as camera:
        if node_name:
            description = format_genicam_node(node_map=camera.node_map, name=node_name)
            console.echo(message=description, level=LogLevel.SUCCESS, raw=True)
        else:
            node_map = camera.node_map
            names = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=blacklist)
            console.echo(message=f"Found {len(names)} writable GenICam nodes:", level=LogLevel.SUCCESS)
            for name in names:
                try:
                    info = read_genicam_node(node_map=node_map, name=name)
                    console.echo(message=f"  {info.name} = {info.value}")
                except Exception:
                    console.echo(message=f"  {name} = <unreadable>")


@configure_group.command("write")
@click.option(
    "-c",
    "--camera-index",
    type=int,
    default=0,
    show_default=True,
    help="The index of the Harvesters camera to write the configuration to.",
)
@click.option(
    "-n",
    "--node-name",
    type=str,
    required=True,
    help="The name of the GenICam node to write.",
)
@click.option(
    "-v",
    "--value",
    type=str,
    required=True,
    help="The value to write to the node. The value is automatically converted to the type expected by the node.",
)
def configuration_write(camera_index: int, node_name: str, value: str) -> None:
    """Writes a value to a GenICam node on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type (integer, float, boolean, or string)
    based on the node's type.
    """
    with harvester_connection(camera_index=camera_index) as camera:
        camera.set_node_value(name=node_name, value=value)
        console.echo(message=f"Node '{node_name}' set to {value}.", level=LogLevel.SUCCESS)


@configure_group.command("dump")
@click.option(
    "-c",
    "--camera-index",
    type=int,
    default=0,
    show_default=True,
    help="The index of the Harvesters camera to dump the configuration from.",
)
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="The path to the output YAML file to write the configuration to.",
)
@click.pass_context
def configuration_dump(context: click.Context, camera_index: int, output_file: Path) -> None:
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    The output YAML includes every writable (ReadWrite) node that is not blacklisted, with its current value, as well
    as the camera model and serial number for identity validation.
    """
    blacklist: frozenset[str] = context.obj["blacklisted_nodes"]

    with harvester_connection(camera_index=camera_index) as camera:
        config = camera.get_configuration(blacklisted_nodes=blacklist)
        config.to_yaml(file_path=output_file)
        console.echo(
            message=f"Configuration saved: {len(config.nodes)} nodes written to {output_file}.",
            level=LogLevel.SUCCESS,
        )


@configure_group.command("load")
@click.option(
    "-c",
    "--camera-index",
    type=int,
    default=0,
    show_default=True,
    help="The index of the Harvesters camera to load the configuration onto.",
)
@click.option(
    "-f",
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="The path to the YAML configuration file to load.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, aborts the operation when a camera identity mismatch is detected between the configuration file "
    "and the connected camera.",
)
@click.pass_context
def configuration_load(context: click.Context, camera_index: int, config_file: Path, *, strict: bool) -> None:
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Applies every non-blacklisted writable node from the configuration file to the camera. Optionally validates that
    the camera model and serial number match the configuration file.
    """
    blacklist: frozenset[str] = context.obj["blacklisted_nodes"]

    with harvester_connection(camera_index=camera_index) as camera:
        config = GenicamConfiguration.from_yaml(file_path=config_file)
        camera.apply_configuration(config=config, strict_identity=strict, blacklisted_nodes=blacklist)
        console.echo(message="Configuration applied successfully.", level=LogLevel.SUCCESS)
