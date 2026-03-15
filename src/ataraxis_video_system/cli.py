"""Provides the Command Line Interface (CLI) installed into the Python environment together with the library."""

from typing import Literal  # pragma: no cover
from pathlib import Path  # pragma: no cover

import click  # pragma: no cover
import numpy as np  # pragma: no cover
from ataraxis_base_utilities import LogLevel, console  # pragma: no cover
from ataraxis_data_structures import DataLogger, assemble_log_archives  # pragma: no cover

from .saver import (
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)  # pragma: no cover
from .camera import (
    CameraInterfaces,
    HarvestersCamera,
    add_cti_file,
    check_cti_file,
    discover_camera_ids,
)  # pragma: no cover
from .mcp_server import run_server as run_mcp  # pragma: no cover
from .video_system import VideoSystem  # pragma: no cover
from .configuration import (
    GenicamConfiguration,
    read_genicam_node,
    format_genicam_node,
    enumerate_genicam_nodes,
)  # pragma: no cover

# Enables console output.
console.enable()  # pragma: no cover

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}  # pragma: no cover
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("axvs", context_settings=CONTEXT_SETTINGS)
def axvs_cli() -> None:  # pragma: no cover
    """Serves as the entry-point for interfacing with all interactive components of the ataraxis-video-system (AXVS)
    library.
    """


@axvs_cli.group("cti")
def cti_group() -> None:  # pragma: no cover
    """Allows working with the GenTL Producer interface (.cti) files."""


@cti_group.command("set")
@click.option(
    "-f",
    "--file-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help=(
        "The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the "
        "file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, is also acceptable. "
        "See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details."
    ),
)
def set_cti_file(file_path: Path) -> None:  # pragma: no cover
    """Configures the library to use the input CTI file for all future runtimes involving GeniCam cameras.

    This library relies on the Harvesters library to interface with GeniCam-compatible cameras. In turn, the Harvesters
    library requires the GenTL Producer interface (.cti) file to discover and interface with compatible cameras. This
    command must be called at least once before calling all other CLIs and APIs that rely on the Harvesters library.
    """
    add_cti_file(cti_path=file_path)

    # Notifies the user that the CTI file has been successfully set.
    console.echo(message=f"AXVS CTI file: Set to {file_path}.", level=LogLevel.SUCCESS)


@cti_group.command("check")
def check_cti_status() -> None:  # pragma: no cover
    """Checks whether the library is configured with a valid GenTL Producer interface (.cti) file.

    This command verifies if a .cti file has been configured and whether it is still valid. The Harvesters camera
    interface requires the GenTL Producer interface (.cti) file to discover and interface with GeniCam-compatible
    cameras. Use this command to verify the configuration status before attempting to use the Harvesters interface.
    """
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
def check_group() -> None:  # pragma: no cover
    """Allows discovering compatible camera devices and verifying host-system compatibility."""


@check_group.command("devices")
def check_devices() -> None:  # pragma: no cover
    """Discovers all cameras compatible with the library and prints their identification information.

    This command is primarily intended to be used during the initial system configuration to determine the positional
    indices of each camera in the list of all cameras discoverable by each supported interface. The discovered indices
    can then be used to initialize the VideoSystem instances to interface with the discovered cameras.
    """
    # Discovers all compatible cameras from both interfaces.
    all_cameras = discover_camera_ids()

    # Separates cameras by interface for display purposes.
    opencv_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.OPENCV]
    harvesters_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]

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
        for num, camera_data in enumerate(opencv_cameras, start=1):
            console.echo(
                message=(
                    f"OpenCV camera {num}: index={camera_data.camera_index}, "
                    f"frame_height={camera_data.frame_height} pixels, frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )

    # Displays Harvesters camera information.
    if not harvesters_cameras:
        console.echo(message="No Harvesters-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        # Note, Harvesters interface supports identifying the camera's model and serial number, which makes it easy to
        # map discovered indices to physical hardware.
        console.echo(message="Available Harvesters cameras:", level=LogLevel.SUCCESS)
        for num, camera_data in enumerate(harvesters_cameras, start=1):
            console.echo(
                message=(
                    f"Harvesters camera {num}: index={camera_data.camera_index}, model={camera_data.model}, "
                    f"serial_code={camera_data.serial_number}, frame_height={camera_data.frame_height} pixels, "
                    f"frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )


@check_group.command("compatibility")
def check_compatibility() -> None:  # pragma: no cover
    """Checks whether the host system meets the requirements for CPU and (optionally) GPU video encoding.

    This command allows checking whether the local system is set up correctly to support saving acquired camera frames
    as videos. As a minimum, this requires that the system has the FFMPEG library installed and available on the
    system's Path. Additionally, to support GPU (hardware) encoding, the system must have an Nvidia GPU. Note; the
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
    "'harvesters' interface for all GeniCam-compatible cameras and the 'opencv' interface for all other cameras.",
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
    help="Determines whether the camera records frames in monochrome (grayscale) or colored spectrum.",
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
) -> None:  # pragma: no cover
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

    # Initializes the VideoSystem.
    video_system = VideoSystem(
        system_id=np.uint8(111),
        data_logger=logger,
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
        quantization_parameter=15,  # Uses the instance's default parameter
    )

    # Starts the system by spawning child processes.
    video_system.start()
    console.echo(message="Live VideoSystem: initialized and started (spawned child processes).", level=LogLevel.INFO)

    # Ensures that manual control instructions are only shown once.
    show_instructions: bool = True
    # Uses terminal input to control the video system.
    while video_system.started:
        if show_instructions:
            message = (
                "Enter 'q' to terminate system's runtime. Enter 'w' to start saving camera frames. "
                "Enter 's' to stop saving camera frames. Note, after termination, the system may stay alive for up "
                "to 600 seconds to finish saving buffered frame data."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
            show_instructions = False

        key = input("\nEnter command key:")
        if key.lower() == "q":
            message = "Terminating the VideoSystem..."
            console.echo(message=message)
            video_system.stop()
            logger.stop()
        elif key.lower() == "w":  # pragma: no cover
            message = "VideoSystem's camera frame saving: Started."
            console.echo(message=message)
            video_system.start_frame_saving()
        elif key.lower() == "s":  # pragma: no cover
            message = "VideoSystem's camera frame saving: Stopped."
            console.echo(message=message)
            video_system.stop_frame_saving()
        else:  # pragma: no cover
            message = (
                f"Unknown input key {key.lower()} encountered while interacting with the VideoSystem. Use 'q' to "
                f"terminate the runtime, 'w' to start saving frames, and 's' to stop saving frames."
            )
            console.echo(message=message, level=LogLevel.WARNING)
    video_system.stop()
    logger.stop()
    console.echo(
        message=f"VideoSystem: Terminated. Saved frames (if any) are available from the {output_directory} directory.",
        level=LogLevel.SUCCESS,
    )
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=True)


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
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None:  # pragma: no cover
    """Starts the Model Context Protocol (MCP) server for agentic interaction with the library.

    The MCP server exposes camera discovery and CTI file management functionality through the MCP protocol, enabling
    AI agents to programmatically interact with the library.
    """
    console.echo(message=f"Starting AXVS MCP server with {transport} transport...", level=LogLevel.INFO)
    run_mcp(transport=transport)


@axvs_cli.group("configure")
def configure_group() -> None:  # pragma: no cover
    """Allows working with the configuration of the GenTL- (Harvesters)-compatible cameras."""


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
    help="The name of a specific GenICam node to read. If not provided, the interface lists all available nodes.",
)
def configuration_read(camera_index: int, node_name: str) -> None:  # pragma: no cover
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, displays detailed information about that specific node. Otherwise, lists all
    available nodes with their current values.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        if node_name:
            description = format_genicam_node(node_map=camera.node_map, name=node_name)
            console.echo(message=description, level=LogLevel.SUCCESS, raw=True)
        else:
            node_map = camera.node_map
            names = enumerate_genicam_nodes(node_map)
            console.echo(message=f"Found {len(names)} writable GenICam nodes:", level=LogLevel.SUCCESS)
            for name in names:
                try:
                    info = read_genicam_node(node_map=node_map, name=name)
                    console.echo(message=f"  {info.name} = {info.value}")
                except Exception:
                    console.echo(message=f"  {name} = <unreadable>")
    finally:
        camera.disconnect()


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
def configuration_write(camera_index: int, node_name: str, value: str) -> None:  # pragma: no cover
    """Writes a value to a GenICam node on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type (integer, float, boolean, or string)
    based on the node's type.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        camera.set_node_value(name=node_name, value=value)
        console.echo(message=f"Node '{node_name}' set to {value}.", level=LogLevel.SUCCESS)
    finally:
        camera.disconnect()


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
def configuration_dump(camera_index: int, output_file: Path) -> None:  # pragma: no cover
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    The output YAML includes all readable nodes with their current values, valid ranges, and enumeration entries,
    as well as the camera model and serial number for identity validation.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        config = camera.get_configuration()
        config.to_yaml(file_path=output_file)
        console.echo(
            message=f"Configuration saved: {len(config.nodes)} nodes written to {output_file}.",
            level=LogLevel.SUCCESS,
        )
    finally:
        camera.disconnect()


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
def configuration_load(camera_index: int, config_file: Path, *, strict: bool) -> None:  # pragma: no cover
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Applies all writable nodes from the configuration file to the camera. Optionally validates that the camera
    model and serial number match the configuration file.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        config = GenicamConfiguration.from_yaml(file_path=config_file)
        camera.apply_configuration(config, strict_identity=strict)
        console.echo(message="Configuration applied successfully.", level=LogLevel.SUCCESS)
    finally:
        camera.disconnect()
