"""Provides the Command Line Interface (CLI) installed into the Python environment together with the library."""

from typing import Literal
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

import click
import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, assemble_log_archives

from ..video import (
    DEFAULT_BLACKLISTED_NODES,
    GENICAM_UNAVAILABLE_REASON,
    VideoSystem,
    CameraInterfaces,
    HarvestersCamera,
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


@dataclass(frozen=True, slots=True)
class _SharedConfigurationParameters:
    """Bundles the options parsed on the ``configure`` group and shared across its ``read``, ``write``, ``dump``, and
    ``load`` subcommands.

    The group callback builds one of these from its options and stores it on the Click context, and each subcommand
    reads it back through the ``_pass_shared_parameters`` decorator.
    """

    camera_index: int | None
    """The index of the Harvesters camera every subcommand operates on, or None when the option was omitted."""

    blacklisted_nodes: frozenset[str]
    """The GenICam node names excluded from the read, dump, and load operations."""

    def require_camera_index(self) -> int:
        """Returns the index of the target camera, raising a Click usage error when ``--camera-index`` was not
        supplied.

        The option cannot be marked required on the group without also blocking ``axvs configure SUBCOMMAND --help``,
        so each subcommand enforces it through this accessor when it actually runs.
        """
        if self.camera_index is None:
            message = (
                "Unable to resolve the target camera for the 'configure' command. The '-c' / '--camera-index' option "
                "must be supplied before the subcommand name, but it was omitted."
            )
            console.error(message=message, error=click.UsageError)
        return self.camera_index


_pass_shared_parameters = click.make_pass_decorator(_SharedConfigurationParameters)
"""Injects the ``configure`` group's ``_SharedConfigurationParameters`` as each subcommand's first argument."""


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
    command must be called at least once before calling all other CLIs and APIs that rely on the Harvesters library,
    unless the AXVS_CTI_PATH environment variable supplies the Producer path for the runtime.
    """
    add_cti_file(cti_path=file_path)

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
    "forces the VideoSystem to use the CPU for encoding the videos. Note, GPU encoding currently requires an "
    "Nvidia GPU that supports hardware video encoding.",
)
@click.option(
    "-o",
    "--output-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    help=(
        "The path to the output directory where to save the acquired camera frames as an .mp4 video file. The frame "
        "acquisition timestamp logs, the camera manifest, and the assembled .npz archives are written to an "
        "'axvs_live_run_data_log' subdirectory beneath it, which is the path to pass to 'axvs process "
        "--log-directory'."
    ),
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
    logger = DataLogger(output_directory=output_directory, instance_name="axvs_live_run")
    logger.start()

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
        display_frame_rate=25,
        color=not monochrome,
        gpu=gpu_index,
        video_encoder="H264",  # Older H264 codec for compatibility with older hardware.
        encoder_speed_preset=EncoderSpeedPresets.FAST,  # Faster encoding speed for compatibility with older hardware.
        output_pixel_format=OutputPixelFormats.YUV420,  # Half-width chroma coding.
        quantization_parameter=15,
    )

    video_system.start()
    console.echo(message="Live VideoSystem: initialized and started (spawned child processes).", level=LogLevel.INFO)

    # Ensures that manual control instructions are only shown once.
    show_instructions: bool = True

    try:
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
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path),
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
    help="The worker processes each job receives. Set to -1 (default) to resolve the width from the archive's "
    "message count, which yields a single worker for a small archive and the declared per-job allocation of 8 "
    "cores for a large one.",
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    default=False,
    show_default=True,
    help="Determines whether to suppress the progress bars during timestamp extraction. The progress bars are "
    "displayed by default.",
)
def process_log_archives(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None,
    specifier: tuple[str, ...],
    *,
    workers: int,
    no_progress: bool,
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
        display_progress=not no_progress,
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

    The MCP server exposes camera discovery, CTI file management, runtime requirements checking, video session control,
    GenICam configuration, camera manifest management, log archive assembly, video file validation, recording
    discovery, and log processing functionality through the MCP protocol, enabling AI agents to programmatically
    interact with the library.
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
    "-c",
    "--camera-index",
    type=int,
    default=None,
    help="The index of the target camera in the list of all cameras discoverable through the Harvesters interface.",
)
@click.option(
    "-b",
    "--blacklisted-node",
    type=str,
    multiple=True,
    default=sorted(DEFAULT_BLACKLISTED_NODES),
    show_default=True,
    help="GenICam node name to exclude from the read, dump, and load operations. Repeat to specify multiple nodes. "
    "Some vendor-specific nodes report ReadWrite access but reject writes at the hardware level. Modify this list to "
    "match your camera hardware. An explicitly named node passed to 'configure write' is always written. Mutually "
    "exclusive with --no-blacklist.",
)
@click.option(
    "--no-blacklist",
    is_flag=True,
    default=False,
    help="Disables all node blacklisting. When set, all ReadWrite nodes are included in the read, dump, and load "
    "operations. Mutually exclusive with --blacklisted-node.",
)
@click.pass_context
def configure_group(
    context: click.Context, camera_index: int | None, blacklisted_node: tuple[str, ...], *, no_blacklist: bool
) -> None:
    """Allows working with the configuration of GenTL (Harvesters) compatible cameras.

    The camera index and the node blacklist are parsed on this group and shared by every subcommand, so they must be
    given before the subcommand name.
    """
    # Consults the parameter source rather than the parsed tuple, since --blacklisted-node carries a non-empty default
    # and therefore arrives populated whether or not the operator named a node.
    blacklist_supplied = context.get_parameter_source("blacklisted_node") is not click.ParameterSource.DEFAULT
    if blacklist_supplied and no_blacklist:
        message = (
            "Unable to run the 'configure' command. The '-b' / '--blacklisted-node' and '--no-blacklist' options are "
            "mutually exclusive, but both were supplied."
        )
        console.error(message=message, error=click.UsageError)

    context.obj = _SharedConfigurationParameters(
        camera_index=camera_index,
        blacklisted_nodes=frozenset() if no_blacklist else frozenset(blacklisted_node),
    )


@configure_group.command("read")
@click.option(
    "-n",
    "--node-name",
    type=str,
    default="",
    help="The name of a specific GenICam node to read. If omitted, lists every writable (ReadWrite) node that is not "
    "blacklisted.",
)
@_pass_shared_parameters
def read_genicam_configuration(shared: _SharedConfigurationParameters, node_name: str) -> None:
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, displays detailed information about that specific node. Otherwise, lists every
    writable (ReadWrite) node that is not blacklisted, with its current value.
    """
    with _connected_genicam_camera(shared=shared) as camera:
        if node_name:
            description = format_genicam_node(node_map=camera.node_map, name=node_name)
            console.echo(message=description, level=LogLevel.SUCCESS, raw=True)
        else:
            node_map = camera.node_map
            names = enumerate_genicam_nodes(node_map=node_map, blacklisted_nodes=shared.blacklisted_nodes)
            console.echo(message=f"Found {len(names)} writable GenICam nodes:", level=LogLevel.SUCCESS)
            for name in names:
                try:
                    info = read_genicam_node(node_map=node_map, name=name)
                    console.echo(message=f"  {info.name} = {info.value}", raw=True)
                except Exception:
                    console.echo(message=f"  {name} = <unreadable>", raw=True)


@configure_group.command("write")
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
@_pass_shared_parameters
def write_genicam_configuration(shared: _SharedConfigurationParameters, node_name: str, value: str) -> None:
    """Writes a value to a GenICam node on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type (integer, float, boolean, or string)
    based on the node's type. The node is read back over the same connection, since a node that reports ReadWrite
    access can still coerce the write to its increment or reject it outright.
    """
    with _connected_genicam_camera(shared=shared) as camera:
        camera.set_node_value(name=node_name, value=value)
        observed = read_genicam_node(node_map=camera.node_map, name=node_name).value
        message = f"Node '{node_name}' written with {value}. The camera reports {observed} for it on this connection."
        console.echo(message=message, level=LogLevel.SUCCESS)


@configure_group.command("dump")
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="The path to the output YAML file to write the configuration to.",
)
@_pass_shared_parameters
def dump_genicam_configuration(shared: _SharedConfigurationParameters, output_file: Path) -> None:
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    The output YAML includes every writable (ReadWrite) node that is not blacklisted, with its current value, as well
    as the camera model and serial number for identity validation.
    """
    with _connected_genicam_camera(shared=shared) as camera:
        config = camera.get_configuration(blacklisted_nodes=shared.blacklisted_nodes)
        config.to_yaml(file_path=output_file)
        console.echo(
            message=f"Configuration saved: {len(config.nodes)} nodes written to {output_file}.",
            level=LogLevel.SUCCESS,
        )


@configure_group.command("load")
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
@_pass_shared_parameters
def load_genicam_configuration(shared: _SharedConfigurationParameters, config_file: Path, *, strict: bool) -> None:
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Applies every non-blacklisted writable node from the configuration file to the camera. Always compares the camera
    model and serial number against the configuration file, warning on a mismatch and aborting instead when --strict
    is set.
    """
    with _connected_genicam_camera(shared=shared) as camera:
        config = GenicamConfiguration.from_yaml(file_path=config_file)
        camera.apply_configuration(config=config, strict_identity=strict, blacklisted_nodes=shared.blacklisted_nodes)
        console.echo(message="Configuration applied successfully.", level=LogLevel.SUCCESS)


@contextmanager
def _connected_genicam_camera(shared: _SharedConfigurationParameters) -> Iterator[HarvestersCamera]:
    """Yields a camera connected to the index the ``configure`` group resolved.

    Every IndexError crossing this block is reported as a Click usage error, since the resolved camera index is the
    only one a subcommand supplies. Such an index names a mistyped option rather than a defect worth unwinding the
    GenTL stack into the terminal for.

    Args:
        shared: The options the ``configure`` group parsed for the running subcommand.

    Yields:
        The connected camera interface.
    """
    try:
        with harvester_connection(camera_index=shared.require_camera_index()) as camera:
            yield camera
    except IndexError as error:
        console.error(message=str(error), error=click.UsageError)
