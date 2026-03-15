"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes camera discovery, CTI file management, runtime requirements checking, and video session
management functionality through the MCP protocol, enabling AI agents to programmatically interact with the
library's core features.
"""

from typing import Literal
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from ataraxis_data_structures import DataLogger

from .saver import (
    VideoEncoders,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import CameraInterfaces, HarvestersCamera, add_cti_file, check_cti_file, discover_camera_ids
from .video_system import VideoSystem
from .configuration import (
    GenicamConfiguration,
    read_genicam_node as read_node_info,
    format_genicam_node,
    enumerate_genicam_nodes,
)

mcp = FastMCP(name="ataraxis-video-system", json_response=True)
"""Initializes the MCP server instance."""

_active_session: VideoSystem | None = None
"""Stores the currently active VideoSystem instance, or None when no session is running."""

_active_logger: DataLogger | None = None
"""Stores the DataLogger instance associated with the active video session, or None when no session is running."""


@mcp.tool()
def list_cameras() -> str:
    """Discovers all cameras compatible with the OpenCV and Harvesters interfaces.

    Returns:
        A newline-separated list of discovered cameras, each showing interface type, index, frame dimensions, and
        frame rate. Harvesters cameras also include model and serial number. Returns a "No cameras discovered"
        message if no cameras are found.
    """
    # Runs the discovery procedure across both OpenCV and Harvesters interfaces. OpenCV cameras are probed by
    # iterating over positional indices, while Harvesters cameras are enumerated through the GenTL Producer.
    all_cameras = discover_camera_ids()

    if not all_cameras:
        return "No cameras discovered on the system."

    # Formats each discovered camera as a human-readable summary line. Harvesters cameras include model and serial
    # number because the GenTL interface exposes this metadata, whereas OpenCV does not.
    lines: list[str] = []

    for camera in all_cameras:
        if camera.interface == CameraInterfaces.OPENCV:
            lines.append(
                f"OpenCV #{camera.camera_index}: "
                f"{camera.frame_width}x{camera.frame_height}@{camera.acquisition_frame_rate}fps"
            )
        else:
            lines.append(
                f"Harvesters #{camera.camera_index}: {camera.model} ({camera.serial_number}) "
                f"{camera.frame_width}x{camera.frame_height}@{camera.acquisition_frame_rate}fps"
            )

    return "\n".join(lines)


@mcp.tool()
def get_cti_status() -> str:
    """Checks whether the library is configured with a valid GenTL Producer interface (.cti) file.

    The Harvesters camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    GeniCam-compatible cameras.

    Returns:
        The configuration status and the path to the configured CTI file, or a "Not configured" message if no valid
        CTI file is set.
    """
    # Reads the persisted CTI file path from the library's configuration storage and verifies that the file still
    # exists on disk. Returns None if no path was previously set or the stored path no longer points to a valid file.
    cti_path = check_cti_file()

    if cti_path is not None:
        return f"CTI: {cti_path}"
    return "CTI: Not configured"


@mcp.tool()
def set_cti_file(file_path: str) -> str:
    """Configures the library to use the specified CTI file for all future runtimes involving GeniCam cameras.

    The Harvesters library requires the GenTL Producer interface (.cti) file to discover and interface with compatible
    cameras. This tool must be called at least once before using the Harvesters interface.

    Args:
        file_path: The absolute path to the CTI file that provides the GenTL Producer interface. It is recommended to
            use the file supplied by the camera vendor, but a general Producer such as mvImpactAcquire is also
            acceptable.

    Returns:
        A confirmation message with the configured CTI file path on success, or an error message describing the
        failure.
    """
    # Validates that the provided path points to an existing file before attempting to persist it.
    path = Path(file_path)

    if not path.exists():
        return f"Error: File not found at {file_path}"

    if not path.is_file():
        return f"Error: Path is not a file: {file_path}"

    # Persists the CTI file path to the library's configuration storage so that it is reused across all future
    # runtimes without needing to be re-specified.
    try:
        add_cti_file(cti_path=path)
    except Exception as e:
        return f"Error: {e}"
    else:
        return f"CTI configured: {path}"


@mcp.tool()
def check_runtime_requirements() -> str:
    """Checks whether the host system meets the requirements for video encoding and camera interfaces.

    Verifies that FFMPEG is installed and accessible, checks for Nvidia GPU availability for hardware-accelerated
    encoding, and checks whether a CTI file is configured for Harvesters camera support.

    Returns:
        A pipe-separated status line showing FFMPEG, GPU, and CTI availability, each marked as "OK", "Missing", or
        "None".
    """
    # Probes the system for each runtime dependency independently. FFMPEG is required for any video encoding, GPU is
    # optional (enables hardware-accelerated H.264/H.265 encoding via NVENC), and the CTI file is only needed for
    # Harvesters camera discovery.
    ffmpeg_available = check_ffmpeg_availability()
    gpu_available = check_gpu_availability()
    cti_path = check_cti_file()

    # Formats each check result as a short status token for compact display.
    ffmpeg_status = "OK" if ffmpeg_available else "Missing"
    gpu_status = "OK" if gpu_available else "None"
    cti_status = "OK" if cti_path is not None else "None"

    return f"FFMPEG: {ffmpeg_status} | GPU: {gpu_status} | CTI: {cti_status}"


@mcp.tool()
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
) -> str:
    """Starts a video capture session with the specified parameters.

    Creates a VideoSystem instance and begins acquiring frames from the camera. Frames are not saved until
    start_frame_saving is called. Only one session can be active at a time.

    Important:
        The AI agent calling this tool MUST ask the user to provide the output_directory path before calling this
        tool. Do not assume or guess the output directory - always prompt the user for an explicit path.

    Args:
        output_directory: The path to the directory where video files will be saved. This must be provided by the
            user - the AI agent should always ask for this value explicitly.
        interface: The camera interface to use ('opencv', 'harvesters', or 'mock'). Defaults to 'opencv'.
        camera_index: The index of the camera to use. Defaults to 0.
        width: The width of frames to capture in pixels. Defaults to 600.
        height: The height of frames to capture in pixels. Defaults to 400.
        frame_rate: The target frame rate in frames per second. Defaults to 30.
        gpu_index: The GPU index for hardware encoding, or -1 for CPU encoding. Defaults to -1.
        display_frame_rate: The rate at which to display acquired frames in a preview window. Defaults to 25 fps.
            Set to 'None' to disable frame display. The display rate cannot exceed the acquisition frame rate.
            Note that frame display is not supported on macOS.
        monochrome: Determines whether to capture in grayscale. Defaults to False (color).

    Returns:
        A summary of the session parameters including interface, camera index, resolution, frame rate, and output
        directory on success, or an error message describing the failure.
    """
    global _active_session, _active_logger

    # Enforces the single-session constraint. Only one VideoSystem can be active at a time because each session
    # exclusively owns the camera device and its associated encoding pipeline.
    if _active_session is not None:
        return "Error: Session already active"

    # Validates the output directory before allocating any resources.
    output_path = Path(output_directory)
    if not output_path.exists():
        return f"Error: Directory not found: {output_directory}"
    if not output_path.is_dir():
        return f"Error: Not a directory: {output_directory}"

    # Maps the string interface name to the corresponding CameraInterfaces enum member.
    if interface.lower() == "mock":
        camera_interface = CameraInterfaces.MOCK
    elif interface.lower() == "harvesters":
        camera_interface = CameraInterfaces.HARVESTERS
    else:
        camera_interface = CameraInterfaces.OPENCV

    try:
        # Initializes and starts the DataLogger first, as the VideoSystem depends on it for runtime event logging.
        _active_logger = DataLogger(output_directory=output_path, instance_name="mcp_video_session")
        _active_logger.start()

        # Creates the VideoSystem with conservative encoding defaults (H.264, fast preset, YUV420) to maximize
        # compatibility across hardware configurations. The system_id 112 distinguishes MCP-initiated sessions from
        # CLI sessions (111) in log output.
        _active_session = VideoSystem(
            system_id=np.uint8(112),
            data_logger=_active_logger,
            output_directory=output_path,
            camera_interface=camera_interface,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            frame_rate=frame_rate,
            display_frame_rate=display_frame_rate,
            color=not monochrome,
            gpu=gpu_index,
            video_encoder=VideoEncoders.H264,
            encoder_speed_preset=EncoderSpeedPresets.FAST,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=15,
        )

        # Spawns camera acquisition and encoding child processes. After this call, frames are being acquired but
        # not yet saved to disk (saving requires an explicit start_frame_saving call).
        _active_session.start()

    except Exception as e:
        # Cleans up partially initialized resources on failure to avoid leaving orphaned processes or file handles.
        if _active_logger is not None:
            _active_logger.stop()
            _active_logger = None
        _active_session = None
        return f"Error: {e}"
    else:
        return f"Session started: {interface} #{camera_index} {width}x{height}@{frame_rate}fps -> {output_directory}"


@mcp.tool()
def stop_video_session() -> str:
    """Stops the active video capture session and releases all resources.

    Stops the VideoSystem and DataLogger, freeing the camera and saving any remaining buffered frames.

    Returns:
        A confirmation that the session has stopped, or an error message if no session is active or shutdown fails.
    """
    global _active_session, _active_logger

    if _active_session is None:
        return "Error: No active session"

    try:
        # Stops the VideoSystem first, which terminates camera acquisition and flushes any buffered frames still in
        # the encoding pipeline. This may block briefly while remaining frames are written to disk.
        _active_session.stop()

        # Stops the DataLogger after the VideoSystem to ensure all runtime events are captured before the log is
        # finalized.
        if _active_logger is not None:
            _active_logger.stop()
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Clears the module-level references regardless of success or failure to allow a new session to be started.
        _active_session = None
        _active_logger = None

    return "Session stopped"


@mcp.tool()
def start_frame_saving() -> str:
    """Starts saving captured frames to the video file.

    Begins writing acquired frames to an MP4 video file in the output directory. A video session must be active.

    Returns:
        A confirmation that recording has started, or an error message if no session is active or the operation fails.
    """
    if _active_session is None:
        return "Error: No active session"

    # Signals the VideoSystem's saver process to begin writing acquired frames to a new MP4 file. Frames acquired
    # before this call are discarded; only frames captured after this point are saved.
    try:
        _active_session.start_frame_saving()
    except Exception as e:
        return f"Error: {e}"
    else:
        return "Recording started"


@mcp.tool()
def stop_frame_saving() -> str:
    """Stops saving frames to the video file.

    Stops writing frames to the video file while keeping the session active. Frame acquisition continues.

    Returns:
        A confirmation that recording has stopped, or an error message if no session is active or the operation fails.
    """
    if _active_session is None:
        return "Error: No active session"

    # Signals the saver process to stop accepting new frames and finalize the current video file. The camera
    # continues acquiring frames, so a subsequent start_frame_saving call will create a new video file.
    try:
        _active_session.stop_frame_saving()
    except Exception as e:
        return f"Error: {e}"
    else:
        return "Recording stopped"


@mcp.tool()
def get_session_status() -> str:
    """Returns the current status of the video session.

    Reports whether a session is active and its current state (acquiring frames, saving frames, etc.).

    Returns:
        The session status as "Inactive", "Running", or "Stopped".
    """
    # No session object exists: either none was started or the previous session was fully torn down.
    if _active_session is None:
        return "Status: Inactive"

    # A session object exists. The 'started' flag indicates whether its child processes are still running. A session
    # that exists but is not started has been stopped and is awaiting cleanup.
    if _active_session.started:
        return "Status: Running"
    return "Status: Stopped"


@mcp.tool()
def read_genicam_node(camera_index: int = 0, node_name: str = "") -> str:  # pragma: no cover
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, returns detailed information about that specific node. If no node name is provided,
    lists all available nodes with their current values.

    Args:
        camera_index: The index of the Harvesters camera to read from.
        node_name: The name of a specific GenICam node to read (e.g., "Width", "ExposureTime"). If empty, all nodes
            are listed.

    Returns:
        Detailed node information for a single node, or a newline-separated summary of all nodes.
    """
    # Opens a temporary connection to the camera. system_id=0 is a placeholder since we only need node map access,
    # not a full VideoSystem lifecycle.
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        # Single-node mode: returns a detailed formatted description including the node's type, current value,
        # valid range or enumeration entries, and access mode.
        if node_name:
            return format_genicam_node(camera.node_map, node_name)

        # All-nodes mode: enumerates every writable node and reads its current value. Nodes that raise exceptions
        # during read (e.g., due to access restrictions or transient hardware state) are reported as <unreadable>
        # rather than aborting the entire listing.
        node_map = camera.node_map
        names = enumerate_genicam_nodes(node_map)
        lines = [f"Found {len(names)} writable GenICam nodes:"]
        for name in names:
            try:
                info = read_node_info(node_map, name)
                lines.append(f"  {info.name} = {info.value}")
            except Exception:
                lines.append(f"  {name} = <unreadable>")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Always disconnects to release the GenTL handle, allowing other processes to access the camera.
        camera.disconnect()


@mcp.tool()
def write_genicam_node(camera_index: int, node_name: str, value: str) -> str:  # pragma: no cover
    """Sets a GenICam node value on a connected Harvesters camera.

    The string value is automatically converted to the appropriate type based on the node's type.

    Args:
        camera_index: The index of the Harvesters camera to write to.
        node_name: The name of the GenICam node to write (e.g., "Width", "ExposureTime").
        value: The string value to write. Automatically converted to the node's native type.

    Returns:
        A confirmation with the node name and written value, or an error description.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        # Connects to the camera and delegates value conversion and writing to the camera's set_node_value method,
        # which inspects the node's type and casts the string value to int, float, bool, or enum as needed.
        camera.connect()
        camera.set_node_value(node_name, value)
    except Exception as e:
        return f"Error: {e}"
    else:
        return f"Node '{node_name}' set to {value}"
    finally:
        camera.disconnect()


@mcp.tool()
def dump_genicam_config(camera_index: int, output_file: str) -> str:  # pragma: no cover
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    Important:
        The AI agent calling this tool MUST ask the user to provide the output_file path before calling this tool.
        Do not assume or guess the output file path.

    Args:
        camera_index: The index of the Harvesters camera to dump the configuration from.
        output_file: The absolute path to the output YAML file. Must be provided by the user.

    Returns:
        A confirmation with the number of nodes saved, or an error description.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        # Reads every accessible node from the camera's GenICam node map and packages them into a
        # GenicamConfiguration object that includes the camera's model, serial number, and per-node metadata
        # (value, range, enum entries).
        config = camera.get_configuration()

        # Serializes the configuration to a YAML file that can later be loaded back onto this or another camera
        # of the same model.
        config.to_yaml(file_path=Path(output_file))
        return f"Configuration saved: {len(config.nodes)} nodes written to {output_file}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        camera.disconnect()


@mcp.tool()
def load_genicam_config(
    camera_index: int, config_file: str, *, strict_identity: bool = False
) -> str:  # pragma: no cover
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Important:
        The AI agent calling this tool MUST ask the user to provide the config_file path before calling this tool.
        Do not assume or guess the configuration file path.

    Args:
        camera_index: The index of the Harvesters camera to load the configuration onto.
        config_file: The absolute path to the YAML configuration file to load. Must be provided by the user.
        strict_identity: Determines whether to abort on camera identity mismatch instead of warning.

    Returns:
        The number of nodes applied and any errors encountered, or an error description.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        # Validates the config file path before attempting deserialization.
        path = Path(config_file)
        if not path.exists():
            return f"Error: File not found at {config_file}"

        # Deserializes the YAML configuration and applies each writable node value to the connected camera. When
        # strict_identity is True, the camera model and serial number must match the values stored in the YAML
        # file; otherwise, a mismatch produces a warning but proceeds with the write.
        config = GenicamConfiguration.from_yaml(file_path=path)
        camera.apply_configuration(config, strict_identity=strict_identity)
    except Exception as e:
        return f"Error: {e}"
    else:
        return "Configuration applied successfully"
    finally:
        camera.disconnect()


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output communication
            and 'streamable-http' for HTTP-based communication.
    """
    # Delegates to the FastMCP run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin; for 'streamable-http' it runs an HTTP server
    # that accepts connections until explicitly terminated.
    mcp.run(transport=transport)
