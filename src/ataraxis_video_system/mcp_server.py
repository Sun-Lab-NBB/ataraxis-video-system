"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes camera discovery, CTI file management, runtime requirements checking, and video session
management functionality through the MCP protocol, enabling AI agents to programmatically interact with the
library's core features.
"""

from typing import Any, Literal  # pragma: no cover
from pathlib import Path  # pragma: no cover
from threading import Lock, Thread  # pragma: no cover
import contextlib  # pragma: no cover
from dataclasses import field, dataclass  # pragma: no cover

import numpy as np  # pragma: no cover
import polars as pl  # pragma: no cover
from ataraxis_time import (  # pragma: no cover  # pragma: no cover
    PrecisionTimer,
    TimerPrecisions,
    TimestampFormats,
    TimestampPrecisions,
    get_timestamp,
)
from mcp.server.fastmcp import FastMCP  # pragma: no cover
from ataraxis_base_utilities import resolve_worker_count  # pragma: no cover
from ataraxis_data_structures import DataLogger, ProcessingStatus, ProcessingTracker  # pragma: no cover

from .saver import (
    VideoEncoders,
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
from .video_system import VideoSystem  # pragma: no cover
from .configuration import (
    DEFAULT_BLACKLISTED_NODES,
    GenicamConfiguration,
    read_genicam_node as read_node_info,
    format_genicam_node,
    enumerate_genicam_nodes,
)  # pragma: no cover
from .log_processing import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    execute_job,
    find_log_archive,
    resolve_recording_roots,
    initialize_processing_tracker,
)  # pragma: no cover

mcp: FastMCP = FastMCP(name="ataraxis-video-system", json_response=True)  # pragma: no cover
"""Stores the MCP server instance used to expose tools to AI agents."""  # pragma: no cover

_active_session: VideoSystem | None = None  # pragma: no cover
"""Stores the currently active VideoSystem instance, or None when no session is running."""  # pragma: no cover

_active_logger: DataLogger | None = None  # pragma: no cover
"""Stores the DataLogger instance associated with the active video session, or None when no session is running."""


@dataclass(slots=True)  # pragma: no cover
class _PendingJob:  # pragma: no cover
    """Describes a single timestamp extraction job queued for execution."""

    log_directory: Path
    """The path to the DataLogger output directory containing the log archive."""
    output_directory: Path
    """The path to the output directory for this log directory's processed data."""
    tracker_path: Path
    """The path to the ProcessingTracker file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    source_id: str
    """The source ID string identifying the log archive to process."""


@dataclass(slots=True)  # pragma: no cover
class _JobExecutionState:  # pragma: no cover
    """Tracks runtime state for batch job execution."""

    all_jobs: dict[str, _PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by job_id."""
    pending_queue: list[_PendingJob] = field(default_factory=list)
    """Jobs awaiting dispatch."""
    active_threads: dict[str, Thread] = field(default_factory=dict)
    """Currently running job_id to Thread mapping."""
    max_parallel_jobs: int = 1
    """The maximum number of jobs to execute concurrently."""
    workers_per_job: int = -1
    """The number of CPU cores to allocate per job."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    canceled: bool = False
    """Indicates whether the execution session has been canceled."""


_job_execution_state: _JobExecutionState | None = None  # pragma: no cover
"""Stores the active execution state for batch log processing jobs."""  # pragma: no cover

_FEATHER_GLOB_PATTERN: str = "camera_*_timestamps.feather"  # pragma: no cover
"""Glob pattern for discovering processed camera timestamp feather files."""  # pragma: no cover

_FEATHER_PREFIX: str = "camera_"  # pragma: no cover
"""Filename prefix for camera timestamp feather files."""  # pragma: no cover

_FEATHER_SUFFIX: str = "_timestamps.feather"  # pragma: no cover
"""Filename suffix for camera timestamp feather files."""  # pragma: no cover


@mcp.tool()  # pragma: no cover
def list_cameras() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def get_cti_status() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def set_cti_file(file_path: str) -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def check_runtime_requirements() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
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
) -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def stop_video_session() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def start_frame_saving() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def stop_frame_saving() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def get_session_status() -> str:  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def read_genicam_node(
    camera_index: int = 0,
    node_name: str = "",
    blacklisted_nodes: list[str] | None = None,
) -> str:  # pragma: no cover
    """Reads GenICam node information from a connected Harvesters camera.

    If a node name is provided, returns detailed information about that specific node. If no node name is provided,
    lists all available nodes with their current values.

    Args:
        camera_index: The index of the Harvesters camera to read from.
        node_name: The name of a specific GenICam node to read (e.g., "Width", "ExposureTime"). If empty, all nodes
            are listed.
        blacklisted_nodes: A list of GenICam node names to exclude from enumeration. Defaults to the built-in
            blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific nodes that report
            ReadWrite access but reject writes at the hardware level. Pass an empty list to disable blacklisting.

    Returns:
        Detailed node information for a single node, or a newline-separated summary of all nodes.
    """
    blacklist = frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES

    # Opens a temporary connection to the camera. system_id=0 is a placeholder since only node map access is needed,
    # not a full VideoSystem lifecycle.
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        # Single-node mode: returns a detailed formatted description including the node's type, current value,
        # valid range or enumeration entries, and access mode.
        if node_name:
            return format_genicam_node(node_map=camera.node_map, name=node_name)

        # All-nodes mode: enumerates every writable node and reads its current value. Nodes that raise exceptions
        # during read (e.g., due to access restrictions or transient hardware state) are reported as <unreadable>
        # rather than aborting the entire listing.
        node_map = camera.node_map
        names = enumerate_genicam_nodes(node_map, blacklisted_nodes=blacklist)
        lines = [f"Found {len(names)} writable GenICam nodes:"]
        for name in names:
            # noinspection PyBroadException
            try:
                info = read_node_info(node_map=node_map, name=name)
                lines.append(f"  {info.name} = {info.value}")
            except Exception:
                lines.append(f"  {name} = <unreadable>")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Always disconnects to release the GenTL handle, allowing other processes to access the camera.
        camera.disconnect()


@mcp.tool()  # pragma: no cover
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
        camera.set_node_value(name=node_name, value=value)
    except Exception as e:
        return f"Error: {e}"
    else:
        return f"Node '{node_name}' set to {value}"
    finally:
        camera.disconnect()


@mcp.tool()  # pragma: no cover
def dump_genicam_config(
    camera_index: int,
    output_file: str,
    blacklisted_nodes: list[str] | None = None,
) -> str:  # pragma: no cover
    """Dumps the full GenICam configuration of a connected Harvesters camera to a YAML file.

    Important:
        The AI agent calling this tool MUST ask the user to provide the output_file path before calling this tool.
        Do not assume or guess the output file path.

    Args:
        camera_index: The index of the Harvesters camera to dump the configuration from.
        output_file: The absolute path to the output YAML file. Must be provided by the user.
        blacklisted_nodes: A list of GenICam node names to exclude from the configuration dump. Defaults to the
            built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific nodes
            that report ReadWrite access but reject writes at the hardware level. Pass an empty list to disable
            blacklisting.

    Returns:
        A confirmation with the number of nodes saved, or an error description.
    """
    blacklist = frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES

    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()

        # Reads every accessible node from the camera's GenICam node map and packages them into a
        # GenicamConfiguration object that includes the camera's model, serial number, and per-node metadata
        # (value, range, enum entries).
        config = camera.get_configuration(blacklisted_nodes=blacklist)

        # Serializes the configuration to a YAML file that can later be loaded back onto this or another camera
        # of the same model.
        config.to_yaml(file_path=Path(output_file))
        return f"Configuration saved: {len(config.nodes)} nodes written to {output_file}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        camera.disconnect()


@mcp.tool()  # pragma: no cover
def load_genicam_config(
    camera_index: int,
    config_file: str,
    *,
    strict_identity: bool = False,
    blacklisted_nodes: list[str] | None = None,
) -> str:  # pragma: no cover
    """Loads a GenICam configuration from a YAML file onto a connected Harvesters camera.

    Important:
        The AI agent calling this tool MUST ask the user to provide the config_file path before calling this tool.
        Do not assume or guess the configuration file path.

    Args:
        camera_index: The index of the Harvesters camera to load the configuration onto.
        config_file: The absolute path to the YAML configuration file to load. Must be provided by the user.
        strict_identity: Determines whether to abort on camera identity mismatch instead of warning.
        blacklisted_nodes: A list of GenICam node names to silently skip during validation and write operations.
            Defaults to the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes
            vendor-specific nodes that report ReadWrite access but reject writes at the hardware level. Pass an
            empty list to disable blacklisting.

    Returns:
        The number of nodes applied and any errors encountered, or an error description.
    """
    blacklist = frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES

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
        camera.apply_configuration(config, strict_identity=strict_identity, blacklisted_nodes=blacklist)
    except Exception as e:
        return f"Error: {e}"
    else:
        return "Configuration applied successfully"
    finally:
        camera.disconnect()


@mcp.tool()  # pragma: no cover
def discover_recording_log_archives_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Discovers log archives under a root directory, grouped hierarchically by recording and log directory.

    Recursively searches the root directory for .npz log archives matching the DataLogger naming convention.
    Archives are grouped by their parent directory (DataLogger output, e.g., ``*_data_log/``), and recording
    roots are resolved using unique path component detection to reliably identify recording session boundaries
    regardless of directory depth or naming conventions. Each log directory is an independent processing unit
    that can be passed directly to prepare_log_processing_batch_tool for targeted processing.

    Args:
        root_directory: The absolute path to the root directory to search for log archives. Searched recursively.

    Returns:
        A dictionary containing hierarchical 'recordings' mapping recording roots to their log directories and
        source IDs, a flat 'log_directories' list for direct use with the batch preparation tool, the union of
        all discovered source IDs in 'all_source_ids', and total counts.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # Discovers all log archives and groups them by parent directory (DataLogger output directory).
    log_dir_data: dict[Path, dict[str, Any]] = {}
    all_source_ids: set[str] = set()
    total_archives = 0

    try:
        for path in sorted(root_path.rglob(f"*{LOG_ARCHIVE_SUFFIX}")):
            source_id = path.name.removesuffix(LOG_ARCHIVE_SUFFIX)
            if not source_id:
                continue

            log_dir = path.parent
            if log_dir not in log_dir_data:
                log_dir_data[log_dir] = {"source_ids": [], "archive_count": 0}

            log_dir_data[log_dir]["source_ids"].append(source_id)
            log_dir_data[log_dir]["archive_count"] += 1
            all_source_ids.add(source_id)
            total_archives += 1
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    if not log_dir_data:
        return {
            "recordings": {},
            "log_directories": [],
            "all_source_ids": [],
            "total_recordings": 0,
            "total_log_directories": 0,
            "total_archives": 0,
        }

    # Resolves recording roots from log directory paths using unique path component detection.
    log_dir_paths = list(log_dir_data.keys())
    try:
        recording_roots = resolve_recording_roots(paths=log_dir_paths)
    except RuntimeError:
        # Falls back to using each log directory's parent as the recording root if unique component detection fails
        # (e.g., single log directory where all components are trivially unique).
        recording_roots = tuple(dict.fromkeys(log_dir.parent for log_dir in log_dir_paths))

    # Builds a mapping from each log directory to its recording root by finding the longest matching root prefix.
    log_dir_to_root: dict[Path, Path] = {}
    for log_dir in log_dir_paths:
        for root in recording_roots:
            if log_dir == root or root in log_dir.parents:
                log_dir_to_root[log_dir] = root
                break
        else:
            # Falls back to the log directory's parent if no root prefix matches.
            log_dir_to_root[log_dir] = log_dir.parent

    # Groups log directories under their resolved recording roots for hierarchical display.
    recordings: dict[str, dict[str, Any]] = {}
    for log_dir, data in log_dir_data.items():
        recording_root = str(log_dir_to_root[log_dir])
        if recording_root not in recordings:
            recordings[recording_root] = {"log_directories": {}}
        recordings[recording_root]["log_directories"][str(log_dir)] = {
            "source_ids": data["source_ids"],
            "archive_count": data["archive_count"],
        }

    return {
        "recordings": recordings,
        "log_directories": sorted(str(log_dir) for log_dir in log_dir_paths),
        "all_source_ids": sorted(all_source_ids),
        "total_recordings": len(recordings),
        "total_log_directories": len(log_dir_data),
        "total_archives": total_archives,
    }


@mcp.tool()  # pragma: no cover
def prepare_log_processing_batch_tool(  # pragma: no cover
    log_directories: list[str],
    source_ids: list[str] | None = None,
    output_directories: list[str] | None = None,
) -> dict[str, Any]:
    """Prepares an execution manifest for batch log processing without starting execution.

    Accepts a list of DataLogger output directories (log directories), each containing .npz log archives.
    Initializes a ProcessingTracker with one timestamp-extraction job per source ID for each log directory.
    Idempotent: if a tracker already exists for a log directory, returns the existing manifest with current
    job statuses instead of reinitializing.

    Use discover_recording_log_archives_tool first to obtain log directory paths. The discovery tool returns
    both a hierarchical 'recordings' view for understanding the session structure and a flat 'log_directories'
    list that can be passed directly to this tool. Select all discovered log directories for full processing,
    or pass a subset for targeted processing of specific sessions or DataLogger outputs.

    Important:
        The AI agent calling this tool MUST ask the user to provide recording or log directory paths before
        calling this tool. Do not assume or guess directory paths.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_recording_log_archives_tool,
            or any directory containing .npz log archives directly (non-recursive, immediate children only).
        source_ids: An optional list of source IDs to process. If not provided, all discovered source IDs in
            each log directory are included.
        output_directories: An optional list of absolute paths for per-log-directory output. Must match the
            length of log_directories. When not provided, output is written to each log directory itself.

    Returns:
        A dictionary containing per-log-directory manifests in 'log_directories' with tracker paths and job
        lists, total counts, and any invalid paths.
    """
    if output_directories is not None and len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    result_log_dirs: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_dir_str in enumerate(log_directories):
        log_dir_path = Path(log_dir_str)

        if not log_dir_path.exists() or not log_dir_path.is_dir():
            invalid_paths.append(log_dir_str)
            continue

        # Discovers archives in this log directory (non-recursive, immediate children only).
        discovered_ids = sorted(
            path.name.removesuffix(LOG_ARCHIVE_SUFFIX)
            for path in log_dir_path.glob(f"*{LOG_ARCHIVE_SUFFIX}")
            if path.name.removesuffix(LOG_ARCHIVE_SUFFIX)
        )

        # Filters by requested source IDs if specified.
        if source_ids is not None:
            filtered_ids = [source_id for source_id in discovered_ids if source_id in source_ids]
        else:
            filtered_ids = discovered_ids

        if not filtered_ids:
            result_log_dirs[log_dir_str] = {"source_ids": [], "jobs": [], "tracker_path": None, "summary": {}}
            continue

        # Resolves the output directory for this log directory. Defaults to the log directory itself.
        output_path = Path(output_directories[entry_index]) if output_directories is not None else log_dir_path

        output_path.mkdir(parents=True, exist_ok=True)
        tracker_path = output_path / TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state.
            try:
                tracker_status = _read_tracker_status(tracker_path=tracker_path)
            except Exception:
                tracker_status = {"jobs": [], "summary": {}}

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(output_path),
                "source_ids": filtered_ids,
                **tracker_status,
            }
            total_jobs += len(tracker_status.get("jobs", []))
        else:
            # Initializes a new tracker with jobs for the filtered source IDs.
            job_ids = initialize_processing_tracker(output_directory=output_path, source_ids=filtered_ids)

            jobs: list[dict[str, str]] = [
                {
                    "job_id": job_ids[source_id],
                    "source_id": source_id,
                    "status": "SCHEDULED",
                    "log_directory": log_dir_str,
                    "output_directory": str(output_path),
                    "tracker_path": str(tracker_path),
                }
                for source_id in filtered_ids
            ]

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(output_path),
                "source_ids": filtered_ids,
                "jobs": jobs,
                "summary": {
                    "total": len(jobs),
                    "succeeded": 0,
                    "failed": 0,
                    "running": 0,
                    "scheduled": len(jobs),
                },
            }
            total_jobs += len(jobs)

    result: dict[str, Any] = {
        "success": True,
        "log_directories": result_log_dirs,
        "total_log_directories": len(result_log_dirs),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()  # pragma: no cover
def execute_log_processing_jobs_tool(  # pragma: no cover
    jobs: list[dict[str, str]],
    *,
    workers_per_job: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution with resource allocation.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager that dispatches jobs with the resolved worker and parallelism counts.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors, each a dictionary with 'log_directory', 'output_directory',
            'tracker_path', 'job_id', and 'source_id' keys.
        workers_per_job: The number of CPU cores per job. Set to -1 for automatic resolution.
        max_parallel_jobs: The maximum number of concurrent jobs. Set to -1 for automatic resolution (defaults to 1).

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved resource allocation, and any invalid jobs.
    """
    global _job_execution_state

    # Enforces single-session constraint.
    if (
        _job_execution_state is not None
        and _job_execution_state.manager_thread is not None
        and _job_execution_state.manager_thread.is_alive()
    ):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"log_directory", "output_directory", "tracker_path", "job_id", "source_id"}
    pending: list[_PendingJob] = []
    all_jobs: dict[str, _PendingJob] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_dict in jobs:
        if not required_keys.issubset(job_dict.keys()):
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {required_keys - job_dict.keys()}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        pending_job = _PendingJob(
            log_directory=Path(job_dict["log_directory"]),
            output_directory=Path(job_dict["output_directory"]),
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            source_id=job_dict["source_id"],
        )
        pending.append(pending_job)
        all_jobs[pending_job.job_id] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves resource allocation.
    resolved_workers = resolve_worker_count(requested_workers=workers_per_job)
    resolved_parallel = max_parallel_jobs if max_parallel_jobs > 0 else 1

    # Creates execution state and starts the manager thread.
    _job_execution_state = _JobExecutionState(
        all_jobs=all_jobs,
        pending_queue=pending,
        max_parallel_jobs=resolved_parallel,
        workers_per_job=resolved_workers,
    )

    manager = Thread(target=_job_execution_manager, daemon=True)
    manager.start()
    _job_execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "workers_per_job": resolved_workers,
        "max_parallel_jobs": resolved_parallel,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()  # pragma: no cover
def get_log_processing_status_tool() -> dict[str, Any]:  # pragma: no cover
    """Returns the current status of the active log processing execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. If no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts
        for pending, running, succeeded, and failed jobs.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    # Reads status from tracker files for each job.
    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    # Groups jobs by tracker path to minimize file reads.
    tracker_jobs: dict[Path, list[_PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)

    for tracker_path, path_jobs in tracker_jobs.items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:
            job_details.extend(
                {"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"} for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id in tracker.jobs:
                job_state = tracker.jobs[job.job_id]
                status = job_state.status

                if status == ProcessingStatus.SUCCEEDED:
                    succeeded_count += 1
                elif status == ProcessingStatus.FAILED:
                    failed_count += 1
                elif status == ProcessingStatus.RUNNING:
                    running_count += 1
                else:
                    scheduled_count += 1

                entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id, "status": status.name}
                if job_state.error_message is not None:
                    entry["error_message"] = job_state.error_message
                job_details.append(entry)
            else:
                job_details.append({"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"})

    return {
        "active": manager_alive,
        "canceled": state.canceled,
        "jobs": job_details,
        "summary": {
            "total": len(state.all_jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


@mcp.tool()  # pragma: no cover
def get_log_processing_timing_tool() -> dict[str, Any]:  # pragma: no cover
    """Returns timing information for all jobs in the active execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps from ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with
        total elapsed seconds and throughput.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()
    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    job_timing: list[dict[str, Any]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0

    # Groups jobs by tracker path to minimize file reads.
    tracker_jobs: dict[Path, list[_PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)

    for tracker_path, path_jobs in tracker_jobs.items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:  # noqa: S112
            continue

        for job in path_jobs:
            if job.job_id not in tracker.jobs:
                continue

            job_info = tracker.jobs[job.job_id]
            entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id}

            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                entry["elapsed_seconds"] = round((current_us - int(job_info.started_at)) / 1_000_000, 2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    entry["duration_seconds"] = round(
                        (int(job_info.completed_at) - int(job_info.started_at)) / 1_000_000, 2
                    )

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level statistics.
    session: dict[str, Any] = {
        "total_elapsed_seconds": round((current_us - earliest_start) / 1_000_000, 2) if earliest_start else 0.0,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": sum(1 for j in job_timing if "elapsed_seconds" in j),
        "pending_count": len(state.all_jobs)
        - completed_count
        - failed_count
        - sum(1 for j in job_timing if "elapsed_seconds" in j),
    }

    if completed_count > 0 and earliest_start is not None:
        elapsed_hours = (current_us - earliest_start) / 1_000_000 / 3600
        if elapsed_hours > 0:
            session["throughput_jobs_per_hour"] = round(completed_count / elapsed_hours, 2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()  # pragma: no cover
def cancel_log_processing_tool() -> dict[str, Any]:  # pragma: no cover
    """Cancels the active log processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    if _job_execution_state is None:
        return {"canceled": False, "message": "No execution session is active."}

    state = _job_execution_state

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_threads)

    # Counts final job statuses from tracker files.
    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
            for job_state in tracker.jobs.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1
        except Exception:  # noqa: S110
            pass

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} job(s) still completing.",
        "final_state": {
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "active_jobs_at_cancel": active_count,
        },
    }


@mcp.tool()  # pragma: no cover
def reset_log_processing_jobs_tool(  # pragma: no cover
    tracker_path: str,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the ProcessingTracker YAML file.
        source_ids: An optional list of source IDs whose jobs should be reset. If not provided, all jobs are reset.

    Returns:
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Identifies which job IDs to reset based on source_ids filter.
    target_ids: set[str] = set()
    if source_ids is not None:
        source_id_set = set(source_ids)
        for job_id, job_state in tracker.jobs.items():
            if job_state.specifier in source_id_set:
                target_ids.add(job_id)
    else:
        target_ids = set(tracker.jobs.keys())

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Collects (job_name, specifier) tuples for the jobs to reset.
    reset_jobs: list[tuple[str, str]] = [
        (tracker.jobs[job_id].job_name, tracker.jobs[job_id].specifier) for job_id in target_ids
    ]

    # Removes target jobs and re-initializes them.
    for job_id in target_ids:
        del tracker.jobs[job_id]
    tracker.to_yaml(file_path=path)

    # Re-initializes the reset jobs.
    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    # Reads back the updated state for the response.
    try:
        updated_status = _read_tracker_status(tracker_path=path)
    except Exception:
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()  # pragma: no cover
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Discovers and summarizes processing status for all log directories under a root directory.

    Recursively searches for camera_processing_tracker.yaml files and aggregates their status. Each tracker
    corresponds to a single DataLogger output directory.

    Args:
        root_directory: The absolute path to the root directory to search for tracker files.

    Returns:
        A dictionary containing per-log-directory status summaries and aggregate counts.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    log_dir_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for tracker_path in sorted(root_path.rglob(TRACKER_FILENAME)):
        log_dir = str(tracker_path.parent)
        try:
            status = _read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            # Derives a high-level status from job counts.
            total = summary.get("total", 0)
            if summary.get("failed", 0) > 0:
                dir_status = "failed"
            elif summary.get("succeeded", 0) == total and total > 0:
                dir_status = "completed"
            elif summary.get("running", 0) > 0:
                dir_status = "processing"
            elif summary.get("scheduled", 0) == total and total > 0:
                dir_status = "not_started"
            else:
                dir_status = "in_progress"

            log_dir_statuses.append(
                {
                    "log_directory": log_dir,
                    "tracker_path": str(tracker_path),
                    "status": dir_status,
                    **status,
                }
            )
        except Exception:
            log_dir_statuses.append(
                {
                    "log_directory": log_dir,
                    "tracker_path": str(tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    return {
        "log_directories": log_dir_statuses,
        "total_log_directories": len(log_dir_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()  # pragma: no cover
def discover_processed_camera_logs_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Discovers processed camera timestamp feather files under a root directory.

    Recursively searches for feather files matching the ``camera_*_timestamps.feather`` naming convention produced by
    the log processing pipeline. Groups results by parent directory and cross-references with processing tracker files
    to identify fully processed, partially processed, and unprocessed directories.

    Args:
        root_directory: The absolute path to the root directory to search for feather files. Searched recursively.

    Returns:
        A dictionary containing per-directory feather file details grouped under 'directories', aggregate counts,
        and a 'status_summary' section derived from cross-referencing tracker files when available.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # Discovers all feather files and groups them by parent directory.
    dir_data: dict[Path, list[dict[str, Any]]] = {}

    try:
        for path in sorted(root_path.rglob(_FEATHER_GLOB_PATTERN)):
            source_id = path.name.removeprefix(_FEATHER_PREFIX).removesuffix(_FEATHER_SUFFIX)
            if not source_id:
                continue

            parent = path.parent
            if parent not in dir_data:
                dir_data[parent] = []

            dir_data[parent].append(
                {
                    "source_id": source_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    if not dir_data:
        return {
            "directories": {},
            "all_source_ids": [],
            "total_directories": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "status_summary": {
                "fully_processed": 0,
                "partially_processed": 0,
                "unprocessed": 0,
                "no_tracker": 0,
            },
        }

    # Builds per-directory results and cross-references with tracker files.
    all_source_ids: set[str] = set()
    total_files = 0
    total_size_bytes = 0
    status_counts: dict[str, int] = {
        "fully_processed": 0,
        "partially_processed": 0,
        "unprocessed": 0,
        "no_tracker": 0,
    }

    directories: dict[str, dict[str, Any]] = {}
    for directory, files in sorted(dir_data.items(), key=lambda item: item[0]):
        source_ids = [entry["source_id"] for entry in files]
        dir_size = sum(entry["size_bytes"] for entry in files)
        all_source_ids.update(source_ids)
        total_files += len(files)
        total_size_bytes += dir_size

        # Checks for a processing tracker in the same directory.
        tracker_candidate = directory / TRACKER_FILENAME
        tracker_path_str: str | None = None
        if tracker_candidate.exists():
            try:
                tracker_status = _read_tracker_status(tracker_path=tracker_candidate)
                tracker_source_ids = {job["source_id"] for job in tracker_status.get("jobs", [])}
                summary = tracker_status.get("summary", {})
                tracker_path_str = str(tracker_candidate)

                # Classifies directory based on tracker state and feather file presence.
                feather_source_set = set(source_ids)
                if summary.get("succeeded", 0) == summary.get("total", 0) and feather_source_set >= tracker_source_ids:
                    processing_status = "fully_processed"
                elif feather_source_set:
                    processing_status = "partially_processed"
                else:
                    processing_status = "unprocessed"
            except Exception:
                processing_status = "no_tracker"
        else:
            processing_status = "no_tracker"

        status_counts[processing_status] += 1

        directories[str(directory)] = {
            "feather_files": files,
            "source_ids": sorted(source_ids),
            "file_count": len(files),
            "total_size_bytes": dir_size,
            "processing_status": processing_status,
            "tracker_path": tracker_path_str,
        }

    return {
        "directories": directories,
        "all_source_ids": sorted(all_source_ids),
        "total_directories": len(directories),
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "status_summary": status_counts,
    }


@mcp.tool()  # pragma: no cover
def analyze_camera_frame_statistics_tool(  # pragma: no cover
    feather_file: str,
    drop_threshold_us: int = 0,
    max_drop_locations: int = 50,
) -> dict[str, Any]:  # pragma: no cover
    """Reads a processed camera timestamp feather file and computes frame acquisition statistics.

    Computes basic recording statistics (total frames, duration, estimated frame rate), inter-frame timing distribution
    (mean, median, standard deviation, min, max), and frame drop analysis (gap detection, estimated drop count, drop
    locations). Frame drops are identified as inter-frame intervals exceeding a threshold, which defaults to 2x the
    median inter-frame interval when not specified.

    Args:
        feather_file: The absolute path to a camera timestamp feather file produced by the log processing pipeline.
            Expected filename pattern: ``camera_{source_id}_timestamps.feather``.
        drop_threshold_us: The inter-frame interval threshold in microseconds above which a gap is classified as a
            frame drop. If set to 0 (default), the threshold is automatically computed as 2x the median inter-frame
            interval.
        max_drop_locations: The maximum number of frame drop locations to include in the output. Caps the
            'drop_locations' list to prevent oversized responses. Defaults to 50.

    Returns:
        A dictionary containing 'basic_stats', 'inter_frame_timing', and 'frame_drop_analysis' sections with computed
        statistics, or an error dictionary if the file cannot be read.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return {"error": f"File does not exist: {feather_file}"}

    if not file_path.is_file():
        return {"error": f"Path is not a file: {feather_file}"}

    # Reads the feather file and validates the expected schema.
    try:
        dataframe = pl.read_ipc(source=file_path)
    except Exception as error:
        return {"error": f"Unable to read feather file: {error}"}

    if "frame_time_us" not in dataframe.columns:
        return {"error": f"Missing required 'frame_time_us' column. Found columns: {dataframe.columns}"}

    timestamps = dataframe["frame_time_us"].to_numpy()
    total_frames = len(timestamps)

    # Handles edge cases for empty or single-frame recordings.
    if total_frames == 0:
        return {
            "file": feather_file,
            "basic_stats": {"total_frames": 0},
            "inter_frame_timing": {},
            "frame_drop_analysis": {},
        }

    if total_frames == 1:
        return {
            "file": feather_file,
            "basic_stats": {
                "total_frames": 1,
                "first_timestamp_us": int(timestamps[0]),
                "last_timestamp_us": int(timestamps[0]),
                "duration_us": 0,
                "duration_seconds": 0.0,
                "estimated_fps": 0.0,
            },
            "inter_frame_timing": {},
            "frame_drop_analysis": {},
        }

    # Computes basic recording statistics.
    first_timestamp_us = int(timestamps[0])
    last_timestamp_us = int(timestamps[-1])
    duration_us = last_timestamp_us - first_timestamp_us
    duration_seconds = round(duration_us / 1_000_000, 6)
    estimated_fps = round((total_frames - 1) / (duration_us / 1_000_000), 3) if duration_us > 0 else 0.0

    # Computes inter-frame interval statistics. Casts to int64 to handle potential uint64 underflow.
    intervals_us = np.diff(timestamps).astype(np.int64)
    mean_us = round(float(np.mean(intervals_us)), 2)
    median_us = round(float(np.median(intervals_us)), 2)
    std_us = round(float(np.std(intervals_us)), 2)
    min_us = int(np.min(intervals_us))
    max_us = int(np.max(intervals_us))

    # Performs frame drop analysis using the specified or auto-detected threshold.
    if drop_threshold_us > 0:
        threshold = float(drop_threshold_us)
        threshold_source = "user_specified"
    else:
        threshold = 2.0 * median_us
        threshold_source = "auto_2x_median"

    drop_mask = intervals_us > threshold
    drop_indices = np.where(drop_mask)[0]
    total_gaps_detected = len(drop_indices)

    if total_gaps_detected > 0:
        # Estimates the number of dropped frames per gap using the median interval as expected spacing.
        expected_interval = median_us if median_us > 0 else 1.0
        dropped_per_gap = np.round(intervals_us[drop_mask] / expected_interval).astype(np.int64) - 1
        total_estimated_dropped_frames = int(np.sum(np.maximum(dropped_per_gap, 0)))

        total_expected_frames = total_frames + total_estimated_dropped_frames
        drop_rate_percent = round(total_estimated_dropped_frames / total_expected_frames * 100, 4)

        longest_gap_us = int(np.max(intervals_us[drop_mask]))
        longest_gap_ms = round(longest_gap_us / 1000, 4)

        # Builds the capped drop locations list.
        drop_locations: list[dict[str, Any]] = []
        for index in drop_indices[:max_drop_locations]:
            gap_us = int(intervals_us[index])
            estimated_lost = max(round(gap_us / expected_interval) - 1, 0)
            drop_locations.append(
                {
                    "frame_index": int(index),
                    "gap_us": gap_us,
                    "gap_ms": round(gap_us / 1000, 4),
                    "estimated_frames_lost": estimated_lost,
                }
            )

        frame_drop_analysis: dict[str, Any] = {
            "threshold_us": round(threshold, 2),
            "threshold_source": threshold_source,
            "total_gaps_detected": total_gaps_detected,
            "total_estimated_dropped_frames": total_estimated_dropped_frames,
            "drop_rate_percent": drop_rate_percent,
            "longest_gap_us": longest_gap_us,
            "longest_gap_ms": longest_gap_ms,
            "drop_locations": drop_locations,
            "drop_locations_truncated": total_gaps_detected > max_drop_locations,
        }
    else:
        frame_drop_analysis = {
            "threshold_us": round(threshold, 2),
            "threshold_source": threshold_source,
            "total_gaps_detected": 0,
            "total_estimated_dropped_frames": 0,
            "drop_rate_percent": 0.0,
            "longest_gap_us": 0,
            "longest_gap_ms": 0.0,
            "drop_locations": [],
            "drop_locations_truncated": False,
        }

    return {
        "file": feather_file,
        "basic_stats": {
            "total_frames": total_frames,
            "first_timestamp_us": first_timestamp_us,
            "last_timestamp_us": last_timestamp_us,
            "duration_us": duration_us,
            "duration_seconds": duration_seconds,
            "estimated_fps": estimated_fps,
        },
        "inter_frame_timing": {
            "mean_us": mean_us,
            "median_us": median_us,
            "std_us": std_us,
            "min_us": min_us,
            "max_us": max_us,
            "mean_ms": round(mean_us / 1000, 4),
            "median_ms": round(median_us / 1000, 4),
            "std_ms": round(std_us / 1000, 4),
            "min_ms": round(min_us / 1000, 4),
            "max_ms": round(max_us / 1000, 4),
        },
        "frame_drop_analysis": frame_drop_analysis,
    }


def _job_worker(job: _PendingJob, workers: int) -> None:  # pragma: no cover
    """Executes a single timestamp extraction job in a background thread.

    Resolves the log archive path, runs timestamp extraction, writes output, and updates the ProcessingTracker.
    If the pipeline terminates without updating the tracker to a terminal state, marks the job as failed.

    Args:
        job: The pending job descriptor containing directory paths, source ID, and job ID.
        workers: The number of CPU cores to allocate for parallel processing within this job.
    """
    tracker = ProcessingTracker(file_path=job.tracker_path)

    # execute_job already calls tracker.fail_job on exception, so the tracker state is updated. The exception
    # is suppressed here to prevent it from terminating the worker thread.
    with contextlib.suppress(Exception):
        log_path = find_log_archive(log_directory=job.log_directory, source_id=job.source_id)
        execute_job(
            log_path=log_path,
            output_directory=job.output_directory,
            source_id=job.source_id,
            job_id=job.job_id,
            workers=workers,
            tracker=tracker,
            display_progress=False,
        )

    # Failsafe: if the tracker was not updated to a terminal state, marks the job as failed.
    try:
        reloaded = ProcessingTracker.from_yaml(file_path=job.tracker_path)
        if job.job_id in reloaded.jobs:
            status = reloaded.jobs[job.job_id].status
            if status not in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
                tracker.fail_job(job_id=job.job_id, error_message="Job terminated without updating tracker status.")
    except Exception:  # noqa: S110
        pass


def _job_execution_manager() -> None:  # pragma: no cover
    """Dispatches queued jobs with concurrency control.

    Runs as a daemon thread, polling at 1-second intervals. Dispatches jobs from the pending queue up to the
    max_parallel_jobs limit. Exits when the queue is empty and no active threads remain.
    """
    if _job_execution_state is None:
        return

    state = _job_execution_state
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    while True:
        with state.lock:
            # Cleans up completed threads.
            completed_ids = [job_id for job_id, thread in state.active_threads.items() if not thread.is_alive()]
            for job_id in completed_ids:
                del state.active_threads[job_id]

            # Exits when no pending jobs and no active threads remain.
            if not state.pending_queue and not state.active_threads:
                break

            # Stops dispatching new jobs if canceled. Waits for active jobs to finish.
            if state.canceled:
                if not state.active_threads:
                    break
            else:
                # Dispatches jobs up to the concurrency limit.
                while state.pending_queue and len(state.active_threads) < state.max_parallel_jobs:
                    job = state.pending_queue.pop(0)
                    thread = Thread(
                        target=_job_worker,
                        kwargs={"job": job, "workers": state.workers_per_job},
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[job.job_id] = thread

        # Polls at 1-second intervals outside the lock to avoid blocking other threads.
        poll_timer.delay(delay=1, allow_sleep=True)


def _read_tracker_status(tracker_path: Path) -> dict[str, Any]:  # pragma: no cover
    """Reads a log processing tracker file and returns structured per-job status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.

    Returns:
        A dictionary containing per-job status details and summary counts.
    """
    tracker = ProcessingTracker.from_yaml(file_path=tracker_path)

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in tracker.jobs.items():
        source_id = job_state.specifier or job_id[:8]
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {"job_id": job_id, "source_id": source_id, "status": status.name}
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        job_details.append(entry)

    return {
        "jobs": job_details,
        "summary": {
            "total": len(tracker.jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


# Placed after all @mcp.tool() definitions so that all tools are registered with the FastMCP instance before the
# server run loop is callable.
def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:  # pragma: no cover
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output communication
            and 'streamable-http' for HTTP-based communication.
    """
    # Delegates to the FastMCP run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin; for 'streamable-http' it runs an HTTP server
    # that accepts connections until explicitly terminated.
    mcp.run(transport=transport)
