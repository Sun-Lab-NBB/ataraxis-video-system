"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes camera discovery, CTI file management, runtime requirements checking, and video session
management functionality through the MCP protocol, enabling AI agents to programmatically interact with the
library's core features.
"""

import json  # pragma: no cover
from typing import Any, Literal  # pragma: no cover
from pathlib import Path  # pragma: no cover
from threading import Lock, Thread  # pragma: no cover
import contextlib  # pragma: no cover
import subprocess  # pragma: no cover
from dataclasses import field, dataclass  # pragma: no cover
from collections.abc import Generator  # pragma: no cover
from concurrent.futures import ProcessPoolExecutor  # pragma: no cover

import numpy as np  # pragma: no cover
import polars as pl  # pragma: no cover
from ataraxis_time import (  # pragma: no cover
    TimeUnits,
    PrecisionTimer,
    TimerPrecisions,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from mcp.server.fastmcp import FastMCP  # pragma: no cover
from ataraxis_base_utilities import resolve_worker_count  # pragma: no cover
from ataraxis_data_structures import (
    DataLogger,
    ProcessingStatus,
    ProcessingTracker,
    delete_directory,
    assemble_log_archives,
)  # pragma: no cover

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
from .manifest import (
    CAMERA_MANIFEST_FILENAME,
    CameraManifest,
    write_camera_manifest,
)  # pragma: no cover
from .video_system import MAXIMUM_QUANTIZATION_VALUE, VideoSystem  # pragma: no cover
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
    CAMERA_TIMESTAMPS_DIRECTORY,
    PARALLEL_PROCESSING_THRESHOLD,
    execute_job,
    find_log_archive,
    resolve_recording_roots,
    initialize_processing_tracker,
)  # pragma: no cover

mcp: FastMCP = FastMCP(name="ataraxis-video-system", json_response=True)  # pragma: no cover
"""Stores the MCP server instance used to expose tools to AI agents."""  # pragma: no cover

_WORKER_SCALING_FACTOR: int = 1000  # pragma: no cover
"""Controls the saturation floor via the formula ``ceil(sqrt(messages / factor))``. The square root models diminishing
returns from process parallelism. This value sets the minimum workers a job receives before the budget division can
push it lower. With a factor of 1,000, a 648,000-message archive (120 fps x 1.5 h) has a saturation floor of 25
workers."""  # pragma: no cover

_WORKER_MULTIPLE: int = 5  # pragma: no cover
"""Worker counts above 1 are rounded down to the nearest multiple of this value for clean allocation."""

_RESERVED_CORES: int = 2  # pragma: no cover
"""The number of CPU cores reserved for system operations. The worker budget is computed as available cores minus this
value, with a minimum of 1."""

_FEATHER_PREFIX: str = "camera_"  # pragma: no cover
"""Filename prefix for camera timestamp feather files."""  # pragma: no cover

_FEATHER_SUFFIX: str = "_timestamps.feather"  # pragma: no cover
"""Filename suffix for camera timestamp feather files."""  # pragma: no cover


_active_session: VideoSystem | None = None  # pragma: no cover
"""Stores the currently active VideoSystem instance, or None when no session is running."""  # pragma: no cover

_active_logger: DataLogger | None = None  # pragma: no cover
"""Stores the DataLogger instance associated with the active video session, or None when no session is running."""

_session_info: dict[str, Any] | None = None  # pragma: no cover
"""Stores session configuration parameters captured at creation time for status reporting."""  # pragma: no cover


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

    @property  # pragma: no cover
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the tracker
        path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)  # pragma: no cover
class _ActiveGroup:  # pragma: no cover
    """Tracks a group of jobs executing sequentially with a shared ProcessPoolExecutor."""

    source_id: str
    """The shared source ID for all jobs in this group, or a unique identifier for single-job groups."""
    jobs: list[_PendingJob]
    """The jobs in this group, processed sequentially by the group worker thread."""
    workers: int
    """The number of CPU cores allocated to this group's ProcessPoolExecutor."""
    thread: Thread
    """The background thread executing the group."""


@dataclass(slots=True)  # pragma: no cover
class _JobExecutionState:  # pragma: no cover
    """Tracks runtime state for batch job execution with budget-based worker allocation.

    The execution manager groups pending jobs by source ID so that archives with similar sizes share a single
    ProcessPoolExecutor. Each group is dispatched as one thread that processes its jobs sequentially, reusing the
    pool across all archives in the group. This avoids the overhead of repeatedly spawning and tearing down worker
    processes for archives of the same size.
    """

    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by (tracker_path, job_id) dispatch key."""
    pending_queue: list[_PendingJob] = field(default_factory=list)
    """Jobs awaiting dispatch."""
    active_groups: list[_ActiveGroup] = field(default_factory=list)
    """Currently executing job groups, each with its own thread and worker allocation."""
    job_message_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    """Maps each dispatch key to its archive message count, probed before execution."""
    worker_budget: int = 1
    """Total CPU cores available for the execution session."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    canceled: bool = False
    """Indicates whether the execution session has been canceled."""


_job_execution_state: _JobExecutionState | None = None  # pragma: no cover
"""Stores the active execution state for batch log processing jobs."""  # pragma: no cover


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
    video_encoder: str = "H264",
    encoder_speed_preset: int = 3,
    output_pixel_format: str = "yuv420p",
    quantization_parameter: int = 15,
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
        interface: The camera interface to use ('opencv', 'harvesters', or 'mock').
        camera_index: The index of the camera to use.
        width: The width of frames to capture in pixels.
        height: The height of frames to capture in pixels.
        frame_rate: The target frame rate in frames per second.
        gpu_index: The GPU index for hardware encoding, or -1 for CPU encoding.
        display_frame_rate: The rate at which to display acquired frames in a preview window. Set to None to
            disable frame display. The display rate cannot exceed the acquisition frame rate. Frame display is
            not supported on macOS.
        monochrome: Determines whether to capture in grayscale.
        video_encoder: The video encoder to use. Must be 'H264' or 'H265'.
        encoder_speed_preset: The encoder speed preset from 1 (fastest) to 7 (slowest). Higher values produce
            better compression at the expense of CPU time.
        output_pixel_format: The output video pixel format. Must be 'yuv420p' or 'yuv444p'.
        quantization_parameter: The quantization parameter controlling compression quality. Lower values produce
            higher quality at larger file sizes (0 = best, 51 = worst).

    Returns:
        A summary of the session parameters including interface, camera index, resolution, frame rate, encoding
        configuration, and output directory on success, or an error message describing the failure.
    """
    global _active_session, _active_logger, _session_info

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

    # Validates and maps the video encoder string to the corresponding enum member.
    encoder_upper = video_encoder.upper()
    if encoder_upper not in {member.value for member in VideoEncoders}:
        return f"Error: Invalid video_encoder '{video_encoder}'. Must be 'H264' or 'H265'."
    resolved_encoder = VideoEncoders(encoder_upper)

    # Validates the encoder speed preset against the legal range (1-7).
    if encoder_speed_preset not in {member.value for member in EncoderSpeedPresets}:
        return f"Error: Invalid encoder_speed_preset {encoder_speed_preset}. Must be 1-7."
    resolved_preset = EncoderSpeedPresets(encoder_speed_preset)

    # Validates the output pixel format against the corresponding enum.
    pixel_format_lower = output_pixel_format.lower()
    if pixel_format_lower not in {member.value for member in OutputPixelFormats}:
        return f"Error: Invalid output_pixel_format '{output_pixel_format}'. Must be 'yuv420p' or 'yuv444p'."
    resolved_pixel_format = OutputPixelFormats(pixel_format_lower)

    # Validates the quantization parameter is within the legal FFMPEG range.
    if not 0 <= quantization_parameter <= MAXIMUM_QUANTIZATION_VALUE:
        return (
            f"Error: quantization_parameter must be between 0 and {MAXIMUM_QUANTIZATION_VALUE}, "
            f"got {quantization_parameter}."
        )

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

        # Creates the VideoSystem with user-specified encoding parameters. The system_id 112 distinguishes
        # MCP-initiated sessions from CLI sessions (111) in log output.
        _active_session = VideoSystem(
            system_id=np.uint8(112),
            data_logger=_active_logger,
            name="live_camera",
            output_directory=output_path,
            camera_interface=camera_interface,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            frame_rate=frame_rate,
            display_frame_rate=display_frame_rate,
            color=not monochrome,
            gpu=gpu_index,
            video_encoder=resolved_encoder,
            encoder_speed_preset=resolved_preset,
            output_pixel_format=resolved_pixel_format,
            quantization_parameter=quantization_parameter,
        )

        # Spawns camera acquisition and encoding child processes. After this call, frames are being acquired but
        # not yet saved to disk (saving requires an explicit start_frame_saving call).
        _active_session.start()

        # Captures session configuration for status reporting. VideoSystem does not expose constructor parameters
        # as public properties, so they are stored here at creation time.
        _session_info = {
            "name": "live_camera",
            "interface": interface.lower(),
            "camera_index": camera_index,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "gpu_encoding": gpu_index >= 0,
            "monochrome": monochrome,
            "video_encoder": resolved_encoder.value,
            "encoder_speed_preset": encoder_speed_preset,
            "output_pixel_format": resolved_pixel_format.value,
            "quantization_parameter": quantization_parameter,
            "output_directory": output_directory,
            "display_frame_rate": display_frame_rate,
        }

    except Exception as e:
        # Cleans up partially initialized resources on failure to avoid leaving orphaned processes or file handles.
        if _active_logger is not None:
            _active_logger.stop()
            _active_logger = None
        _active_session = None
        _session_info = None
        return f"Error: {e}"
    else:
        return (
            f"Session started: {interface} #{camera_index} {width}x{height}@{frame_rate}fps "
            f"encoder={resolved_encoder.value} preset={encoder_speed_preset} "
            f"pixel_format={resolved_pixel_format.value} qp={quantization_parameter} -> {output_directory}"
        )


@mcp.tool()  # pragma: no cover
def stop_video_session() -> dict[str, Any]:  # pragma: no cover
    """Stops the active video capture session, releases all resources, and assembles log archives.

    Stops the VideoSystem and DataLogger, freeing the camera and saving any remaining buffered frames. After
    stopping the DataLogger, assembles raw .npy log entries into consolidated .npz archives grouped by source ID.

    Returns:
        A dictionary containing the session stop status, video file path, log directory path, archive assembly
        result, and source IDs found in assembled archives. Returns an error dictionary if no session is active or
        shutdown fails.
    """
    global _active_session, _active_logger, _session_info

    if _active_session is None:
        return {"error": "No active session"}

    # Captures output paths before stopping, since module-level references are cleared in the finally block.
    video_path = _active_session.video_file_path
    log_directory: Path | None = _active_logger.output_directory if _active_logger is not None else None

    try:
        # Stops the VideoSystem first, which terminates camera acquisition and flushes any buffered frames still in
        # the encoding pipeline. This may block briefly while remaining frames are written to disk.
        _active_session.stop()

        # Stops the DataLogger after the VideoSystem to ensure all runtime events are captured before the log is
        # finalized.
        if _active_logger is not None:
            _active_logger.stop()
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clears the module-level references regardless of success or failure to allow a new session to be started.
        _active_session = None
        _active_logger = None
        _session_info = None

    # Assembles log archives from the DataLogger output directory. Assembly converts raw .npy files into
    # consolidated .npz archives grouped by source ID, which the log processing pipeline requires as input.
    archives_assembled = False
    source_ids: list[str] = []
    if log_directory is not None:
        try:
            assemble_log_archives(log_directory=log_directory, remove_sources=True, verbose=False)
            archives_assembled = True
            source_ids = _scan_archive_source_ids(directory=log_directory)
        except Exception:  # noqa: S110
            # Archive assembly failure is non-fatal. The primary operation (stopping the session) has already
            # succeeded. The archives_assembled flag communicates the failure without raising an error.
            pass

    return {
        "status": "stopped",
        "video_file": str(video_path) if video_path is not None else None,
        "log_directory": str(log_directory) if log_directory is not None else None,
        "archives_assembled": archives_assembled,
        "source_ids": source_ids,
    }


@mcp.tool()  # pragma: no cover
def assemble_log_archives_tool(  # pragma: no cover
    log_directory: str,
    *,
    remove_sources: bool = True,
    verify_integrity: bool = False,
) -> dict[str, Any]:
    """Consolidates raw .npy log entries in a DataLogger output directory into .npz archives by source ID.

    Assembles the raw .npy files produced by a DataLogger instance into consolidated .npz archives, one per unique
    source ID. This is required before the log processing pipeline can extract frame timestamps.

    This tool is useful when log archives need to be assembled independently of a video session stop operation,
    for example when processing log directories from previous sessions or when the automatic assembly was skipped or
    failed.

    Important:
        The AI agent calling this tool MUST ask the user to provide the log_directory path before calling this
        tool. Do not assume or guess the log directory path.

    Args:
        log_directory: The absolute path to the DataLogger output directory containing raw .npy log entries. Must
            be provided by the user.
        remove_sources: Determines whether to remove the original .npy files after successful archive assembly.
        verify_integrity: Determines whether to verify archive integrity against original log entries before
            removing sources.

    Returns:
        A dictionary containing the assembly status, directory path, list of created archive filenames, extracted
        source IDs, and archive count. Returns an error dictionary if the directory does not exist or assembly
        fails.
    """
    directory_path = Path(log_directory)

    if not directory_path.exists():
        return {"error": f"Directory not found: {log_directory}"}

    if not directory_path.is_dir():
        return {"error": f"Not a directory: {log_directory}"}

    try:
        assemble_log_archives(
            log_directory=directory_path,
            remove_sources=remove_sources,
            verify_integrity=verify_integrity,
            verbose=False,
        )
    except Exception as e:
        return {"error": f"Archive assembly failed: {e}"}

    # Scans for created archives and extracts source IDs from filenames.
    source_ids = _scan_archive_source_ids(directory=directory_path)
    archives = [f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids]

    return {
        "status": "assembled",
        "directory": log_directory,
        "archives": archives,
        "source_ids": source_ids,
        "archive_count": len(archives),
    }


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
def get_session_status() -> dict[str, Any]:  # pragma: no cover
    """Returns detailed status information about the current video session.

    Reports whether a session is active, and when active, includes camera interface, resolution, frame rate,
    encoding parameters, video file path, and log directory.

    Returns:
        A dictionary containing session status and configuration details. Returns ``{"status": "inactive"}``
        when no session exists.
    """
    # No session object exists: either none was started or the previous session was fully torn down.
    if _active_session is None:
        return {"status": "inactive"}

    # Determines the high-level session state from the started flag.
    status = "running" if _active_session.started else "stopped"
    result: dict[str, Any] = {"status": status}

    # Enriches the response with session configuration captured at creation time.
    if _session_info is not None:
        result["name"] = _session_info["name"]
        result["interface"] = _session_info["interface"]
        result["camera_index"] = _session_info["camera_index"]
        result["resolution"] = f"{_session_info['width']}x{_session_info['height']}"
        result["frame_rate"] = _session_info["frame_rate"]
        result["monochrome"] = _session_info["monochrome"]
        result["encoder"] = _session_info["video_encoder"]
        result["encoder_speed_preset"] = _session_info["encoder_speed_preset"]
        result["output_pixel_format"] = _session_info["output_pixel_format"]
        result["quantization_parameter"] = _session_info["quantization_parameter"]
        result["gpu_encoding"] = _session_info["gpu_encoding"]
        result["output_directory"] = _session_info["output_directory"]
        result["display_frame_rate"] = _session_info["display_frame_rate"]

    # Appends runtime-derived information from the VideoSystem and DataLogger instances.
    video_path = _active_session.video_file_path
    result["video_file"] = str(video_path) if video_path is not None else None

    if _active_logger is not None:
        result["log_directory"] = str(_active_logger.output_directory)

    return result


@mcp.tool()  # pragma: no cover
def validate_video_file_tool(video_file: str) -> dict[str, Any]:  # pragma: no cover
    """Validates a video file and extracts metadata using ffprobe.

    Runs ffprobe on the specified video file to extract duration, frame count, codec, resolution, file size,
    and bit rate. Verifies video integrity after a recording session.

    Important:
        The AI agent calling this tool MUST ask the user to provide the video_file path before calling this
        tool. Do not assume or guess the video file path.

    Args:
        video_file: The absolute path to the video file to validate. Must be provided by the user.

    Returns:
        A dictionary containing video metadata (duration, frame count, codec, resolution, file size, bit rate)
        on success, or an error dictionary if the file cannot be read or ffprobe is not available.
    """
    file_path = Path(video_file)

    if not file_path.exists():
        return {"error": f"File not found: {video_file}"}

    if not file_path.is_file():
        return {"error": f"Not a file: {video_file}"}

    # Runs ffprobe to extract stream and format metadata in JSON format. Limits analysis to the first 10 MB and
    # 5 seconds of the file to avoid scanning the entire file for containers that lack header-level metadata.
    # Selects only the first video stream to skip audio, subtitle, and data streams.
    try:
        probe_result = subprocess.run(
            args=[
                "ffprobe",
                "-v",
                "quiet",
                "-probesize",
                "10000000",
                "-analyzeduration",
                "5000000",
                "-select_streams",
                "v:0",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return {"error": "ffprobe is not available on the system PATH. Install FFMPEG to use this tool."}
    except subprocess.CalledProcessError as e:
        return {"error": f"ffprobe failed: {e.stderr.strip() if e.stderr else 'unknown error'}"}

    # Parses the JSON output from ffprobe.
    try:
        probe_data = json.loads(probe_result.stdout)
    except json.JSONDecodeError:
        return {"error": "Unable to parse ffprobe output."}

    # Extracts the first video stream from the probe output.
    video_stream: dict[str, Any] | None = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        return {"error": "No video stream found in file."}

    format_info = probe_data.get("format", {})

    # Extracts frame count. ffprobe may report it as nb_frames or may not include it for some containers.
    frame_count_raw = video_stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw and frame_count_raw != "N/A" else None

    # Extracts duration from the format-level metadata (more reliable than stream-level for MP4 containers).
    duration_raw = format_info.get("duration")
    duration_seconds = round(float(duration_raw), 6) if duration_raw else None

    # Extracts bit rate from the format-level metadata.
    bit_rate_raw = format_info.get("bit_rate")
    bit_rate_bps = int(bit_rate_raw) if bit_rate_raw else None

    # Extracts file size, falling back to filesystem stat if ffprobe does not report it.
    size_raw = format_info.get("size")
    file_size_bytes = int(size_raw) if size_raw else file_path.stat().st_size

    return {
        "file": video_file,
        "valid": True,
        "codec": video_stream.get("codec_name"),
        "codec_long_name": video_stream.get("codec_long_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "bit_rate_bps": bit_rate_bps,
        "file_size_bytes": file_size_bytes,
        "pixel_format": video_stream.get("pix_fmt"),
        "frame_rate": video_stream.get("r_frame_rate"),
    }


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
        blacklisted_nodes: A list of GenICam node names to exclude from enumeration. When None, uses the built-in
            blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific nodes that report
            ReadWrite access but reject writes at the hardware level. Pass an empty list to disable blacklisting.

    Returns:
        Detailed node information for a single node, or a newline-separated summary of all nodes.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
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
    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Delegates value conversion and writing to the camera's set_node_value method, which inspects the node's
            # type and casts the string value to int, float, bool, or enum as needed.
            camera.set_node_value(name=node_name, value=value)
    except Exception as e:
        return f"Error: {e}"
    else:
        return f"Node '{node_name}' set to {value}"


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
        blacklisted_nodes: A list of GenICam node names to exclude from the configuration dump. When None, uses
            the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes vendor-specific
            nodes that report ReadWrite access but reject writes at the hardware level. Pass an empty list to
            disable blacklisting.

    Returns:
        A confirmation with the number of nodes saved, or an error description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
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
            When None, uses the built-in blacklist (CustomerIDKey, CustomerValueKey, TestPattern) which excludes
            vendor-specific nodes that report ReadWrite access but reject writes at the hardware level. Pass an
            empty list to disable blacklisting.

    Returns:
        The number of nodes applied and any errors encountered, or an error description.
    """
    blacklist = _resolve_blacklist(blacklisted_nodes=blacklisted_nodes)

    # Validates the config file path before opening a camera connection.
    path = Path(config_file)
    if not path.exists():
        return f"Error: File not found at {config_file}"

    try:
        with _harvester_connection(camera_index=camera_index) as camera:
            # Deserializes the YAML configuration and applies each writable node value to the connected camera. When
            # strict_identity is True, the camera model and serial number must match the values stored in the YAML
            # file; otherwise, a mismatch produces a warning but proceeds with the write.
            config = GenicamConfiguration.from_yaml(file_path=path)
            camera.apply_configuration(config, strict_identity=strict_identity, blacklisted_nodes=blacklist)
    except Exception as e:
        return f"Error: {e}"
    else:
        return "Configuration applied successfully"


@mcp.tool()  # pragma: no cover
def read_camera_manifest_tool(manifest_path: str) -> dict[str, Any]:  # pragma: no cover
    """Reads a camera manifest file and returns its contents.

    Reads the specified camera_manifest.yaml file and returns the list of camera sources registered in it.
    Each source entry contains the numeric source ID and the colloquial name assigned to the camera.

    Args:
        manifest_path: The absolute path to the camera_manifest.yaml file to read.

    Returns:
        A dictionary containing the list of sources with their IDs and names, or an error message.
    """
    path = Path(manifest_path)

    if not path.exists():
        return {"error": f"Manifest file does not exist: {manifest_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {manifest_path}"}

    try:
        manifest = CameraManifest.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Failed to read manifest: {error}"}

    return {
        "manifest_path": manifest_path,
        "sources": [{"id": source.id, "name": source.name} for source in manifest.sources],
        "total_sources": len(manifest.sources),
    }


@mcp.tool()  # pragma: no cover
def write_camera_manifest_tool(  # pragma: no cover
    log_directory: str,
    source_id: int,
    name: str,
) -> dict[str, Any]:
    """Writes or updates a camera manifest file in the specified log directory.

    Registers a camera source in the camera_manifest.yaml file located in the target log directory. If
    the manifest already exists, appends the new source entry. Otherwise, creates a new manifest. Use this
    tool to retroactively tag existing log archives as axvs-produced, or to manually register additional
    camera sources.

    Args:
        log_directory: The absolute path to the DataLogger output directory where the manifest file is stored.
        source_id: The numeric source ID to register in the manifest.
        name: The colloquial human-readable name for the camera source (e.g., 'face_camera').

    Returns:
        A dictionary confirming the write operation with the manifest path and registered source, or an error message.
    """
    dir_path = Path(log_directory)

    if not dir_path.exists():
        return {"error": f"Directory does not exist: {log_directory}"}

    if not dir_path.is_dir():
        return {"error": f"Path is not a directory: {log_directory}"}

    if not name:
        return {"error": "The 'name' parameter must be a non-empty string."}

    try:
        write_camera_manifest(log_directory=dir_path, source_id=source_id, name=name)
    except Exception as error:
        return {"error": f"Failed to write manifest: {error}"}

    manifest_path = dir_path / CAMERA_MANIFEST_FILENAME
    return {
        "manifest_path": str(manifest_path),
        "registered_source": {"id": source_id, "name": name},
        "status": "success",
    }


@mcp.tool()  # pragma: no cover
def discover_camera_data_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Discovers confirmed camera recordings under a root directory.

    Recursively searches for camera_manifest.yaml files to identify camera sources. Only sources whose log
    archives (``{source_id}_log.npz``) exist on disk are included. For each confirmed source, resolves the
    paired video file and processed timestamp feather file from pre-collected file indices. Video files are
    matched by camera name first, then by source ID pattern, preferring the closest match by path proximity
    to the log directory. Returns a flat list of resolved source entries.

    Args:
        root_directory: The absolute path to the root directory to search. Searched recursively.

    Returns:
        A dictionary containing a 'sources' list where each entry has 'recording_root', 'source_id', 'name',
        'log_archive', 'video_file', 'timestamps_file', and 'log_directory' keys, a flat 'log_directories'
        list for batch processing, and aggregate counts. Video and timestamp paths are None when the
        corresponding file cannot be found.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # Discovers all camera manifests and collects only sources whose log archives exist on disk.
    confirmed_sources: list[tuple[Path, int, str, Path]] = []
    log_dirs_with_archives: set[Path] = set()

    try:
        for manifest_path in sorted(root_path.rglob(CAMERA_MANIFEST_FILENAME)):
            log_dir = manifest_path.parent

            try:
                manifest = CameraManifest.from_yaml(file_path=manifest_path)
            except Exception:  # noqa: S112
                continue

            if not manifest.sources:
                continue

            for source in manifest.sources:
                archive_path = log_dir / f"{source.id}{LOG_ARCHIVE_SUFFIX}"
                if not archive_path.exists():
                    continue

                confirmed_sources.append((log_dir, source.id, source.name, archive_path))
                log_dirs_with_archives.add(log_dir)
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    if not confirmed_sources:
        return {
            "sources": [],
            "log_directories": [],
            "total_sources": 0,
            "total_log_directories": 0,
        }

    # Pre-collects all video files and camera_timestamps directories under the search root in two rglob passes.
    # Avoids redundant filesystem walks when resolving multiple sources.
    all_video_files = tuple(sorted(root_path.rglob("*.mp4")))
    timestamps_dirs = tuple(
        candidate for candidate in sorted(root_path.rglob(CAMERA_TIMESTAMPS_DIRECTORY)) if candidate.is_dir()
    )

    # Resolves recording roots and builds the log-directory-to-root mapping.
    log_dir_paths = sorted(log_dirs_with_archives)
    log_dir_to_root = _resolve_log_dir_roots(log_dir_paths=log_dir_paths)

    # Builds the flat list of resolved source entries. Each entry pairs the confirmed log archive with its
    # recording root, matched video file, and processed timestamp feather file.
    sources_output: list[dict[str, Any]] = []
    for log_dir, source_id, name, archive_path in confirmed_sources:
        video_path = _match_video_file(
            all_video_files=all_video_files, log_directory=log_dir, source_id=source_id, name=name
        )
        feather_path = _find_feather_file(timestamps_dirs=timestamps_dirs, source_id=source_id)

        sources_output.append(
            {
                "recording_root": str(log_dir_to_root[log_dir]),
                "source_id": str(source_id),
                "name": name,
                "log_archive": str(archive_path),
                "video_file": video_path,
                "timestamps_file": str(feather_path) if feather_path is not None else None,
                "log_directory": str(log_dir),
            }
        )

    return {
        "sources": sources_output,
        "log_directories": sorted(str(log_dir) for log_dir in log_dir_paths),
        "total_sources": len(sources_output),
        "total_log_directories": len(log_dir_paths),
    }


@mcp.tool()  # pragma: no cover
def prepare_log_processing_batch_tool(  # pragma: no cover
    log_directories: list[str],
    source_ids: list[str],
    output_directories: list[str] | None = None,
) -> dict[str, Any]:
    """Prepares an execution manifest for batch log processing without starting execution.

    Accepts log directories and source IDs from discover_camera_data_tool and initializes a ProcessingTracker
    with one timestamp-extraction job per source ID for each log directory. Idempotent: if a tracker already
    exists for a log directory, returns the existing manifest with current job statuses instead of
    reinitializing. Requires prior discovery — the caller must provide confirmed source IDs rather than
    relying on implicit archive or manifest discovery.

    Important:
        The AI agent calling this tool MUST run discover_camera_data_tool first to obtain log directory paths
        and confirmed source IDs. Do not assume or guess directory paths or source IDs.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_camera_data_tool.
        source_ids: The list of confirmed source IDs to process. Accepts IDs from the 'source_id' field of
            entries in the 'sources' list returned by discover_camera_data_tool. Applied uniformly: each log
            directory creates jobs for every source ID in this list that has a matching archive on disk.
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

    source_id_set = set(source_ids)
    result_log_dirs: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_dir_str in enumerate(log_directories):
        log_dir_path = Path(log_dir_str)

        if not log_dir_path.exists() or not log_dir_path.is_dir():
            invalid_paths.append(log_dir_str)
            continue

        # Filters the requested source IDs to those that have a matching archive in this log directory.
        # Discovery already confirmed these archives exist, but the check guards against stale data.
        filtered_ids = sorted(
            source_id for source_id in source_id_set if (log_dir_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}").exists()
        )

        if not filtered_ids:
            result_log_dirs[log_dir_str] = {"source_ids": [], "jobs": [], "tracker_path": None, "summary": {}}
            continue

        # Resolves the output directory for this log directory. Falls back to the log directory itself.
        output_path = Path(output_directories[entry_index]) if output_directories is not None else log_dir_path

        # Creates the camera_timestamps subdirectory under the output path for tracker and feather files.
        timestamps_path = output_path / CAMERA_TIMESTAMPS_DIRECTORY
        timestamps_path.mkdir(parents=True, exist_ok=True)
        tracker_path = timestamps_path / TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state.
            try:
                tracker_status = _read_tracker_status(tracker_path=tracker_path)
            except Exception:
                tracker_status = {"jobs": [], "summary": {}}

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(timestamps_path),
                "source_ids": filtered_ids,
                **tracker_status,
            }
            total_jobs += len(tracker_status.get("jobs", []))
        else:
            # Initializes a new tracker with jobs for the filtered source IDs.
            job_ids = initialize_processing_tracker(output_directory=timestamps_path, source_ids=filtered_ids)

            jobs: list[dict[str, str]] = [
                {
                    "job_id": job_ids[source_id],
                    "source_id": source_id,
                    "status": "SCHEDULED",
                    "log_directory": log_dir_str,
                    "output_directory": str(timestamps_path),
                    "tracker_path": str(tracker_path),
                }
                for source_id in filtered_ids
            ]

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(timestamps_path),
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
    worker_budget: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution with budget-based worker allocation.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager that allocates CPU cores to each job based on its archive size. The worker budget directly
    controls memory footprint since each worker spawns a separate process. Large archives (>= 2000 messages) receive
    more workers, while small archives receive 1 worker since they process sequentially regardless. The manager fills
    available budget greedily, dispatching smaller jobs alongside large ones when cores are available.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors, each a dictionary with 'log_directory', 'output_directory',
            'tracker_path', 'job_id', and 'source_id' keys.
        worker_budget: The total number of CPU cores available for the execution session. Directly controls memory
            footprint. Set to -1 for automatic resolution via resolve_worker_count.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved worker budget, per-job message counts, and
        any invalid jobs.
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
    all_jobs: dict[tuple[str, str], _PendingJob] = {}
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
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the total worker budget.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=_RESERVED_CORES)

    # Probes archive message counts for all pending jobs. This reads only the zip directory of each .npz file,
    # which is fast and does not load message data into memory.
    job_message_counts: dict[tuple[str, str], int] = {}
    for job in pending:
        job_message_counts[job.dispatch_key] = _probe_archive_message_count(job=job)

    # Creates execution state and starts the manager thread.
    _job_execution_state = _JobExecutionState(
        all_jobs=all_jobs,
        pending_queue=pending,
        job_message_counts=job_message_counts,
        worker_budget=resolved_budget,
    )

    manager = Thread(target=_job_execution_manager, daemon=True)
    manager.start()
    _job_execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "job_message_counts": job_message_counts,
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

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
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

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
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
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(duration_seconds, 2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level statistics.
    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            2,
        )

    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": sum(1 for j in job_timing if "elapsed_seconds" in j),
        "pending_count": len(state.all_jobs)
        - completed_count
        - failed_count
        - sum(1 for j in job_timing if "elapsed_seconds" in j),
    }

    if completed_count > 0 and earliest_start is not None:
        elapsed_hours = convert_time(
            time=current_us - earliest_start,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.HOUR,
            as_float=True,
        )
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
        active_count = len(state.active_groups)

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
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} group(s) still completing.",
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

            dir_status = _derive_tracker_status(summary=summary)

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
def analyze_camera_frame_statistics_tool(  # pragma: no cover
    feather_files: list[str],
    drop_threshold_us: int = 0,
    max_drop_locations: int = 50,
) -> dict[str, Any]:  # pragma: no cover
    """Reads one or more processed camera timestamp feather files and computes frame acquisition statistics.

    For each file, computes basic recording statistics (total frames, duration, estimated frame rate), inter-frame
    timing distribution (mean, median, standard deviation, min, max), and frame drop analysis (gap detection,
    estimated drop count, drop locations). Frame drops are identified as inter-frame intervals exceeding a threshold,
    which defaults to 2x the median inter-frame interval when not specified. Accepts the 'timestamps_file' paths
    returned by discover_camera_data_tool.

    Args:
        feather_files: The list of absolute paths to camera timestamp feather files produced by the log processing
            pipeline. Expected filename pattern: ``camera_{source_id}_timestamps.feather``. Accepts paths from the
            'timestamps_file' field returned by discover_camera_data_tool.
        drop_threshold_us: The inter-frame interval threshold in microseconds above which a gap is classified as a
            frame drop. When 0, the threshold is automatically computed as 2x the median inter-frame interval.
            Applied uniformly to all files.
        max_drop_locations: The maximum number of frame drop locations to include per file. Caps the
            'drop_locations' list to prevent oversized responses.

    Returns:
        A dictionary containing a 'results' list with per-file statistics (each with 'file', 'basic_stats',
        'inter_frame_timing', and 'frame_drop_analysis' keys) and a 'total_files' count. Files that cannot be
        read produce an entry with 'file' and 'error' keys instead of statistics.
    """
    results = [
        _analyze_single_feather(
            feather_file=feather_file, drop_threshold_us=drop_threshold_us, max_drop_locations=max_drop_locations
        )
        for feather_file in feather_files
    ]

    return {"results": results, "total_files": len(results)}


@mcp.tool()  # pragma: no cover
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]:  # pragma: no cover
    """Deletes the camera_timestamps subdirectory under one or more output directories.

    Removes each ``camera_timestamps/`` subdirectory and all of its contents, including processed feather files
    and the processing tracker. Uses ``delete_directory`` from ataraxis-data-structures for parallel file deletion
    with platform-safe retry logic. After cleanup, the output directories can be passed to
    prepare_log_processing_batch_tool to reinitialize from scratch. Accepts the 'log_directories' list returned
    by discover_camera_data_tool.

    Args:
        output_directories: The list of absolute paths to output directories containing ``camera_timestamps/``
            subdirectories to delete.

    Returns:
        A dictionary containing a 'results' list with per-directory outcomes (each with 'output_directory',
        'cleaned' flag, and either 'timestamps_path' or 'error') and a 'total_cleaned' count.
    """
    results = [_clean_single_output(output_directory=directory) for directory in output_directories]
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_directories": len(results)}


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


def _resolve_blacklist(blacklisted_nodes: list[str] | None) -> frozenset[str]:  # pragma: no cover
    """Resolves an optional blacklist parameter to a frozenset suitable for GenICam configuration functions.

    Converts a list of node names to a frozenset when provided, or returns the library default blacklist when None.

    Args:
        blacklisted_nodes: A list of GenICam node names to exclude, or None to use the default blacklist.

    Returns:
        A frozenset of blacklisted node names.
    """
    return frozenset(blacklisted_nodes) if blacklisted_nodes is not None else DEFAULT_BLACKLISTED_NODES


@contextlib.contextmanager  # pragma: no cover
def _harvester_connection(camera_index: int) -> Generator[HarvestersCamera, None, None]:  # pragma: no cover
    """Opens a temporary connection to a Harvesters camera and guarantees disconnection on exit.

    Creates a HarvestersCamera with system_id=0 (placeholder, since only node map access is needed) and connects it.
    The camera is always disconnected in the finally block, releasing the GenTL handle for other processes.

    Args:
        camera_index: The index of the Harvesters camera to connect to.

    Yields:
        The connected HarvestersCamera instance.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        yield camera
    finally:
        camera.disconnect()


def _scan_archive_source_ids(directory: Path) -> list[str]:  # pragma: no cover
    """Scans a directory for assembled log archives and extracts source IDs from their filenames.

    Matches files ending with the log archive suffix and strips the suffix to recover the source ID string. Results
    are returned in sorted order.

    Args:
        directory: The directory to scan for log archives.

    Returns:
        A sorted list of source ID strings extracted from archive filenames.
    """
    return sorted(
        source_id
        for archive_path in directory.glob(f"*{LOG_ARCHIVE_SUFFIX}")
        if (source_id := archive_path.name.removesuffix(LOG_ARCHIVE_SUFFIX))
    )


def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]:  # pragma: no cover
    """Resolves each log directory to its recording root.

    Uses unique path component detection to identify recording session boundaries. Falls back to using each
    log directory's parent when unique component detection fails (e.g., single log directory).

    Args:
        log_dir_paths: The sorted list of log directory paths to resolve.

    Returns:
        A mapping from each log directory to its recording root path.
    """
    try:
        recording_roots = resolve_recording_roots(paths=log_dir_paths)
    except RuntimeError:
        recording_roots = tuple(dict.fromkeys(log_dir.parent for log_dir in log_dir_paths))

    log_dir_to_root: dict[Path, Path] = {}
    for log_dir in log_dir_paths:
        for root in recording_roots:
            if log_dir == root or root in log_dir.parents:
                log_dir_to_root[log_dir] = root
                break
        else:
            log_dir_to_root[log_dir] = log_dir.parent

    return log_dir_to_root


def _match_video_file(  # pragma: no cover
    all_video_files: tuple[Path, ...],
    log_directory: Path,
    source_id: int,
    name: str,
) -> str | None:  # pragma: no cover
    """Matches a confirmed source to a pre-collected video file by name or source ID.

    Searches the pre-collected video file list for a match, preferring the closest file by path proximity
    to the log directory. Tries the camera name pattern first (``{name}`` in filename stem), then falls back
    to the source ID pattern (``{source_id:03d}`` in filename stem). When multiple candidates match, selects
    the one sharing the most leading path components with the log directory.

    Args:
        all_video_files: Pre-collected ``.mp4`` file paths from the search root.
        log_directory: The directory containing the camera manifest. Used as the proximity reference.
        source_id: The numeric source ID from the manifest.
        name: The colloquial camera name from the manifest (e.g., ``'body_camera'``). Tried first before
            falling back to the source ID.

    Returns:
        The string path to the matched video file, or None if no match is found.
    """
    log_parts = log_directory.parts

    # Counts leading path components shared between a candidate video and the log directory. Higher values
    # indicate closer proximity in the directory tree.
    def proximity(video_path: Path) -> int:
        shared = 0
        for log_part, video_part in zip(log_parts, video_path.parts, strict=False):
            if log_part != video_part:
                break
            shared += 1
        return shared

    # Tries name-based matching first, since users may rename video files to meaningful names.
    if name:
        name_matches = [video for video in all_video_files if name in video.stem]
        if name_matches:
            return str(max(name_matches, key=proximity))

    # Falls back to source ID pattern using the zero-padded VideoSystem naming convention.
    id_pattern = f"{source_id:03d}"
    id_matches = [video for video in all_video_files if id_pattern in video.stem]
    if id_matches:
        return str(max(id_matches, key=proximity))

    return None


def _find_feather_file(timestamps_dirs: tuple[Path, ...], source_id: int) -> Path | None:  # pragma: no cover
    """Searches pre-discovered ``camera_timestamps/`` directories for a processed feather file matching a source ID.

    Performs a flat (non-recursive) glob inside each ``camera_timestamps/`` directory for a feather file matching the
    ``camera_{source_id}_timestamps.feather`` naming convention. The caller is responsible for pre-discovering
    ``camera_timestamps/`` directories via a single ``rglob`` pass over the search root.

    Args:
        timestamps_dirs: Pre-discovered ``camera_timestamps/`` directory paths collected from the search root.
        source_id: The numeric source ID to search for.

    Returns:
        The path to the feather file, or None if not found.
    """
    pattern = f"{_FEATHER_PREFIX}{source_id}{_FEATHER_SUFFIX}"
    for timestamps_dir in timestamps_dirs:
        matches = list(timestamps_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _derive_tracker_status(summary: dict[str, Any]) -> str:  # pragma: no cover
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if any
    are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise.

    Args:
        summary: A dictionary containing 'total', 'succeeded', 'failed', 'running', and 'scheduled' counts.

    Returns:
        A status string: one of 'failed', 'completed', 'processing', 'not_started', or 'in_progress'.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0) > 0:
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


def _group_jobs_by_tracker(state: _JobExecutionState) -> dict[Path, list[_PendingJob]]:  # pragma: no cover
    """Groups all jobs in an execution state by their tracker file path.

    Minimizes redundant file reads by batching jobs that share the same tracker, so each tracker YAML file is
    deserialized only once when iterating over the groups.

    Args:
        state: The active job execution state containing the job registry.

    Returns:
        A dictionary mapping each tracker path to its list of pending job descriptors.
    """
    tracker_jobs: dict[Path, list[_PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


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


def _analyze_single_feather(  # pragma: no cover
    feather_file: str,
    drop_threshold_us: int,
    max_drop_locations: int,
) -> dict[str, Any]:  # pragma: no cover
    """Reads a single camera timestamp feather file and computes frame acquisition statistics.

    Args:
        feather_file: The absolute path to the feather file.
        drop_threshold_us: The inter-frame interval threshold in microseconds. When 0, auto-detected as 2x median.
        max_drop_locations: The maximum number of frame drop locations to include.

    Returns:
        A dictionary containing 'file', 'basic_stats', 'inter_frame_timing', and 'frame_drop_analysis' keys,
        or 'file' and 'error' keys if the file cannot be read.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return {"file": feather_file, "error": f"File does not exist: {feather_file}"}

    if not file_path.is_file():
        return {"file": feather_file, "error": f"Path is not a file: {feather_file}"}

    # Reads the feather file and validates the expected schema.
    try:
        dataframe = pl.read_ipc(source=file_path)
    except Exception as error:
        return {"file": feather_file, "error": f"Unable to read feather file: {error}"}

    if "frame_time_us" not in dataframe.columns:
        return {"file": feather_file, "error": f"Missing required 'frame_time_us' column. Found: {dataframe.columns}"}

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
    duration_seconds = round(
        convert_time(time=duration_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True), 6
    )
    estimated_fps = round((total_frames - 1) / duration_seconds, 3) if duration_seconds > 0 else 0.0

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
        longest_gap_ms = round(
            convert_time(
                time=longest_gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
            ),
            4,
        )

        # Builds the capped drop locations list.
        drop_locations: list[dict[str, Any]] = []
        for index in drop_indices[:max_drop_locations]:
            gap_us = int(intervals_us[index])
            gap_ms = round(
                convert_time(
                    time=gap_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
                ),
                4,
            )
            estimated_lost = max(round(gap_us / expected_interval) - 1, 0)
            drop_locations.append(
                {
                    "frame_index": int(index),
                    "gap_us": gap_us,
                    "gap_ms": gap_ms,
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

    # Converts inter-frame interval statistics from microseconds to milliseconds.
    mean_ms, median_ms, std_ms, min_ms, max_ms = (
        round(
            convert_time(time=value, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True),
            4,
        )
        for value in (mean_us, median_us, std_us, min_us, max_us)
    )

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
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        },
        "frame_drop_analysis": frame_drop_analysis,
    }


def _clean_single_output(output_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Deletes the camera_timestamps subdirectory under a single output directory.

    Args:
        output_directory: The absolute path to the output directory.

    Returns:
        A dictionary containing 'output_directory', 'cleaned' flag, and either 'timestamps_path' or 'error'.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"output_directory": output_directory, "cleaned": False, "error": "Directory does not exist."}

    if not output_path.is_dir():
        return {"output_directory": output_directory, "cleaned": False, "error": "Path is not a directory."}

    timestamps_path = output_path / CAMERA_TIMESTAMPS_DIRECTORY

    if not timestamps_path.exists():
        return {"output_directory": output_directory, "cleaned": True, "message": "Nothing to clean."}

    try:
        delete_directory(directory_path=timestamps_path)
    except Exception as error:
        return {
            "output_directory": output_directory,
            "cleaned": False,
            "timestamps_path": str(timestamps_path),
            "error": f"Unable to delete: {error}",
        }

    return {"output_directory": output_directory, "cleaned": True, "timestamps_path": str(timestamps_path)}


def _probe_archive_message_count(job: _PendingJob) -> int:  # pragma: no cover
    """Probes the message count of a job's log archive by reading the .npz zip directory.

    Reconstructs the archive path from the job's log directory and source ID, then reads the file list from the .npz
    archive without loading any message data. The message count is the total entry count minus one (excluding the onset
    message).

    Args:
        job: The pending job descriptor containing the log directory and source ID.

    Returns:
        The number of data messages in the archive, or 0 if the archive cannot be read.
    """
    archive_path = job.log_directory / f"{job.source_id}{LOG_ARCHIVE_SUFFIX}"
    if not archive_path.exists():
        return 0

    try:
        with np.load(file=archive_path, allow_pickle=False) as archive:
            return max(0, len(archive.files) - 1)
    except Exception:
        return 0


def _compute_sqrt_minimum(message_count: int) -> int:  # pragma: no cover
    """Computes the minimum useful worker count for an archive based on square root scaling.

    The formula ``ceil(sqrt(messages / _WORKER_SCALING_FACTOR))`` models diminishing returns from additional
    workers. The result is snapped to the nearest multiple of ``_WORKER_MULTIPLE`` for clean allocation. Archives
    below the parallel processing threshold always return 1.

    Args:
        message_count: The number of data messages in the job's archive.

    Returns:
        The minimum number of workers that meaningfully benefit this archive size.
    """
    if message_count < PARALLEL_PROCESSING_THRESHOLD:
        return 1

    raw = int(np.ceil(np.sqrt(message_count / _WORKER_SCALING_FACTOR)))
    if raw <= 1:
        return 1

    return max(_WORKER_MULTIPLE, round(raw / _WORKER_MULTIPLE) * _WORKER_MULTIPLE)


def _group_worker(jobs: list[_PendingJob], workers: int, state: _JobExecutionState) -> None:  # pragma: no cover
    """Executes a group of jobs sequentially using a shared ProcessPoolExecutor.

    Creates one ProcessPoolExecutor for the entire group and processes each job in sequence, reusing the pool
    across all archives. This avoids the overhead of spawning and tearing down worker processes for each individual
    archive. Checks for cancellation between jobs to allow responsive shutdown. If a job's tracker is not updated
    to a terminal state, marks it as failed.

    Args:
        jobs: The list of pending job descriptors to process sequentially.
        workers: The number of CPU cores allocated to this group's ProcessPoolExecutor.
        state: The execution state, checked for cancellation between jobs.
    """
    shared_executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for job in jobs:
            # Checks for cancellation between jobs so the group stops promptly.
            if state.canceled:
                break

            tracker = ProcessingTracker(file_path=job.tracker_path)

            # execute_job already calls tracker.fail_job on exception, so the tracker state is updated. The
            # exception is suppressed here to prevent it from terminating the group worker thread.
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
                    executor=shared_executor,
                )

            # Failsafe: if the tracker was not updated to a terminal state, marks the job as failed.
            try:
                reloaded = ProcessingTracker.from_yaml(file_path=job.tracker_path)
                if job.job_id in reloaded.jobs:
                    status = reloaded.jobs[job.job_id].status
                    if status not in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
                        tracker.fail_job(
                            job_id=job.job_id, error_message="Job terminated without updating tracker status."
                        )
            except Exception:  # noqa: S110
                pass
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)


def _job_execution_manager() -> None:  # pragma: no cover
    """Dispatches queued jobs as worker-tier groups with shared process pools.

    Runs as a daemon thread, polling at 1-second intervals. Each dispatch cycle classifies pending jobs into
    small (< 2000 messages, 1 worker each) and parallel (>= 2000 messages). Parallel jobs are grouped by their
    precomputed worker tier from ``_compute_sqrt_minimum``, which snaps archive sizes to discrete worker counts
    (multiples of 5). Jobs in the same tier share a single ProcessPoolExecutor sized exactly to that tier.
    Each tier is split into as many concurrent groups as the budget allows. Small jobs are dispatched individually.
    Exits when the queue is empty and no active groups remain.
    """
    if _job_execution_state is None:
        return

    state = _job_execution_state
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    while True:
        with state.lock:
            # Removes completed groups and frees their budget.
            state.active_groups = [group for group in state.active_groups if group.thread.is_alive()]

            # Exits when no pending jobs and no active groups remain.
            if not state.pending_queue and not state.active_groups:
                break

            # Stops dispatching new groups if canceled. Waits for active groups to finish.
            if state.canceled:
                if not state.active_groups:
                    break
            else:
                available = state.worker_budget - sum(group.workers for group in state.active_groups)
                if available < 1:
                    poll_timer.delay(delay=1, allow_sleep=True)
                    continue

                # Classifies pending jobs into small (sequential) and parallel.
                small_pending: list[_PendingJob] = []
                parallel_pending: list[_PendingJob] = []
                for job in state.pending_queue:
                    message_count = state.job_message_counts.get(job.dispatch_key, 0)
                    if message_count < PARALLEL_PROCESSING_THRESHOLD:
                        small_pending.append(job)
                    else:
                        parallel_pending.append(job)

                dispatch_groups: list[tuple[list[_PendingJob], int]] = []

                # Phase 1: Groups parallel jobs by worker tier. Each job's optimal worker count is precomputed
                # via _compute_sqrt_minimum, which snaps archive sizes to discrete tiers (multiples of 5). Jobs
                # in the same tier share a ProcessPoolExecutor sized exactly to that tier. Each tier is split
                # into as many concurrent groups as the available budget allows.
                if parallel_pending and available >= _WORKER_MULTIPLE:
                    worker_tiers: dict[int, list[_PendingJob]] = {}
                    for job in parallel_pending:
                        tier = _compute_sqrt_minimum(message_count=state.job_message_counts.get(job.dispatch_key, 0))
                        worker_tiers.setdefault(tier, []).append(job)

                    # Dispatches tiers from largest to smallest so large archives get budget priority.
                    for tier_workers in sorted(worker_tiers, reverse=True):
                        if available < tier_workers:
                            continue

                        tier_jobs = worker_tiers[tier_workers]
                        max_concurrent = available // tier_workers
                        concurrent = min(max_concurrent, len(tier_jobs))

                        # Splits tier jobs evenly across concurrent groups via chunking.
                        chunk_size = -(-len(tier_jobs) // concurrent)  # Ceiling division.
                        for start in range(0, len(tier_jobs), chunk_size):
                            chunk = tier_jobs[start : start + chunk_size]
                            dispatch_groups.append((chunk, tier_workers))
                            available -= tier_workers

                # Phase 2: Fills remaining budget with small jobs (1 worker each, dispatched individually).
                for job in small_pending:
                    if available < 1:
                        break
                    dispatch_groups.append(([job], 1))
                    available -= 1

                # Dispatches all groups.
                for group_jobs, workers in dispatch_groups:
                    for job in group_jobs:
                        state.pending_queue.remove(job)

                    thread = Thread(
                        target=_group_worker,
                        kwargs={"jobs": group_jobs, "workers": workers, "state": state},
                        daemon=True,
                    )
                    thread.start()
                    state.active_groups.append(
                        _ActiveGroup(
                            source_id=group_jobs[0].source_id,
                            jobs=group_jobs,
                            workers=workers,
                            thread=thread,
                        )
                    )

        # Polls at 1-second intervals outside the lock to avoid blocking other threads.
        poll_timer.delay(delay=1, allow_sleep=True)
