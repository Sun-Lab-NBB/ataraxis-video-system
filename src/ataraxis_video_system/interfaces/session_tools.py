"""Provides MCP tools for managing the lifecycle of a single active video capture session."""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_data_structures import DataLogger, assemble_log_archives

from ..video import (
    MAXIMUM_QUANTIZATION_VALUE,
    VideoSystem,
    VideoEncoders,
    CameraInterfaces,
    OutputPixelFormats,
    EncoderSpeedPresets,
)
from .mcp_instance import mcp, scan_archive_source_ids

_active_session: VideoSystem | None = None
"""Stores the currently active VideoSystem instance, or None when no session is running."""

_active_logger: DataLogger | None = None
"""Stores the DataLogger instance associated with the active video session, or None when no session is running."""

_session_info: dict[str, Any] | None = None
"""Stores session configuration parameters captured at creation time for status reporting."""


@mcp.tool()
def start_video_session_tool(
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
) -> str:
    """Starts a video capture session with the specified parameters.

    Creates a VideoSystem instance and begins acquiring frames from the camera. Frames are not saved until
    start_frame_saving_tool is called. Only one session can be active at a time.

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
        # not yet saved to disk (saving requires an explicit start_frame_saving_tool call).
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

    except Exception as error:
        # Cleans up partially initialized resources on failure to avoid leaving orphaned processes or file handles.
        if _active_logger is not None:
            _active_logger.stop()
            _active_logger = None
        _active_session = None
        _session_info = None
        return f"Error: {error}"
    else:
        return (
            f"Session started: {interface} #{camera_index} {width}x{height}@{frame_rate}fps "
            f"encoder={resolved_encoder.value} preset={encoder_speed_preset} "
            f"pixel_format={resolved_pixel_format.value} qp={quantization_parameter} -> {output_directory}"
        )


@mcp.tool()
def stop_video_session_tool() -> dict[str, Any]:
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
    except Exception as error:
        return {"error": str(error)}
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
            source_ids = scan_archive_source_ids(directory=log_directory)
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


@mcp.tool()
def start_frame_saving_tool() -> str:
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
    except Exception as error:
        return f"Error: {error}"
    else:
        return "Recording started"


@mcp.tool()
def stop_frame_saving_tool() -> str:
    """Stops saving frames to the video file.

    Stops writing frames to the video file while keeping the session active. Frame acquisition continues.

    Returns:
        A confirmation that recording has stopped, or an error message if no session is active or the operation fails.
    """
    if _active_session is None:
        return "Error: No active session"

    # Signals the saver process to stop accepting new frames and finalize the current video file. The camera
    # continues acquiring frames, so a subsequent start_frame_saving_tool call will create a new video file.
    try:
        _active_session.stop_frame_saving()
    except Exception as error:
        return f"Error: {error}"
    else:
        return "Recording stopped"


@mcp.tool()
def get_session_status_tool() -> dict[str, Any]:
    """Returns detailed status information about the current video session.

    Reports whether a session is active. When active, the response includes the session configuration captured at
    creation time (camera name, interface, resolution, frame rate, encoding parameters, output directory, and
    display frame rate) together with the runtime video file path and log directory.

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
