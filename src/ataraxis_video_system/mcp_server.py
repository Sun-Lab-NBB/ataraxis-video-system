"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

This module exposes camera discovery, CTI file management, runtime requirements checking, and video session
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
from .camera import CameraInterfaces, add_cti_file, check_cti_file, discover_camera_ids
from .video_system import VideoSystem

# Initializes the MCP server instance.
mcp = FastMCP(name="ataraxis-video-system", json_response=True)

# Module-level state for video session management.
_active_session: VideoSystem | None = None
_active_logger: DataLogger | None = None


@mcp.tool()
def list_cameras() -> str:
    """Discovers all cameras compatible with the OpenCV and Harvesters interfaces.

    Returns a formatted string containing information about all discovered cameras, including their interface type,
    index, frame dimensions, and frame rate. For Harvesters cameras, model and serial number information is also
    included.
    """
    all_cameras = discover_camera_ids()

    if len(all_cameras) == 0:
        return "No cameras discovered on the system."

    # Separates cameras by interface for display purposes.
    opencv_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.OPENCV]
    harvesters_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]

    result_lines: list[str] = []

    # Formats OpenCV camera information.
    if len(opencv_cameras) > 0:
        result_lines.append("OpenCV Cameras:")
        for num, camera_data in enumerate(opencv_cameras, start=1):
            result_lines.append(f"  {num}. Camera {camera_data.camera_index}")
            result_lines.append(f"     • Resolution: {camera_data.frame_width}×{camera_data.frame_height} px")
            result_lines.append(f"     • Frame rate: {camera_data.acquisition_frame_rate} fps")
    else:
        result_lines.append("OpenCV Cameras: None discovered")

    # Formats Harvesters camera information.
    if len(harvesters_cameras) > 0:
        result_lines.append("\nHarvesters Cameras:")
        for num, camera_data in enumerate(harvesters_cameras, start=1):
            result_lines.append(f"  {num}. Camera {camera_data.camera_index}")
            result_lines.append(f"     • Model: {camera_data.model}")
            result_lines.append(f"     • Serial: {camera_data.serial_number}")
            result_lines.append(f"     • Resolution: {camera_data.frame_width}×{camera_data.frame_height} px")
            result_lines.append(f"     • Frame rate: {camera_data.acquisition_frame_rate} fps")
    else:
        result_lines.append("\nHarvesters Cameras: None discovered")

    return "\n".join(result_lines)


@mcp.tool()
def get_cti_status() -> str:
    """Checks whether the library is configured with a valid GenTL Producer interface (.cti) file.

    The Harvesters camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    GeniCam-compatible cameras. Returns the configuration status and the path to the configured CTI file if one exists.
    """
    cti_path = check_cti_file()

    if cti_path is not None:
        return f"CTI Status: Configured\n• Path: {cti_path}"
    return "CTI Status: Not configured\n• Use set_cti_file to configure a GenTL Producer (.cti) file"


@mcp.tool()
def set_cti_file(file_path: str) -> str:
    """Configures the library to use the specified CTI file for all future runtimes involving GeniCam cameras.

    The Harvesters library requires the GenTL Producer interface (.cti) file to discover and interface with compatible
    cameras. This tool must be called at least once before using the Harvesters interface.

    Args:
        file_path: The absolute path to the CTI file that provides the GenTL Producer interface. It is recommended to
            use the file supplied by the camera vendor, but a general Producer such as mvImpactAcquire is also
            acceptable.
    """
    path = Path(file_path)

    if not path.exists():
        return f"Error: File not found\n• Path: {file_path}"

    if not path.is_file():
        return f"Error: Not a file\n• Path: {file_path}"

    try:
        add_cti_file(cti_path=path)
    except Exception as e:
        return f"Error: Configuration failed\n• Details: {e}"
    else:
        return f"CTI File Configured\n• Path: {path}"


@mcp.tool()
def check_runtime_requirements() -> str:
    """Checks whether the host system meets the requirements for video encoding and camera interfaces.

    Verifies that FFMPEG is installed and accessible, checks for Nvidia GPU availability for hardware-accelerated
    encoding, and checks whether a CTI file is configured for Harvesters camera support. Returns a status indicating
    whether requirements are fully met, partially met, or not met.
    """
    ffmpeg_available = check_ffmpeg_availability()
    gpu_available = check_gpu_availability()
    cti_path = check_cti_file()

    # Builds the status string based on available components.
    ffmpeg_status = "Available" if ffmpeg_available else "Not installed or not accessible"
    gpu_status = "Available" if gpu_available else "Not available (no Nvidia GPU detected)"
    cti_status = f"Configured ({cti_path})" if cti_path is not None else "Not configured"

    # Determines overall status.
    if not ffmpeg_available:
        overall_status = "NOT MET"
    elif not gpu_available or cti_path is None:
        overall_status = "PARTIAL"
    else:
        overall_status = "FULLY MET"

    return (
        f"Runtime Requirements: {overall_status}\n"
        f"• FFMPEG: {ffmpeg_status}\n"
        f"• GPU Encoding: {gpu_status}\n"
        f"• CTI File: {cti_status}"
    )


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
        monochrome: Determines whether to capture in grayscale. Defaults to False (color).
        gpu_index: The GPU index for hardware encoding, or -1 for CPU encoding. Defaults to -1.
        display_frame_rate: The rate at which to display acquired frames in a preview window. Defaults to 25 fps.
            Set to None to disable frame display. The display rate cannot exceed the acquisition frame rate.
            Note that frame display is not supported on macOS.
    """
    global _active_session, _active_logger

    # Checks if a session is already active.
    if _active_session is not None:
        return "Error: Session already active\n• Stop the current session before starting a new one"

    # Validates the output directory.
    output_path = Path(output_directory)
    if not output_path.exists():
        return f"Error: Directory not found\n• Path: {output_directory}"
    if not output_path.is_dir():
        return f"Error: Not a directory\n• Path: {output_directory}"

    # Resolves the camera interface.
    if interface.lower() == "mock":
        camera_interface = CameraInterfaces.MOCK
    elif interface.lower() == "harvesters":
        camera_interface = CameraInterfaces.HARVESTERS
    else:
        camera_interface = CameraInterfaces.OPENCV

    try:
        # Initializes and starts the DataLogger.
        _active_logger = DataLogger(output_directory=output_path, instance_name="mcp_video_session")
        _active_logger.start()

        # Initializes the VideoSystem.
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

        # Starts the video system.
        _active_session.start()

    except Exception as e:
        # Cleans up on failure.
        if _active_logger is not None:
            _active_logger.stop()
            _active_logger = None
        _active_session = None
        return f"Error: Failed to start session\n• Details: {e}"
    else:
        display_status = f"{display_frame_rate} fps" if display_frame_rate is not None else "Disabled"
        return (
            f"Video Session Started\n"
            f"• Interface: {interface}\n"
            f"• Camera: {camera_index}\n"
            f"• Resolution: {width}×{height} px\n"
            f"• Frame rate: {frame_rate} fps\n"
            f"• Display: {display_status}\n"
            f"• Output: {output_directory}"
        )


@mcp.tool()
def stop_video_session() -> str:
    """Stops the active video capture session and releases all resources.

    Stops the VideoSystem and DataLogger, freeing the camera and saving any remaining buffered frames.
    """
    global _active_session, _active_logger

    if _active_session is None:
        return "Error: No active session\n• No video session is currently running"

    try:
        _active_session.stop()
        if _active_logger is not None:
            _active_logger.stop()
    except Exception as e:
        return f"Error: Failed to stop session\n• Details: {e}"
    finally:
        _active_session = None
        _active_logger = None

    return "Video Session Stopped\n• Camera released\n• Resources freed"


@mcp.tool()
def start_frame_saving() -> str:
    """Starts saving captured frames to the video file.

    Begins writing acquired frames to an MP4 video file in the output directory. A video session must be active.
    """
    if _active_session is None:
        return "Error: No active session\n• Start a video session first using start_video_session"

    try:
        _active_session.start_frame_saving()
    except Exception as e:
        return f"Error: Failed to start recording\n• Details: {e}"
    else:
        return "Recording Started\n• Frames are being saved to video file"


@mcp.tool()
def stop_frame_saving() -> str:
    """Stops saving frames to the video file.

    Stops writing frames to the video file while keeping the session active. Frame acquisition continues.
    """
    if _active_session is None:
        return "Error: No active session\n• No video session is currently running"

    try:
        _active_session.stop_frame_saving()
    except Exception as e:
        return f"Error: Failed to stop recording\n• Details: {e}"
    else:
        return "Recording Stopped\n• Session remains active\n• Frame acquisition continues"


@mcp.tool()
def get_session_status() -> str:
    """Returns the current status of the video session.

    Reports whether a session is active and its current state (acquiring frames, saving frames, etc.).
    """
    if _active_session is None:
        return "Session Status: Inactive\n• No video session is currently running"

    if _active_session.started:
        return "Session Status: Active\n• State: Running (acquiring frames)"
    else:
        return "Session Status: Active\n• State: Stopped"


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output communication
            and 'streamable-http' for HTTP-based communication.
    """
    mcp.run(transport=transport)
