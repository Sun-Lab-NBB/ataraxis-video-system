"""Provides MCP tools for camera discovery, GenTL Producer (.cti) file management, and runtime requirement checks."""

from pathlib import Path

from ..video import (
    GENICAM_UNAVAILABLE_REASON,
    CameraInterfaces,
    add_cti_file,
    check_cti_file,
    discover_camera_ids,
    check_gpu_availability,
    check_ffmpeg_availability,
    genicam_runtime_available,
)
from .mcp_instance import mcp


@mcp.tool()
def list_cameras_tool() -> str:
    """Discovers all cameras compatible with the OpenCV and Harvesters interfaces.

    Returns:
        A newline-separated list of discovered cameras, each showing interface type, index, frame dimensions, and
        frame rate. Harvesters cameras also include model and serial number. Returns a "No cameras discovered"
        message if no cameras are found. A trailing note reports the skipped Harvesters discovery when the GenICam
        camera runtime is unavailable.
    """
    # OpenCV cameras are probed by iterating over positional indices, while Harvesters cameras are enumerated through
    # the GenTL Producer.
    all_cameras = discover_camera_ids()

    # Names the skipped interface, so that an empty or OpenCV-only listing is not read as an absence of GenICam
    # hardware on a host that is unable to enumerate it in the first place.
    skip_note = "" if genicam_runtime_available() else f"\nHarvesters discovery skipped. {GENICAM_UNAVAILABLE_REASON}"

    if not all_cameras:
        return f"No cameras discovered on the system.{skip_note}"

    # Harvesters cameras include model and serial number because the GenTL interface exposes this metadata, whereas
    # OpenCV does not.
    lines = [
        f"OpenCV #{camera.camera_index}: {camera.frame_width}x{camera.frame_height}@{camera.acquisition_frame_rate}fps"
        if camera.interface == CameraInterfaces.OPENCV
        else f"Harvesters #{camera.camera_index}: {camera.model} ({camera.serial_number}) "
        f"{camera.frame_width}x{camera.frame_height}@{camera.acquisition_frame_rate}fps"
        for camera in all_cameras
    ]

    return "\n".join(lines) + skip_note


@mcp.tool()
def get_cti_status_tool() -> str:
    """Checks whether the library is configured with a valid GenTL Producer interface (.cti) file.

    The Harvesters camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    GenICam-compatible cameras.

    Returns:
        The configuration status and the path to the configured CTI file, or a "Not configured" message if no valid
        CTI file is set. Reports the unavailable interface instead when the GenICam camera runtime is unavailable.
    """
    # Reports the unavailable interface before the configuration, since check_cti_file() reports None for both an
    # unconfigured Producer and an absent runtime, and directing the caller to configure a Producer it can never load
    # would send it down a dead end.
    if not genicam_runtime_available():
        return f"CTI: Unavailable. {GENICAM_UNAVAILABLE_REASON}"

    # Reads the persisted CTI file path from the library's configuration storage and verifies that the file still
    # exists on disk. Returns None if no path was previously set or the stored path no longer points to a valid file.
    cti_path = check_cti_file()

    if cti_path is not None:
        return f"CTI: {cti_path}"
    return "CTI: Not configured"


@mcp.tool()
def set_cti_file_tool(file_path: str) -> str:
    """Configures the library to use the specified CTI file for all future runtimes involving GenICam cameras.

    The Harvesters library requires the GenTL Producer interface (.cti) file to discover and interface with compatible
    cameras. This tool must be called at least once before using the Harvesters interface, unless the ``AXVS_CTI_PATH``
    environment variable supplies the Producer path for the runtime.

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
    except Exception as error:
        return f"Error: {error}"
    else:
        return f"CTI configured: {path}"


@mcp.tool()
def check_runtime_requirements_tool() -> str:
    """Checks whether the host system meets the requirements for video encoding and camera interfaces.

    Verifies that FFMPEG is installed and accessible, checks for Nvidia GPU availability for hardware-accelerated
    encoding, and checks whether a CTI file is configured for Harvesters camera support.

    Returns:
        A pipe-separated status line showing FFMPEG, GPU, and CTI availability, each marked as "OK", "Missing", or
        "None". The CTI field instead reads "Unsupported" when the GenICam camera runtime is unavailable.
    """
    # Probes the system for each runtime dependency independently. FFMPEG is required for any video encoding, GPU is
    # optional (enables hardware-accelerated H.264/H.265 encoding via NVENC), and the CTI file is needed for every
    # Harvesters operation, from camera discovery through connecting and acquiring frames.
    ffmpeg_available = check_ffmpeg_availability()
    gpu_available = check_gpu_availability()
    cti_path = check_cti_file()

    ffmpeg_status = "OK" if ffmpeg_available else "Missing"
    gpu_status = "OK" if gpu_available else "None"

    # Distinguishes an absent runtime from a missing configuration, since check_cti_file() reports None for both and
    # telling the agent to configure a Producer it can never use would send it down a dead end.
    if not genicam_runtime_available():
        cti_status = "Unsupported"
    elif cti_path is not None:
        cti_status = "OK"
    else:
        cti_status = "None"

    return f"FFMPEG: {ffmpeg_status} | GPU: {gpu_status} | CTI: {cti_status}"
