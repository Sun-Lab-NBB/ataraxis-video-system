"""Provides a unified API that allows other library modules to interface with any supported camera hardware.

These interfaces abstract the necessary procedures to connect to the camera and continuously grab the
acquired frames.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
import ctypes
from typing import TYPE_CHECKING, Any
from pathlib import Path
import platform
from contextlib import contextmanager
from dataclasses import dataclass

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray
    from genicam.genapi import NodeMap

import cv2
import numpy as np
import platformdirs
from ataraxis_time import TimeUnits, PrecisionTimer, TimerPrecisions, rate_to_interval
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

try:
    from harvesters.core import Harvester, ImageAcquirer
    from harvesters.util.pfnc import bgr_formats, rgb_formats, mono_location_formats
except ImportError:  # pragma: no cover
    # The GenICam camera runtime is not installed on the macOS hosts that _GENICAM_RUNTIME_CLAIMED excludes, where the
    # 'genicam' distribution publishes no wheel. Guarding the import keeps this module usable for the OpenCV and Mock
    # interfaces there, and every entry point that reaches GenICam hardware calls _require_genicam_runtime() before
    # touching the names below. The format collections fall back to empty, which no reachable code consults, because
    # the frame grab loop that reads them is only reachable through a connection the guard refuses to open. Only one of
    # the two branches runs on any single host, so the fallback stays out of coverage measurement.
    Harvester = None
    ImageAcquirer = None
    bgr_formats = ()
    rgb_formats = ()
    mono_location_formats = ()

from .saver import InputPixelFormats
from .configuration import (
    DEFAULT_BLACKLISTED_NODES,
    GenicamNodeInfo,
    GenicamConfiguration,
    read_genicam_node,
    read_genicam_nodes,
    write_genicam_node,
    format_genicam_node,
    apply_genicam_configuration,
)

_GENICAM_RUNTIME_CLAIMED: bool = sys.platform != "darwin" or (
    sys.version_info < (3, 14) and platform.machine() == "arm64"
)
"""Tracks whether this library claims the GenICam camera runtime as a dependency on the host evaluating it.

This mirrors the environment marker the 'harvesters' and 'genicam' distributions carry in the project metadata, which is
the only way to separate a host that never installs the runtime from one whose installation is damaged. The marker and
this expression are edited together. The 'genicam' distribution publishes a macOS wheel only for Apple Silicon on Python
3.12 and 3.13, so Intel Macs and Python 3.14 install no runtime while every other platform installs one.
"""

GENICAM_UNAVAILABLE_REASON: str = (
    (
        "The 'harvesters' and 'genicam' distributions that supply the GenICam camera runtime install together with "
        "this library on this platform, so a runtime that does not import indicates a damaged installation. Reinstall "
        "the library to restore the GenICam camera interface."
    )
    if _GENICAM_RUNTIME_CLAIMED
    else (
        "This host does not support the GenICam camera interface, as the 'genicam' distribution that supplies its "
        "runtime publishes a macOS wheel only for Apple Silicon running Python 3.12 or 3.13. Use the 'opencv' camera "
        "interface, or drive the GenICam cameras from a Linux host, a Windows host, or an Apple Silicon Mac running "
        "Python 3.12 or 3.13."
    )
)
"""Explains why the GenICam camera runtime is unavailable, which every interface reports when the runtime is absent.

The explanation is resolved from the host rather than from the failed import, because the hosts that install no runtime
report an expected limitation while every other host installs it alongside the library, making an absent runtime a
broken environment there. Resolving it as a single conditional expression keeps both wordings out of a platform branch
that only one host is ever able to execute.
"""

_MONOCHROME_FORMATS: set[str] = set(mono_location_formats)
"""Stores the monochrome Harvesters color formats as a set, which keeps membership checks constant in the format
count."""

_COLOR_FORMATS: set[str] = set(bgr_formats) | set(rgb_formats)
"""Stores the BGR and RGB Harvesters color formats as a set, which keeps membership checks constant in the format
count."""

_ALL_RGB_FORMATS: set[str] = set(rgb_formats)
"""Stores the RGB Harvesters color formats as a set, which keeps membership checks constant in the format count."""

_CTI_PATH_VARIABLE: str = "AXVS_CTI_PATH"
"""Stores the name of the environment variable that overrides the persisted GenTL Producer interface (.cti) file path
for the duration of a single runtime.
"""

_FAIL_CRITICAL_ERRORS_MODE: int = 0x0001
"""Stores the Windows SEM_FAILCRITICALERRORS error mode, which hands a critical error to the thread that caused it
instead of announcing the error in a message box."""

_SET_THREAD_ERROR_MODE: Any | None = None
"""Stores the Windows API that overrides the error mode of the calling thread, or None on a platform whose dynamic
loader raises no message box.

The entry point is resolved once at import, which keeps the platform decision out of the guard that consumes it.
"""

if sys.platform == "win32":  # pragma: no cover - platform branch, only a Windows host executes it.
    _SET_THREAD_ERROR_MODE = ctypes.windll.kernel32.SetThreadErrorMode

_FRAME_POOL_SIZE: int = 10
"""Determines the size of the frame pool used by the MockCamera instances."""

_SLEEP_GRANULARITY: int = 16 if sys.platform == "win32" else 1
"""Stores the number of milliseconds by which a sleeping millisecond-precision delay overshoots the period it is asked
for on the host.

Windows resolves a sleep against a system timer tick of roughly 15.6 milliseconds, so a sleep returns at the first tick
that follows the requested period, while every other supported platform resolves it within a millisecond. A delay that
has to land inside a deadline therefore sleeps this many milliseconds short of it and spins the remainder, and it spins
a period shorter than the granularity outright.
"""

_MAXIMUM_NON_WORKING_IDS: int = 5
"""The consecutive failed probes that end OpenCV index discovery, after which the cameras found so far are
reported."""

_MAXIMUM_EVALUATED_IDS: int = 100
"""The maximum number of camera indices OpenCV index discovery evaluates."""


class CameraInterfaces(StrEnum):
    """Defines the supported camera interface backends compatible with the VideoSystem class."""

    HARVESTERS = "harvesters"
    """The preferred backend for all cameras that support the GenICam standard, which includes most scientific
    and industrial machine-vision cameras, based on the 'Harvesters' library and compatible with USB, Ethernet, and
    PCIE interfaces.
    """
    OPENCV = "opencv"
    """The backend for all cameras that do not support the GenICam standard, based on the 'OpenCV' library
    and primarily compatible with consumer-grade cameras that use the USB interface.
    """
    MOCK = "mock"
    """The mock backend that simulates frame acquisition without camera hardware, used for testing and dry runs."""


@dataclass(frozen=True, slots=True)
class CameraInformation:
    """Stores descriptive information about a camera discoverable through OpenCV or Harvesters libraries."""

    camera_index: int
    """The index of the camera in the list of all cameras discoverable through the evaluated interface
    (OpenCV or Harvesters)."""
    interface: CameraInterfaces | str
    """The interface that discovered the camera."""
    frame_width: int
    """The width of the frames acquired by the camera, in pixels."""
    frame_height: int
    """The height of the frames acquired by the camera, in pixels."""
    acquisition_frame_rate: int
    """The frame rate at which the camera acquires frames, in frames per second, or 0 for Harvesters cameras that do
    not implement the optional AcquisitionFrameRate feature."""
    serial_number: str | None = None
    """Only for Harvesters-discoverable cameras. Contains the camera's serial number."""
    model: str | None = None
    """Only for Harvesters-discoverable cameras. Contains the camera's model name."""


def discover_camera_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about all accessible cameras.

    OpenCV cameras are discovered first, followed by Harvesters cameras (if a CTI file has been configured).

    Notes:
        For OpenCV cameras, it is impossible to retrieve serial numbers or camera models.

        For Harvesters cameras, this function requires a valid CTI file to be configured via the add_cti_file()
        function, the 'axvs cti set' CLI command, or the ``AXVS_CTI_PATH`` environment variable, which takes precedence
        over the persisted path. If no CTI file is configured, Harvesters camera discovery is skipped. Harvesters
        discovery is also skipped where the GenICam runtime is absent, which is every Intel Mac and every macOS host
        running Python 3.14.

    Returns:
        A tuple of CameraInformation instances for all discovered cameras from both interfaces.
    """
    opencv_cameras = _get_opencv_ids()

    # Skips Harvesters discovery where the GenICam runtime is absent, since discovery reports the cameras this machine
    # is able to reach rather than asserting that every interface is available on every platform.
    if not genicam_runtime_available():
        return opencv_cameras

    # Attempts to discover Harvesters-compatible cameras. Skips if no CTI file is configured.
    try:
        harvesters_cameras = _get_harvesters_ids()
    except FileNotFoundError:
        # No CTI file configured, skips Harvesters discovery.
        harvesters_cameras = ()

    return opencv_cameras + harvesters_cameras


def genicam_runtime_available() -> bool:
    """Determines whether the GenICam camera runtime is available in this environment.

    The runtime is supplied by the 'harvesters' and 'genicam' distributions, which this library installs on every
    platform other than the Intel Macs and the macOS hosts running Python 3.14, where 'genicam' publishes no wheel.

    Returns:
        True when the runtime is importable, False otherwise.
    """
    return Harvester is not None


def add_cti_file(cti_path: Path) -> None:
    """Configures the 'harvesters' camera interface to use the provided .cti file during all future runtimes.

    The 'harvesters' camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    compatible GenTL devices (cameras).

    Notes:
        The path to the .cti file is stored inside the user's data directory, so that it can be reused between library
        calls.

    Args:
        cti_path: The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, is also acceptable.
            See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.

    Raises:
        NotImplementedError: If the GenICam camera runtime is not available in this environment.
        FileNotFoundError: If the supplied .cti file does not exist.
        OSError: If the supplied .cti file is not a loadable GenTL Producer.
    """
    _require_genicam_runtime(action="configure the GenTL Producer interface (.cti) file")

    # Resolves the path before it is verified and persisted. A relative path validates against the directory the
    # command happened to run from and then fails to resolve in every later runtime started from anywhere else,
    # which contradicts the reuse this function exists to provide.
    cti_path = Path(cti_path).expanduser().resolve()

    harvester = Harvester()
    with _suppress_loader_error_dialog():
        harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)

    application_directory = Path(platformdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
    cti_path_file = application_directory / "cti_path.txt"

    ensure_directory_exists(path=cti_path_file, is_file=True)

    with cti_path_file.open("w") as file:
        file.write(str(cti_path))


def check_cti_file() -> Path | None:
    """Checks whether the library is configured to use a GenTL Producer interface (.cti) file.

    The 'harvesters' camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    compatible GenTL devices (cameras). The ``AXVS_CTI_PATH`` environment variable takes precedence over the persisted
    path, matching the resolution order applied when connecting to a camera.

    Returns:
        The Path to the configured .cti file if one exists and is valid, or None otherwise. Also returns None where the
        GenICam runtime that consumes the Producer is absent, which is every Intel Mac and every macOS host running
        Python 3.14.
    """
    # Reports the unusable state rather than raising, since this function answers whether the interface is ready to use
    # and an unsupported platform is one of the ways it is not.
    if not genicam_runtime_available():
        return None

    override = os.environ.get(_CTI_PATH_VARIABLE)
    if override:
        cti_path = Path(override).expanduser().resolve()
    else:
        application_directory = Path(platformdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
        cti_path_file = application_directory / "cti_path.txt"

        if not cti_path_file.exists():
            return None

        with cti_path_file.open() as file:
            cti_path = Path(file.read().strip())

    try:
        harvester = Harvester()
        with _suppress_loader_error_dialog():
            harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)
    except Exception:
        # The configured CTI file is no longer valid.
        return None
    else:
        return cti_path


class OpenCVCamera:
    """Interfaces with the specified OpenCV-compatible camera hardware to acquire frame data.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        color: Determines whether the camera acquires colored or monochrome images. This determines how to store the
            acquired frames. Colored frames are saved using the 'BGR' channel order, monochrome images are reduced to
            a single-channel format. If this argument is not explicitly provided, the instance acquires colored frames.
        camera_index: The index of the camera in the list of all cameras discoverable by OpenCV, e.g.: 0 for the first
            available camera, 1 for the second, etc. This specifies the camera hardware the instance should interface
            with at runtime.
        frame_rate: The desired rate, in frames per second, at which to capture the data. Note that whether the
            requested rate is attainable depends on the hardware capabilities of the camera and the communication
            interface. If this argument is not explicitly provided, the instance uses the default frame rate of the
            connected camera.
        frame_width: The desired width of the acquired frames, in pixels. Note that the requested width must be
            compatible with the range of frame dimensions supported by the camera hardware. If this argument is not
            explicitly provided, the instance uses the default frame width of the connected camera.
        frame_height: Same as 'frame_width', but specifies the desired height of the acquired frames, in pixels. If this
            argument is not explicitly provided, the instance uses the default frame height of the connected camera.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _color: Determines whether the camera acquires colored or monochrome images.
        _camera_index: Stores the index of the camera hardware in the list of all OpenCV-discoverable cameras connected
            to the host-machine.
        _frame_rate: Stores the camera's frame acquisition rate.
        _frame_width: Stores the width of the camera's frames.
        _frame_height: Stores the height of the camera's frames.
        _camera: Stores the OpenCV VideoCapture object that interfaces with the camera.
        _acquiring: Tracks whether the camera is currently acquiring frames.
    """

    def __init__(
        self,
        system_id: int,
        camera_index: int = 0,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        *,
        color: bool = True,
    ) -> None:
        self._system_id: int = system_id
        self._color: bool = color
        self._camera_index: int = camera_index
        self._frame_rate: int = 0 if frame_rate is None else frame_rate
        self._frame_width: int = 0 if frame_width is None else frame_width
        self._frame_height: int = 0 if frame_height is None else frame_height
        self._camera: cv2.VideoCapture | None = None
        self._acquiring: bool = False

    def __del__(self) -> None:
        """Releases the underlying VideoCapture object when the instance is garbage-collected."""
        self.disconnect()

    def __repr__(self) -> str:
        """Returns the string representation of the OpenCVCamera instance."""
        return (
            f"OpenCVCamera(system_id={self._system_id}, camera_index={self._camera_index}, "
            f"frame_rate={self.frame_rate} frames / second, frame_width={self.frame_width} pixels, "
            f"frame_height={self.frame_height} pixels, connected={self._camera is not None}, "
            f"acquiring={self._acquiring})"
        )

    def connect(self) -> None:
        """Connects to the managed camera hardware.

        Raises:
            ValueError: If the instance is configured to override hardware-defined acquisition parameters and the
                camera rejects the user-defined frame height, width, or acquisition rate parameters.
        """
        if self._camera is not None:  # pragma: no cover - defensive guard, no caller reconnects a connected interface.
            return

        self._camera = cv2.VideoCapture(index=self._camera_index, apiPreference=cv2.CAP_ANY)

        # Overrides the requested camera acquisition parameters if necessary. If the camera does not accept the
        # requested parameters, terminates with an error message. Otherwise, queries the acquisition parameters from
        # the connected camera.
        if self._frame_rate != 0:
            self._camera.set(propId=cv2.CAP_PROP_FPS, value=float(self._frame_rate))

            # Rounds rather than truncates, so a camera that honors the request and reports it as 29.97 is not
            # rejected by the cast. Any genuine deviation is rejected below, since the library performs no software
            # decimation and would otherwise stamp the requested rate onto a stream acquired at a different one.
            actual_frame_rate = round(self._camera.get(propId=cv2.CAP_PROP_FPS))
            if actual_frame_rate != self._frame_rate:
                message = (
                    f"Unable to configure the OpenCVCamera interface for the VideoSystem with id {self._system_id}. "
                    f"Attempted configuring the camera to acquire frames at the rate of {self._frame_rate} "
                    f"frames per second, but the camera automatically adjusted the acquisition rate to "
                    f"{actual_frame_rate}. This indicates that the camera does not support the requested frame "
                    f"acquisition rate."
                )
                console.error(error=ValueError, message=message)
        else:
            self._frame_rate = int(self._camera.get(propId=cv2.CAP_PROP_FPS))

        if self._frame_width != 0:
            self._camera.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=float(self._frame_width))
            actual_frame_width = int(self._camera.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
            if actual_frame_width != self._frame_width:
                message = (
                    f"Unable to configure the OpenCVCamera interface for the VideoSystem with id {self._system_id}. "
                    f"Attempted configuring the camera to acquire frames with the width of {self._frame_width} pixels, "
                    f"but the camera automatically adjusted the frame width to {actual_frame_width}. This indicates "
                    f"that the camera does not support the requested frame height and width combination."
                )
                console.error(error=ValueError, message=message)
        else:
            self._frame_width = int(self._camera.get(propId=cv2.CAP_PROP_FRAME_WIDTH))

        if self._frame_height != 0:
            self._camera.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=float(self._frame_height))
            actual_frame_height = int(self._camera.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_frame_height != self._frame_height:
                message = (
                    f"Unable to configure the OpenCVCamera interface for the VideoSystem with id {self._system_id}. "
                    f"Attempted configuring the camera to acquire frames with the height of {self._frame_height} "
                    f"pixels, but the camera automatically adjusted the frame height to {actual_frame_height}. This "
                    f"indicates that the camera does not support the requested frame height and width combination."
                )
                console.error(error=ValueError, message=message)
        else:
            self._frame_height = int(self._camera.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))

    def disconnect(self) -> None:
        """Disconnects from the managed camera hardware."""
        if self._camera is None:
            return

        self._camera.release()
        self._acquiring = False
        self._camera = None

    @property
    def frame_rate(self) -> int:
        """Returns the acquisition rate of the camera, in frames per second (fps)."""
        return self._frame_rate

    @property
    def frame_width(self) -> int:
        """Returns the width of the acquired frames, in pixels."""
        return self._frame_width

    @property
    def frame_height(self) -> int:
        """Returns the height of the acquired frames, in pixels."""
        return self._frame_height

    @property
    def pixel_color_format(self) -> InputPixelFormats:
        """Returns the pixel color format of the acquired frames."""
        if self._color:
            return InputPixelFormats.BGR
        return InputPixelFormats.MONOCHROME

    def grab_frame(self) -> NDArray[np.floating[Any] | np.integer[Any]]:
        """Grabs the first available frame from the managed camera's acquisition buffer.

        This method has to be called repeatedly (cyclically) to fetch the newly acquired frames from the camera.

        Notes:
            The first time this method is called, the camera initializes frame acquisition, which is carried out
            asynchronously. If the camera supports buffering, it continuously saves the frames into its circular buffer.
            If the camera does not support buffering, the frame data must be fetched before the camera acquires the next
            frame to prevent frame loss.

            Due to the initial setup of the buffering procedure, the first call to this method incurs a significant
            delay.

        Returns:
            A NumPy array that stores the frame data. Depending on whether the camera acquires colored or monochrome
            images, the returned arrays have the shape (height, width, channels) or (height, width). Color data uses
            the BGR channel order.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            BrokenPipeError: If the instance fails to fetch a frame from the connected camera hardware.
        """
        if self._camera is None:
            message = (
                f"Unable to acquire a frame from the OpenCVCamera interface for the VideoSystem with id "
                f"{self._system_id}. The interface must be connected to the camera hardware, but it is currently "
                f"disconnected. Call the connect() method prior to calling the grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)

        # Flips the acquisition tracker to True the first time this method is called for a connected camera.
        if not self._acquiring:
            self._acquiring = True

        frame: NDArray[np.floating[Any] | np.integer[Any]]
        success, frame = self._camera.read()
        if not success:
            message = (
                f"Unable to acquire a frame from the OpenCVCamera interface for the VideoSystem with id "
                f"{self._system_id}. The camera hardware must return a frame image for each acquisition request, but "
                f"it returned none. This indicates an initialization or a connectivity issue."
            )
            console.error(message=message, error=BrokenPipeError)

        if not self._color:
            # Converts the frame data from using BGR color space (default for all frames) to Monochrome if needed.
            frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        return frame


class HarvestersCamera:
    """Interfaces with the specified GenICam-compatible camera hardware to acquire frame data.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        camera_index: The index of the camera in the list of all cameras discoverable by Harvesters, e.g.: 0 for the
            first available camera, 1 for the second, etc. This specifies the camera hardware the instance should
            interface with at runtime.
        frame_rate: The desired rate, in frames per second, at which to capture the data. Note that whether the
            requested rate is attainable depends on the hardware capabilities of the camera and the communication
            interface. If this argument is not explicitly provided, the instance adopts the camera's
            AcquisitionFrameRate value, or reports 0 for cameras that do not implement that optional feature.
        frame_width: The desired width of the acquired frames, in pixels. Note that the requested width must be
            compatible with the range of frame dimensions supported by the camera hardware. If this argument is not
            explicitly provided, the instance uses the default frame width of the connected camera.
        frame_height: Same as 'frame_width', but specifies the desired height of the acquired frames, in pixels. If this
            argument is not explicitly provided, the instance uses the default frame height of the connected camera.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _camera_index: Stores the index of the camera hardware in the list of all Harvesters-discoverable cameras
            connected to the host-machine.
        _frame_rate: Stores the camera's frame acquisition rate.
        _frame_width: Stores the width of the camera's frames.
        _frame_height: Stores the height of the camera's frames.
        _harvester: Stores the Harvester interface object that discovers and manages the list of accessible GenTL
            cameras.
        _camera: Stores the Harvesters ImageAcquirer object that interfaces with the camera.
        _color: Tracks whether the frames are acquired using a monochrome or a colored data format.
        _model: Stores the model name of the connected camera. Populated during connect(), reset during disconnect().
        _serial_number: Stores the serial number of the connected camera. Populated during connect(), reset during
            disconnect().
    """

    def __init__(
        self,
        system_id: int,
        camera_index: int = 0,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via the VideoSystem constructor, which
        # performs the necessary input filtering.

        self._system_id: int = system_id
        self._camera_index: int = camera_index
        self._frame_rate: int = 0 if frame_rate is None else frame_rate
        self._frame_width: int = 0 if frame_width is None else frame_width
        self._frame_height: int = 0 if frame_height is None else frame_height

        # Pre-creates the attribute to store the initialized Harvester class to discover the list of available cameras.
        # While the object was pickleable in earlier Harvesters versions, it is now not pickleable and must be handled
        # similar to how ImageAcquirer objects are handled.
        self._harvester: Harvester | None = None

        self._camera: ImageAcquirer | None = None

        # Tracks whether the acquired frames use a monochrome or a colored data format.
        self._color: bool = False

        # Stores the camera model and serial number. Populated during connect(), reset during disconnect().
        self._model: str = ""
        self._serial_number: str = ""

    def __del__(self) -> None:
        """Releases the underlying ImageAcquirer object when the instance is garbage-collected."""
        self.disconnect()

    def __repr__(self) -> str:
        """Returns the string representation of the HarvestersCamera instance."""
        return (
            f"HarvestersCamera(system_id={self._system_id}, camera_index={self._camera_index}, "
            f"frame_rate={self.frame_rate} frames / second, frame_width={self.frame_width} pixels, "
            f"frame_height={self.frame_height} pixels, connected={self._camera is not None}, "
            f"acquiring={self.is_acquiring})"
        )

    def connect(self) -> None:
        """Connects to the managed camera hardware.

        Raises:
            NotImplementedError: If the GenICam camera runtime is not available in this environment.
            FileNotFoundError: If no .cti file path has been configured or the configured file does not exist.
            OSError: If the configured .cti file is not a loadable GenTL Producer.
            IndexError: If the camera index exceeds the number of cameras the configured GenTL Producer discovers.
        """
        if self._camera is not None:
            return

        _require_genicam_runtime(action=f"connect to the GenICam camera at index {self._camera_index}")

        self._harvester = Harvester()
        with _suppress_loader_error_dialog():
            self._harvester.add_file(file_path=str(_get_cti_path()), check_existence=True, check_validity=True)
        # Suppresses stdout and stderr to avoid verbose CTI printouts.
        with _suppress_output():
            self._harvester.update()

        self._camera = self._harvester.create(search_key=self._camera_index)

        device_info = self._harvester.device_info_list[self._camera_index]
        self._model = device_info.model
        self._serial_number = device_info.serial_number

        node_map = self._camera.remote_device.node_map

        # Overrides the requested camera acquisition parameters if necessary. Note, there is no guarantee that the
        # camera accepts the requested parameters.
        if self._frame_width != 0:
            node_map.Width.value = self._frame_width
        if self._frame_height != 0:
            node_map.Height.value = self._frame_height

        # Sets the frame rate last, as it is affected by frame width and height. Devices that omit the optional
        # AcquisitionFrameRate feature retain the requested rate, as there is no hardware value to negotiate against.
        frame_rate_node = _get_frame_rate_node(node_map=node_map)
        if frame_rate_node is not None:
            if self._frame_rate != 0:
                frame_rate_node.value = self._frame_rate

            # Rounds the way discovery reads the same node, so that the camera interface and discover_camera_ids()
            # report one rate for one device rather than differing by a frame whenever the node clamps.
            self._frame_rate = int(round(number=frame_rate_node.value, ndigits=0))

        self._frame_width = int(node_map.Width.value)
        self._frame_height = int(node_map.Height.value)

    def disconnect(self) -> None:
        """Disconnects from the managed camera hardware."""
        if self._camera is None or self._harvester is None:
            return

        if self._camera.is_acquiring():
            self._camera.stop()

        self._camera.destroy()
        self._camera = None
        self._harvester.reset()
        self._harvester = None
        self._model = ""
        self._serial_number = ""

    @property
    def is_connected(self) -> bool:
        """Returns True if the instance is connected to the camera hardware."""
        return self._camera is not None

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames."""
        if self._camera is not None:
            return bool(self._camera.is_acquiring())
        return False

    @property
    def frame_rate(self) -> int:
        """Returns the acquisition rate of the camera, in frames per second (fps), which is the requested rate for
        cameras that do not implement the optional AcquisitionFrameRate feature.
        """
        return self._frame_rate

    @property
    def frame_width(self) -> int:
        """Returns the width of the acquired frames, in pixels."""
        return self._frame_width

    @property
    def frame_height(self) -> int:
        """Returns the height of the acquired frames, in pixels."""
        return self._frame_height

    @property
    def pixel_color_format(self) -> InputPixelFormats:
        """Returns the pixel color format of the acquired frames."""
        if self._color:
            return InputPixelFormats.BGR
        return InputPixelFormats.MONOCHROME

    @property
    def model(self) -> str:
        """Returns the model name of the connected camera, or an empty string if not connected."""
        return self._model

    @property
    def serial_number(self) -> str:
        """Returns the serial number of the connected camera, or an empty string if not connected."""
        return self._serial_number

    @property
    def node_map(self) -> NodeMap:
        """Returns the GenICam node map of the connected camera, or raises ``ConnectionError`` if not connected."""
        if self._camera is None:
            message = (
                f"Unable to access the node map for VideoSystem with id {self._system_id}. The camera is not "
                f"connected. Call the connect() method first."
            )
            console.error(message=message, error=ConnectionError)

        return self._camera.remote_device.node_map

    def get_node_info(self, name: str) -> GenicamNodeInfo:
        """Reads a single readable value node from the connected camera and returns its name and current value.

        Args:
            name: The feature name of the node to read (e.g., "Width", "ExposureTime").

        Returns:
            A ``GenicamNodeInfo`` instance containing the node's name and current value.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            ValueError: If the node is not a readable value node.
            AttributeError: If the named node does not exist on the camera's node map.
        """
        return read_genicam_node(node_map=self.node_map, name=name)

    def get_node_description(self, name: str) -> str:
        """Reads a single readable value node from the connected camera and returns a formatted description string.

        Args:
            name: The feature name of the node to read (e.g., "Width", "ExposureTime").

        Returns:
            A multi-line formatted string containing the node's full metadata.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            ValueError: If the node is not a readable value node.
            AttributeError: If the named node does not exist on the camera's node map.
        """
        return format_genicam_node(node_map=self.node_map, name=name)

    def set_node_value(self, name: str, value: str) -> None:
        """Sets the value of a single writable (ReadWrite) GenICam feature node on the connected camera.

        Args:
            name: The feature name of a writable node (e.g., "Width", "ExposureTime").
            value: The string representation of the value to write. Coerced to the node's native type automatically.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            AttributeError: If the named node does not exist on the camera's node map.
            ValueError: If the named node does not have ReadWrite access or the value cannot be coerced.
            RuntimeError: If the write operation fails.
        """
        write_genicam_node(node_map=self.node_map, name=name, value=value)

    def get_configuration(
        self,
        blacklisted_nodes: frozenset[str] = DEFAULT_BLACKLISTED_NODES,
    ) -> GenicamConfiguration:
        """Enumerates all ReadWrite GenICam nodes on the connected camera and returns the configuration.

        Args:
            blacklisted_nodes: A set of node names to exclude from the configuration. Defaults to
                ``DEFAULT_BLACKLISTED_NODES``, which excludes vendor-specific nodes known to report ReadWrite access
                but reject writes at the hardware level. Pass an empty frozenset to disable blacklisting.

        Returns:
            A ``GenicamConfiguration`` instance containing the camera identity and all ReadWrite node values.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
        """
        return GenicamConfiguration(
            camera_model=self._model,
            camera_serial_number=self._serial_number,
            nodes=read_genicam_nodes(node_map=self.node_map, blacklisted_nodes=blacklisted_nodes),
        )

    def apply_configuration(
        self,
        config: GenicamConfiguration,
        *,
        strict_identity: bool = False,
        blacklisted_nodes: frozenset[str] = DEFAULT_BLACKLISTED_NODES,
    ) -> None:
        """Applies a ``GenicamConfiguration`` to the connected camera.

        Args:
            config: The configuration instance containing ReadWrite nodes to apply.
            strict_identity: Determines whether to abort on camera identity mismatch instead of warning.
            blacklisted_nodes: A set of node names to silently skip during validation and write operations. Defaults
                to ``DEFAULT_BLACKLISTED_NODES``, which excludes vendor-specific nodes known to report ReadWrite
                access but reject writes at the hardware level. Pass an empty frozenset to disable blacklisting.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            ValueError: If the camera identity mismatches (strict mode) or any node is missing or not writable.
            RuntimeError: If any non-blacklisted node write operation fails.
        """
        apply_genicam_configuration(
            node_map=self.node_map,
            config=config,
            current_model=self._model,
            current_serial=self._serial_number,
            strict=strict_identity,
            blacklisted_nodes=blacklisted_nodes,
        )

    def grab_frame(self) -> NDArray[np.integer[Any]]:
        """Grabs the first available frame from the managed camera's acquisition buffer.

        This method has to be called repeatedly (cyclically) to fetch the newly acquired frames from the camera.

        Notes:
            The first time this method is called, the camera initializes frame acquisition, which is carried out
            asynchronously. The acquired frames are temporarily stored in the camera's circular buffer until they are
            fetched by this method.

            Due to the initial setup of the buffering procedure, the first call to this method incurs a significant
            delay.

        Returns:
            A NumPy array that stores the frame data. Depending on whether the camera acquires colored or monochrome
            images, the returned arrays have the shape (height, width, channels) or (height, width). Color data uses
            the BGR channel order.

        Raises:
            ConnectionError: If the instance is not connected to the camera hardware.
            BrokenPipeError: If the instance fails to fetch a frame from the connected camera hardware.
            ValueError: If the acquired frame data uses an unsupported data (color) format.
        """
        if self._camera is None:
            message = (
                f"Unable to acquire a frame from the HarvestersCamera interface for the VideoSystem with id "
                f"{self._system_id}. The interface must be connected to the camera hardware, but it is currently "
                f"disconnected. Call the connect() method prior to calling the grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)

        # Triggers camera frame acquisition the first time this method is called.
        if not self._camera.is_acquiring():
            self._camera.start()

        # Retrieves the next available image buffer from the camera. The result is bound before its context is
        # entered, since a failed fetch answers with None and entering a None context raises an opaque TypeError in
        # place of the diagnostic error below.
        buffer = self._camera.fetch()

        if buffer is None:
            message = (
                f"Unable to acquire a frame from the HarvestersCamera interface for the VideoSystem with id "
                f"{self._system_id}. The camera hardware must return a frame image for each acquisition request, but "
                f"it returned none. This indicates an initialization or a connectivity issue."
            )
            console.error(message=message, error=BrokenPipeError)

        # Uses the 'with' context to properly re-queue the buffer to acquire further images.
        with buffer:
            content = buffer.payload.components[0]

            # Collects the information necessary to reshape the originally 1-dimensional frame array into the
            # 2-dimensional array using the correct number and order of color channels.
            width = content.width
            height = content.height
            data_format = content.data_format

            if data_format in _MONOCHROME_FORMATS:
                # Uses copy, which is VERY important. Once the buffer is released, the original 'content' is lost,
                # so NumPy needs to copy the data instead of using the default referencing behavior.
                out_array: NDArray[np.integer[Any]] = content.data.reshape(height, width).copy()
                self._color = False
                return out_array

            if data_format in _COLOR_FORMATS:
                reshaped_data: NDArray[np.integer[Any]] = content.data.reshape(
                    height,
                    width,
                    int(content.num_components_per_pixel),
                )

                # Swaps every R and B value (RGB -> BGR) to produce BGR images. This ensures consistency with the
                # OpenCVCamera API. Only applies to RGB formats, since BGR formats are used as-is.
                if data_format in _ALL_RGB_FORMATS:
                    frame: NDArray[np.integer[Any]] = reshaped_data[:, :, ::-1].copy()
                else:
                    frame = reshaped_data.copy()

                self._color = True

                return frame

            message = (
                f"Unable to process a frame acquired by the HarvestersCamera interface for the VideoSystem with id "
                f"{self._system_id}. The frame data (color) format must belong to the unpacked Monochrome, RGB, or "
                f"BGR families, but got {data_format}."
            )
            console.error(message=message, error=ValueError)
            # Satisfies ruff RET503. console.error() is NoReturn, so this line never executes.
            return np.zeros((0, 0), dtype=np.uint8)  # pragma: no cover


@contextmanager
def harvester_connection(camera_index: int) -> Generator[HarvestersCamera, None, None]:
    """Opens a temporary connection to the target GenICam camera for the duration of the managed block.

    Notes:
        The camera is created with a placeholder system identifier, since a connection opened this way exposes the
        camera's GenICam node map rather than acquiring frames. The connection is always closed on exit, which
        releases the GenTL handle for other processes.

    Args:
        camera_index: The index of the camera in the list of all cameras discoverable by Harvesters.

    Yields:
        The connected camera interface.
    """
    camera = HarvestersCamera(system_id=0, camera_index=camera_index)
    try:
        camera.connect()
        yield camera
    finally:
        camera.disconnect()


class MockCamera:
    """Simulates (mocks) the behavior of the OpenCVCamera and HarvestersCamera classes without the need to interface
    with a physical camera.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        frame_rate: The simulated frame acquisition rate of the camera, in frames per second. If this argument is not
            explicitly provided, the instance simulates 30 frames per second.
        frame_width: The simulated camera frame width, in pixels. If this argument is not explicitly provided, the
            instance simulates a width of 600 pixels.
        frame_height: The simulated camera frame height, in pixels. If this argument is not explicitly provided, the
            instance simulates a height of 400 pixels.
        color: Determines whether to generate frames in the BGR color mode instead of the grayscale (monochrome) mode.
            If this argument is not explicitly provided, the instance generates BGR frames.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _color: Determines whether to simulate monochrome or BGR (colored) frame images.
        _camera: Tracks whether the camera is 'connected'.
        _frame_rate: Stores the camera's frame acquisition rate.
        _frame_width: Stores the width of the camera's frames.
        _frame_height: Stores the height of the camera's frames.
        _acquiring: Tracks whether the camera is currently acquiring video frames.
        _frames: Stores the pool of pre-generated frame images used to simulate camera frame acquisition.
        _current_frame_index: The index of the currently evaluated frame in the pre-generated frame pool buffer. This
            is used to simulate the behavior of the cyclic buffer used by physical cameras.
        _timer: After the camera is 'connected', this attribute is used to store the timer class that controls the
            simulated camera's frame rate.
        _time_between_frames: Stores the number of milliseconds that has to pass between two consecutive frame
            acquisitions, used to simulate a physical camera's frame rate.
    """

    def __init__(
        self,
        system_id: int,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        *,
        color: bool = True,
    ) -> None:
        self._system_id: int = system_id
        self._color: bool = color
        self._frame_rate: int = 30 if frame_rate is None else frame_rate
        self._frame_width: int = 600 if frame_width is None else frame_width
        self._frame_height: int = 400 if frame_height is None else frame_height
        self._camera: bool = False
        self._acquiring: bool = False

        # Creates a random number generator for reproducible frame generation.
        random_generator = np.random.default_rng(seed=42)

        # Statically generates the frame pool used for reproducible testing during grab_frame() method calls. The pool
        # is built from _FRAME_POOL_SIZE, because grab_frame() wraps its index against the same constant. Grayscale
        # frames carry a single channel, so only the colored branch reorders channels.
        channels = 3 if self._color else 1
        raw_frames = [
            random_generator.integers(
                low=0,
                high=256,
                size=(self._frame_height, self._frame_width, channels),
                dtype=np.uint8,
            )
            for _ in range(_FRAME_POOL_SIZE)
        ]

        # Casts to a tuple to optimize runtime efficiency.
        self._frames: tuple[NDArray[np.uint8], ...] = tuple(
            cv2.cvtColor(src=raw_frame, code=cv2.COLOR_RGB2BGR)  # type: ignore[misc]  # cvtColor returns MatLike.
            if self._color
            else raw_frame
            for raw_frame in raw_frames
        )
        self._current_frame_index: int = 0

        # Cannot be initialized here due to the use of multiprocessing in the VideoSystem class.
        self._timer: PrecisionTimer | None = None

        # Uses the frame_rate to derive the number of milliseconds that has to pass between two consecutive frame
        # acquisitions, used to simulate a physical camera's frame rate during grab_frame() runtime.
        self._time_between_frames: float = rate_to_interval(
            rate=self._frame_rate, to_units=TimeUnits.MILLISECOND, as_float=True
        )

    def __repr__(self) -> str:
        """Returns the string representation of the MockCamera instance."""
        return (
            f"MockCamera(system_id={self._system_id}, frame_rate={self._frame_rate} frames / second, "
            f"frame_width={self._frame_width} pixels, frame_height={self._frame_height} pixels, "
            f"connected={self._camera}, acquiring={self._acquiring})"
        )

    def connect(self) -> None:
        """Simulates connecting to the camera hardware."""
        self._camera = True

        # Uses millisecond precision, which supports simulating up to 1000 fps. Initializes here to make the class
        # compatible with the VideoSystem class that uses multiprocessing.
        self._timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    def disconnect(self) -> None:
        """Simulates disconnecting from the camera hardware."""
        self._camera = False
        self._acquiring = False
        self._timer = None

    @property
    def frame_rate(self) -> int:
        """Returns the acquisition rate of the camera, in frames per second (fps)."""
        return self._frame_rate

    @property
    def frame_width(self) -> int:
        """Returns the width of the acquired frames, in pixels."""
        return self._frame_width

    @property
    def frame_height(self) -> int:
        """Returns the height of the acquired frames, in pixels."""
        return self._frame_height

    @property
    def pixel_color_format(self) -> InputPixelFormats:
        """Returns the pixel color format of the acquired frames."""
        if self._color:
            return InputPixelFormats.BGR
        return InputPixelFormats.MONOCHROME

    def grab_frame(self) -> NDArray[np.uint8]:
        """Grabs the first available frame from the managed camera's acquisition buffer.

        This method has to be called repeatedly (cyclically) to fetch the newly acquired frames from the camera.

        Returns:
            A NumPy array that stores the frame data. Colored frames have the shape (height, width, 3) using the BGR
            channel order, while monochrome frames retain a singleton channel dimension with the shape
            (height, width, 1).

        Raises:
            ConnectionError: If the method is called for a class not currently 'connected' to a camera.
        """
        # Prevents calling this method before connecting to the camera's hardware. connect() creates the timer
        # alongside the connection, so the two are checked together and the checked state narrows the timer for mypy.
        if not self._camera or self._timer is None:
            message = (
                f"Unable to simulate a frame acquisition using the MockCamera interface for the VideoSystem with id "
                f"{self._system_id}. The interface must be simulating a connection to the camera hardware, but the "
                f"connection simulation has not been started. Call the connect() method prior to calling the "
                f"grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)

        # Flips the acquiring flag the first time this method is called.
        if not self._acquiring:
            self._acquiring = True

        # Simulates the blocking behavior of physical camera interfaces by using the timer class to enforce a certain
        # frame rate. Sleeping all but the host's sleep granularity releases the logical core the way a physical
        # interface blocking on its driver does. The spin that follows absorbs the sleep's overshoot, so the frame
        # period lands on the same millisecond the timer reaches when the whole interval is spun. A wait shorter than
        # the granularity is spun instead, because a sleep that short returns past the frame deadline and costs the
        # simulation the frame rate it is asked for.
        sleep_time = int(self._time_between_frames - self._timer.elapsed) - _SLEEP_GRANULARITY
        if sleep_time > 0:
            self._timer.delay(delay=sleep_time, allow_sleep=sleep_time >= _SLEEP_GRANULARITY)
        while self._timer.elapsed < self._time_between_frames:
            pass

        frame = self._frames[self._current_frame_index].copy()

        # Resets the timer to measure the time elapsed since the last frame acquisition.
        self._timer.reset()

        # Wrapping the index against the pool size simulates the behavior of a cyclic buffer.
        self._current_frame_index = (self._current_frame_index + 1) % _FRAME_POOL_SIZE

        return frame


def _get_opencv_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about the cameras accessible through
    the OpenCV library.

    Notes:
        Currently, it is impossible to retrieve serial numbers or camera models from OpenCV. Therefore, while this
        function tries to provide some ID information, it is typically insufficient to identify specific cameras. It is
        advised to test each discovered camera with the 'axvs run' CLI command to identify the mapping between the
        discovered indices (IDs) and physical cameras.

    Returns:
         A tuple of CameraInformation instances, one for each discovered OpenCV-compatible camera.
    """
    # Disables OpenCV error logging to avoid flushing the terminal with failed connection attempts.
    previous_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

    try:
        non_working_count = 0
        working_ids: list[CameraInformation] = []

        # Iterates over IDs until it discovers 5 non-working IDs. Evaluates 100 IDs at maximum to prevent infinite
        # execution. Suppresses stdout and stderr to silence V4L2 ioctl warnings emitted by the kernel driver
        # during device probing (e.g. 'ioctl(VIDIOC_QBUF): Bad file descriptor').
        for evaluated_id in range(_MAXIMUM_EVALUATED_IDS):
            try:
                with _suppress_output():
                    camera = cv2.VideoCapture(index=evaluated_id)
                try:
                    with _suppress_output():
                        is_opened = camera.isOpened() and camera.read()[0]
                    if is_opened:
                        frame_width = int(camera.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(camera.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
                        acquisition_rate = int(camera.get(propId=cv2.CAP_PROP_FPS))
                        camera_data = CameraInformation(
                            camera_index=evaluated_id,
                            interface=CameraInterfaces.OPENCV,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            acquisition_frame_rate=acquisition_rate,
                        )
                        working_ids.append(camera_data)
                        non_working_count = 0
                    else:
                        non_working_count += 1
                finally:
                    with _suppress_output():
                        camera.release()

            except Exception as error:
                # Marks any ID that raises a runtime error as non-working and notifies the user.
                console.echo(
                    message=f"OpenCV camera discovery: Failed to evaluate camera index {evaluated_id}. Error: {error}",
                    level=LogLevel.WARNING,
                )
                non_working_count += 1

            # Breaks the loop early once 5 or more non-working IDs are found consecutively.
            if non_working_count >= _MAXIMUM_NON_WORKING_IDS:
                break

        # Deduplicates cameras that map to the same physical device. On Linux, V4L2 often creates consecutive
        # device nodes per camera (e.g. /dev/video0 and /dev/video1 for one USB camera). To detect duplicates
        # cross-platform, checks consecutive pairs with identical properties: holds one camera open and tests whether
        # the next can simultaneously read frames. A physical camera can typically only stream to one VideoCapture
        # instance at a time, so a failed read indicates a duplicate node.
        unique_cameras: list[CameraInformation] = []
        skip_next = False

        for index, candidate in enumerate(working_ids):
            if skip_next:
                skip_next = False
                continue

            unique_cameras.append(candidate)

            # Checks the next candidate only if it has identical properties (consecutive duplicate pattern).
            if index + 1 < len(working_ids):
                next_candidate = working_ids[index + 1]
                if (
                    candidate.frame_width == next_candidate.frame_width
                    and candidate.frame_height == next_candidate.frame_height
                    and candidate.acquisition_frame_rate == next_candidate.acquisition_frame_rate
                ):
                    with _suppress_output():
                        holder = cv2.VideoCapture(index=candidate.camera_index)
                    try:
                        with _suppress_output():
                            _ = holder.read()
                            challenger = cv2.VideoCapture(index=next_candidate.camera_index)
                        try:
                            with _suppress_output():
                                can_read = challenger.isOpened() and challenger.read()[0]
                            if not can_read:
                                skip_next = True
                        finally:
                            with _suppress_output():
                                challenger.release()
                    finally:
                        with _suppress_output():
                            holder.release()

        return tuple(unique_cameras)

    finally:
        cv2.utils.logging.setLogLevel(previous_log_level)


def _get_harvesters_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about the cameras accessible
    through the Harvesters library.

    Notes:
        This function bundles the discovered ID (index) information with the serial number and the model for each
        camera to support identifying the cameras.

    Returns:
        A tuple of CameraInformation instances, one for each discovered Harvesters-compatible camera.
    """
    cti_path = _get_cti_path()

    # Instantiates the class and adds the input .cti file. Both checks are requested, since _get_cti_path() returns
    # the configured path unvalidated and a Producer that cannot be loaded would otherwise leave discovery reporting
    # an empty camera list that is indistinguishable from a healthy Producer with no cameras attached.
    harvester = Harvester()
    with _suppress_loader_error_dialog():
        harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)

    # Suppresses stdout and stderr to avoid verbose printouts about missing CTI features.
    with _suppress_output():
        harvester.update()

    working_ids: list[CameraInformation] = []
    for index, camera_info in enumerate(harvester.device_info_list):
        try:
            camera = harvester.create(search_key=index)
            try:
                node_map = camera.remote_device.node_map

                # Retrieves frame dimensions and acquisition rate from the camera's node map. Devices that omit the
                # optional AcquisitionFrameRate feature report a rate of 0.
                frame_width = int(node_map.Width.value)
                frame_height = int(node_map.Height.value)
                frame_rate_node = _get_frame_rate_node(node_map=node_map)
                acquisition_rate = 0 if frame_rate_node is None else int(round(number=frame_rate_node.value, ndigits=0))

                camera_data = CameraInformation(
                    camera_index=index,
                    interface=CameraInterfaces.HARVESTERS,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    acquisition_frame_rate=acquisition_rate,
                    serial_number=camera_info.serial_number,
                    model=camera_info.model,
                )
                working_ids.append(camera_data)
            finally:
                camera.destroy()

        except Exception as error:
            # Skips any device that cannot be connected or queried for any reason and notifies the user.
            console.echo(
                message=f"Harvesters camera discovery: Failed to query device at index {index}. Error: {error}",
                level=LogLevel.WARNING,
            )
            continue

    harvester.remove_file(file_path=str(cti_path))
    harvester.reset()

    return tuple(working_ids)


def _require_genicam_runtime(action: str) -> None:
    """Aborts the requested action when the GenICam camera runtime is absent from this environment.

    Args:
        action: The action the caller is unable to carry out, phrased as an infinitive clause without its subject.

    Raises:
        NotImplementedError: If the runtime is not importable.
    """
    if genicam_runtime_available():
        return

    message = f"Unable to {action}. {GENICAM_UNAVAILABLE_REASON}"
    console.error(message=message, error=NotImplementedError)


def _get_frame_rate_node(node_map: NodeMap) -> Any | None:
    """Resolves the node that reports the camera's frame acquisition rate.

    Notes:
        AcquisitionFrameRate is an optional SFNC feature, so devices that derive their rate from exposure and readout
        timing alone, which includes GenTL Producer simulators, do not implement it.

    Args:
        node_map: The GenICam node map of the connected camera.

    Returns:
        The AcquisitionFrameRate node, or None if the camera does not implement the feature.
    """
    return getattr(node_map, "AcquisitionFrameRate", None)


def _get_cti_path() -> Path:
    """Resolves and returns the path to the CTI file that provides the GenTL Producer interface.

    The returned path is not validated here, callers are responsible for validation via
    ``harvester.add_file(check_existence=True, check_validity=True)`` to avoid redundant Harvester instantiation.

    Notes:
        The ``AXVS_CTI_PATH`` environment variable takes precedence over the persisted path when it is set. Since the
        variable is inherited by spawned child processes, it redirects every process of a runtime to an alternative
        Producer without modifying the path persisted for future runtimes.

    Returns:
        The path to the GenTL Producer interface (.cti) file.

    Raises:
        FileNotFoundError: If the function is unable to resolve the path to the .cti file.
    """
    # Honors the runtime override before consulting the persisted path. The override is resolved, so that a relative
    # value is interpreted against the working directory in effect when it is read rather than reaching
    # harvester.add_file() unresolved. Every spawned child inherits that working directory, so each child resolves the
    # variable to the same file the parent did.
    override = os.environ.get(_CTI_PATH_VARIABLE)
    if override:
        return Path(override).expanduser().resolve()

    application_directory = Path(platformdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
    cti_path_file = application_directory / "cti_path.txt"

    if not cti_path_file.exists():
        message = (
            "Unable to resolve the path to the GenTL Producer interface (.cti) file to use for the harvesters camera "
            "interface, as the .cti file has not been set. Set the .cti file path by calling the 'axvs cti set' CLI "
            "command."
        )
        console.error(message=message, error=FileNotFoundError)

    with cti_path_file.open() as file:
        return Path(file.read().strip())


@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    """Silences verbose subprocess and driver output by redirecting stdout and stderr to os.devnull.

    The redirection happens at the file descriptor level, so it covers output written by native libraries that bypass
    the Python streams. The Harvesters library prints messages about missing features in the CTI file when calling
    update(), and the kernel driver emits V4L2 ioctl warnings while OpenCV probes camera indices.
    """
    devnull = os.open(path=os.devnull, flags=os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(fd=devnull, fd2=1)
    os.dup2(fd=devnull, fd2=2)
    try:
        yield
    finally:
        os.dup2(fd=old_stdout, fd2=1)
        os.dup2(fd=old_stderr, fd2=2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


@contextmanager
def _suppress_loader_error_dialog() -> Generator[None, None, None]:
    """Reports a Windows dynamic loader failure through the raised exception alone, without a message box.

    Windows treats a file that is not a valid dynamic library as a critical error and announces it in a 'Bad Image'
    message box, which blocks the thread that attempted the load until a human dismisses it. The Producer validity
    check loads the configured file, so a mistyped or damaged Producer would otherwise stall the CLI, the MCP server,
    and the test suite behind a dialog no headless caller is able to answer. Every other platform reports the same
    failure through the raised exception.
    """
    if _SET_THREAD_ERROR_MODE is None:
        yield
        return

    # Overrides the error mode of the calling thread rather than the process, so that the application the library runs
    # inside keeps the error handling it configured for its own threads.
    previous_mode = ctypes.c_uint()
    _SET_THREAD_ERROR_MODE(_FAIL_CRITICAL_ERRORS_MODE, ctypes.byref(previous_mode))
    try:
        yield
    finally:
        _SET_THREAD_ERROR_MODE(previous_mode.value, None)
