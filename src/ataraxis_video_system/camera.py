"""Provides a unified API that allows other library modules to interface with any supported camera hardware.

Primarily, these interfaces abstract the necessary procedures to connect to the camera and continuously grab the
acquired frames.
"""

import os
from enum import StrEnum
from typing import Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Generator

import cv2
import numpy as np
import appdirs
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer, TimerPrecisions
from harvesters.core import Harvester, ImageAcquirer  # type: ignore[import-untyped]
from harvesters.util.pfnc import (  # type: ignore[import-untyped]
    bgr_formats,
    rgb_formats,
    mono_location_formats,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from .saver import InputPixelFormats


@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    """Silences verbose outputs from the Harvesters library by redirecting stdout and stderr to os.devnull.

    The Harvesters library prints messages about missing features in the CTI file when calling update(). This context
    manager suppresses those printouts by temporarily redirecting stdout and stderr at the file descriptor level.
    """
    # Redirects stdout (fd 1) and stderr (fd 2) to devnull
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        # Restores stdout and stderr
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


# Repackages Harvesters color formats into sets to optimize the efficiency of the HarvestersCamera grab_frames() method:
_mono_formats = set(mono_location_formats)
_color_formats = set(bgr_formats) | set(rgb_formats)
_all_rgb_formats = set(rgb_formats)

# Determines the size of the frame pool used by the MockCamera instances.
_FRAME_POOL_SIZE = 10

# Determines the maximum number of failed test attempts allowed when running the get_opencv_ids() function before the
# runtime is terminated. This is used to interrupt the function early when it runs out of valid OpenCV-compatible camera
# objects to test.
_MAXIMUM_NON_WORKING_IDS = 5


class CameraInterfaces(StrEnum):
    """Specifies the supported camera interface backends compatible with the VideoSystem class."""

    HARVESTERS = "harvesters"
    """
    This is the preferred backend for all cameras that support the GeniCam standard, which includes most scientific and
    industrial machine-vision cameras. This backend is based on the 'Harvesters' library and works with all 
    GeniCam-compatible interfaces (USB, Ethernet, PCIE).
    """
    OPENCV = "opencv"
    """
    This is the backend used for all cameras that do not support the GeniCam standard. This backend is based on the 
    'OpenCV' library and primarily works for consumer-grade cameras that use the USB interface.
    """
    MOCK = "mock"
    """
    This backend is used exclusively for internal library testing and should not be used in production projects.
    """


@dataclass()
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
    """The frame rate at which the camera acquires frames, in frames per second."""
    serial_number: str | None = None
    """Only for Harvesters-discoverable cameras. Contains the camera's serial number."""
    model: str | None = None
    """Only for Harvesters-discoverable cameras. Contains the camera's model name."""


def _get_opencv_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about the cameras accessible through
    the OpenCV library.

    Notes:
        Currently, it is impossible to retrieve serial numbers or camera models from OpenCV. Therefore, while this
        method tries to provide some ID information, it is typically insufficient to identify specific cameras. It is
        advised to test each discovered camera with the 'axvs run' CLI command to identify the mapping between the
        discovered indices (IDs) and physical cameras.

    Returns:
         A tuple of CameraData instances, one for each discovered OpenCV-compatible camera.
    """
    # Disables OpenCV error logging to avoid flushing the terminal with failed connection attempts.
    prev_log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)

    try:
        non_working_count = 0
        working_ids: list[CameraInformation] = []

        # This loop iterates over IDs until it discovers 5 non-working IDs. The loop is designed to evaluate 100 IDs at
        # maximum to prevent infinite execution.
        for evaluated_id in range(100):
            try:
                # Evaluates each ID (index) by instantiating a video-capture object and reading one image and dimension
                # data from the connected camera (if any was connected).
                camera = cv2.VideoCapture(evaluated_id)

                # If the evaluated camera can be connected and returns images, it's index is appended to the ID list
                if camera.isOpened() and camera.read()[0]:
                    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    acquisition_rate = int(camera.get(cv2.CAP_PROP_FPS))
                    camera_data = CameraInformation(
                        camera_index=evaluated_id,
                        interface=CameraInterfaces.OPENCV,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        acquisition_frame_rate=acquisition_rate,
                    )
                    working_ids.append(camera_data)
                    non_working_count = 0  # Resets non-working count whenever a working camera is found.
                else:
                    non_working_count += 1

                camera.release()  # Releases the camera object to recreate it above for the next cycle

            except Exception as e:
                # Marks any ID that raises a runtime error as non-working and notifies the user.
                console.echo(
                    message=f"OpenCV camera discovery: Failed to evaluate camera index {evaluated_id}. Error: {e}",
                    level=LogLevel.WARNING,
                )
                non_working_count += 1

            # Breaks the loop early if more than 5 non-working IDs are found consecutively
            if non_working_count >= _MAXIMUM_NON_WORKING_IDS:
                break

        return tuple(working_ids)  # Converts to tuple before returning to caller.

    finally:
        # Restores previous log level
        cv2.setLogLevel(prev_log_level)


def _get_harvesters_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about the cameras accessible
    through the Harvesters library.

    Notes:
        This method bundles the discovered ID (index) information with the serial number and the model for each camera
        to support identifying the cameras.

    Returns:
        A tuple of CameraInformation instances, one for each discovered Harvesters-compatible camera.
    """
    # Instantiates the class and adds the input .cti file.
    harvester = Harvester()
    harvester.add_file(file_path=str(_get_cti_path()))

    # Gets the list of accessible cameras. Suppresses stdout to avoid verbose printouts about missing CTI features.
    with _suppress_output():
        harvester.update()

    # Loops over all discovered cameras and retrieves detailed information from each camera
    working_ids: list[CameraInformation] = []
    for index, camera_info in enumerate(harvester.device_info_list):
        try:
            # Accesses the remote device node map to get camera properties
            camera = harvester.create(search_key=index)
            node_map = camera.remote_device.node_map

            # Retrieves frame dimensions and acquisition rate from the camera's node map.
            frame_width = int(node_map.Width.value)
            frame_height = int(node_map.Height.value)
            acquisition_rate = int(round(number=node_map.AcquisitionFrameRate.value, ndigits=0))

            # Creates CameraInformation instance with all retrieved data
            camera_data = CameraInformation(
                camera_index=index,  # Uses the enumerated index as the camera index
                interface=CameraInterfaces.HARVESTERS,
                frame_width=frame_width,
                frame_height=frame_height,
                acquisition_frame_rate=acquisition_rate,
                serial_number=camera_info.serial_number,
                model=camera_info.model,
            )
            working_ids.append(camera_data)

        except Exception as e:
            # Skips any device that cannot be connected or queried for any reason and notifies the user.
            console.echo(
                message=f"Harvesters camera discovery: Failed to query device at index {index}. Error: {e}",
                level=LogLevel.WARNING,
            )
            continue

    # Resets the harvester instance after discovering the camera IDs.
    harvester.remove_file(file_path=str(_get_cti_path()))
    harvester.reset()

    return tuple(working_ids)  # Converts to tuple before returning to caller.


def discover_camera_ids() -> tuple[CameraInformation, ...]:
    """Discovers and reports the identifier (indices) and descriptive information about all accessible cameras.

    This function discovers cameras through both OpenCV and Harvesters interfaces and returns a combined tuple of
    CameraInformation instances. OpenCV cameras are discovered first, followed by Harvesters cameras (if a CTI file
    has been configured).

    Notes:
        For OpenCV cameras, it is impossible to retrieve serial numbers or camera models. It is advised to test each
        discovered OpenCV camera with the 'axvs run' CLI command to identify the mapping between the discovered indices
        and physical cameras.

        For Harvesters cameras, this function requires a valid CTI file to be configured via the add_cti_file()
        function or the 'axvs cti' CLI command. If no CTI file is configured, Harvesters camera discovery is skipped.

    Returns:
        A tuple of CameraInformation instances for all discovered cameras from both interfaces.
    """
    # Discovers OpenCV-compatible cameras.
    opencv_cameras = _get_opencv_ids()

    # Attempts to discover Harvesters-compatible cameras. Skips if no CTI file is configured.
    try:
        harvesters_cameras = _get_harvesters_ids()
    except FileNotFoundError:
        # No CTI file configured, skips Harvesters discovery.
        harvesters_cameras = ()

    return opencv_cameras + harvesters_cameras


def add_cti_file(cti_path: Path) -> None:
    """Configures the 'harvesters' camera interface to use the provided .cti file during all future runtimes.

    The 'harvesters' camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    compatible GenTL devices (cameras). This function configures the local machine to use the specified .cti file
    for all future runtimes that use the 'harvesters' camera interface.

    Notes:
        The path to the .cti file is stored inside the user's data directory, so that it can be reused between library
        calls.

    Args:
        cti_path: The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, is also acceptable.
            See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.
    """
    # Verifies the input CTI file.
    harvester = Harvester()
    harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)

    # Resolves the path to the library-specific .txt file used to store the path to the currently used .cti file.
    app_dir = Path(appdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
    cti_path_file = app_dir.joinpath("cti_path.txt")

    # In case this function is called before the app directory is created, ensures the app directory exists
    ensure_directory_exists(cti_path_file)

    # Overwrites the contents of the path file with the verified .cti file path.
    with cti_path_file.open("w") as f:
        f.write(str(cti_path))


def check_cti_file() -> Path | None:
    """Checks whether the library is configured to use a GenTL Producer interface (.cti) file.

    The 'harvesters' camera interface requires the GenTL Producer interface (.cti) file to discover and interface with
    compatible GenTL devices (cameras). This function checks if a valid .cti file path has been configured and returns
    the path if it exists.

    Returns:
        The Path to the configured .cti file if one exists and is valid, or None otherwise.
    """
    # Resolves the path to the .cti path file using appdirs.
    app_dir = Path(appdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
    cti_path_file = app_dir.joinpath("cti_path.txt")

    # Checks if the path file exists.
    if not cti_path_file.exists():
        return None

    # Reads the stored .cti file path.
    with cti_path_file.open() as f:
        cti_path = Path(f.read().strip())

    # Verifies the CTI file still exists and is valid.
    try:
        harvester = Harvester()
        harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)
    except Exception:
        # The configured CTI file is no longer valid.
        return None
    else:
        return cti_path


def _get_cti_path() -> Path:
    """Resolves and returns the path to the CTI file that provides the GenTL Producer interface.

    This service function is used when initializing HarvestersCamera instances to resolve the GenTL Producer interface.

    Returns:
        The path to the GenTL Producer interface (.cti) file.

    Raises:
        FileNotFoundError: If the function is unable to resolve the path to the .cti file.
    """
    # Uses appdirs to locate the user's data directory and resolve the path to the .cti path file.
    app_dir = Path(appdirs.user_data_dir(appname="ataraxis_video_system", appauthor="sun_lab"))
    cti_path_file = app_dir.joinpath("cti_path.txt")

    # If the path file or the library data directory does not exist, aborts with an error.
    if not cti_path_file.exists():
        message = (
            "Unable to resolve the path to the GenTL Producer interface (.cti) file to use for the harvesters camera "
            "interface, as the .cti file has not been set. Set the .cti file path by calling the 'axvs cti' CLI "
            "command."
        )
        console.error(message=message, error=FileNotFoundError)

    # Once the location of the path storage file is resolved, reads the .cti file path.
    with cti_path_file.open() as f:
        cti_path = Path(f.read().strip())

    # Verifies the CTI file's validity before returning it to the caller.
    harvester = Harvester()
    harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)
    return cti_path


class OpenCVCamera:
    """Interfaces with the specified OpenCV-compatible camera hardware to acquire frame data.

    Notes:
        This class should not be initialized manually! Use the VideoSystem's add_camera() method to create all camera
        interface instances.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        color: Specifies whether the camera acquires colored or monochrome images. This determines how to store the
            acquired frames. Colored frames are saved using the 'BGR' channel order, monochrome images are reduced to
            a single-channel format.
        camera_index: The index of the camera in the list of all cameras discoverable by OpenCV, e.g.: 0 for the first
            available camera, 1 for the second, etc. This specifies the camera hardware the instance should interface
            with at runtime.
        frame_rate: The desired rate, in frames per second, at which to capture the data. Note; whether the requested
            rate is attainable depends on the hardware capabilities of the camera and the communication interface. If
            this argument is not explicitly provided, the instance uses the default frame rate of the connected camera.
        frame_width: The desired width of the acquired frames, in pixels. Note; the requested width must be compatible
            with the range of frame dimensions supported by the camera hardware. If this argument is not explicitly
            provided, the instance uses the default frame width of the connected camera.
        frame_height: Same as 'frame_width', but specifies the desired height of the acquired frames, in pixels. If this
            argument is not explicitly provided, the instance uses the default frame height of the connected camera.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _color: Specifies whether the camera acquires colored or monochrome images.
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
        # Saves class parameters to class attributes
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
        # Prevents re-connecting to an already connected camera
        if self._camera is not None:
            return

        # Instantiates the OpenCV VideoCapture object to acquire images from the camera, using the specified camera ID
        # (index).
        self._camera = cv2.VideoCapture(index=self._camera_index, apiPreference=cv2.CAP_ANY)

        # If necessary, overrides the requested camera acquisition parameters. If the camera does not accept the
        # requested parameters, terminates with an error message. Otherwise, queries the acquisition parameters from
        # the connected camera.
        if self._frame_rate != 0:
            self._camera.set(propId=cv2.CAP_PROP_FPS, value=float(self._frame_rate))
            actual_frame_rate = int(self._camera.get(propId=cv2.CAP_PROP_FPS))
            if actual_frame_rate < self._frame_rate:
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
        # Prevents disconnecting from an already disconnected camera
        if self._camera is None:
            return

        # Disconnects from the camera
        self._camera.release()
        self._acquiring = False
        self._camera = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the instance is connected to the camera hardware."""
        return self._camera is not None

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames."""
        return self._acquiring

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
        # Prevents calling this method before connecting to the camera's hardware
        if self._camera is None:
            message = (
                f"The OpenCVCamera instance for the VideoSystem with id {self._system_id} is not connected to the "
                f"camera hardware, and cannot acquire images. Call the connect() method prior to calling the "
                f"grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)
            # Fallback to appease mypy, should not be reachable
            raise ConnectionError(message)  # pragma: no cover

        # Flips the acquisition tracker to True the first time this method is called for a connected camera.
        if not self._acquiring:
            self._acquiring = True

        frame: NDArray[np.floating[Any] | np.integer[Any]]
        success, frame = self._camera.read()
        if not success:
            message = (
                f"The OpenCVCamera instance for the VideoSystem with id {self._system_id} has failed to grab a frame "
                f"image from the camera hardware, which is not expected. This indicates initialization or connectivity "
                f"issues."
            )
            console.error(message=message, error=BrokenPipeError)

        if not self._color:
            # Converts the frame data from using BGR color space (default for all frames) to Monochrome if needed
            frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        return frame


class HarvestersCamera:
    """Interfaces with the specified GeniCam-compatible camera hardware to acquire frame data.

    Notes:
        This class should not be initialized manually! Use the VideoSystem's add_camera() method to create all camera
        interface instances.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        camera_index: The index of the camera in the list of all cameras discoverable by Harvesters, e.g.: 0 for the
            first available camera, 1 for the second, etc. This specifies the camera hardware the instance should
            interface with at runtime.
        frame_rate: The desired rate, in frames per second, at which to capture the data. Note; whether the requested
            rate is attainable depends on the hardware capabilities of the camera and the communication interface. If
            this argument is not explicitly provided, the instance uses the default frame rate of the connected camera.
        frame_width: The desired width of the acquired frames, in pixels. Note; the requested width must be compatible
            with the range of frame dimensions supported by the camera hardware. If this argument is not explicitly
            provided, the instance uses the default frame width of the connected camera.
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
    """

    def __init__(
        self,
        system_id: int,
        camera_index: int = 0,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._system_id: int = system_id
        self._camera_index: int = camera_index
        self._frame_rate: int = 0 if frame_rate is None else frame_rate
        self._frame_width: int = 0 if frame_width is None else frame_width
        self._frame_height: int = 0 if frame_height is None else frame_height

        # Pre-creates the attribute to store the initialized Harvester class to discover the list of available cameras.
        # While the object was pickleable in earlier Harvesters versions, it is now not pickleable and must be handled
        # similar to how ImageAcquirer objects are handled.
        self._harvester: Harvester | None = None

        # Pre-creates the attribute to store the initialized ImageAcquirer object for the connected camera.
        self._camera: ImageAcquirer | None = None

        # Tracks whether the acquired frames use a monochrome or a colored data format.
        self._color: bool = False

    def __del__(self) -> None:
        """Releases the underlying ImageAcquirer object when the instance is garbage-collected."""
        self.disconnect()  # Releases the camera object

    def __repr__(self) -> str:
        """Returns the string representation of the HarvestersCamera instance."""
        return (
            f"HarvestersCamera(system_id={self._system_id}, camera_index={self._camera_index}, "
            f"frame_rate={self.frame_rate} frames / second, frame_width={self.frame_width} pixels, "
            f"frame_height={self.frame_height} pixels, connected={self._camera is not None}, "
            f"acquiring={self.is_acquiring})"
        )

    def connect(self) -> None:
        """Connects to the managed camera hardware."""
        # Prevents connecting to an already connected camera.
        if self._camera is not None:
            return

        # Initializes the Harvester class to discover the list of available cameras.
        self._harvester = Harvester()
        # Adds the .cti file to the class. This also verifies the file's existence and validity.
        self._harvester.add_file(file_path=str(_get_cti_path()), check_existence=True, check_validity=True)
        # Discovers compatible cameras using the GenTL interface. Suppresses stdout to avoid verbose CTI printouts.
        with _suppress_output():
            self._harvester.update()

        # Initializes an ImageAcquirer camera interface object to interface with the camera's hardware.
        self._camera = self._harvester.create(search_key=self._camera_index)

        # If necessary, overrides the requested camera acquisition parameters. Note, there is no guarantee that the
        # camera accepts the requested parameters.
        if self._frame_width != 0:
            self._camera.remote_device.node_map.Width.value = self._frame_width
        if self._frame_height != 0:
            self._camera.remote_device.node_map.Height.value = self._frame_height
        # The frame rate has to be set last, as it is affected by frame width and height
        if self._frame_rate != 0:
            self._camera.remote_device.node_map.AcquisitionFrameRate.value = self._frame_rate

        # Queries the current camera acquisition parameters and stores them in class attributes.
        self._frame_rate = int(self._camera.remote_device.node_map.AcquisitionFrameRate.value)
        self._frame_width = int(self._camera.remote_device.node_map.Width.value)
        self._frame_height = int(self._camera.remote_device.node_map.Height.value)

    def disconnect(self) -> None:
        """Disconnects from the managed camera hardware."""
        # Prevents disconnecting from an already disconnected camera.
        if self._camera is None or self._harvester is None:
            return

        self._camera.stop()  # Stops image acquisition

        # Discards any unconsumed buffers to ensure proper memory release
        while self._camera.num_holding_filled_buffers != 0:
            _ = self._camera.fetch()  # pragma: no cover

        self._camera.destroy()  # Releases the camera object
        self._camera = None  # Sets the camera object to None
        self._harvester.reset()  # Resets and removes the Harvester object
        self._harvester = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the instance is connected to the camera hardware."""
        return self._camera is not None

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames."""
        if self._camera is not None:
            return bool(self._camera.is_acquiring())
        return False  # If the camera is not connected, it cannot be acquiring images.

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
        if not self._camera:
            message = (
                f"The HarvestersCamera instance for the VideoSystem with id {self._system_id} is not connected to the "
                f"camera hardware and cannot acquire images. Call the connect() method prior to calling the "
                f"grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)
            # Fallback to appease mypy, should not be reachable
            raise ConnectionError(message)  # pragma: no cover

        # Triggers camera frame acquisition the first time this method is called.
        if not self._camera.is_acquiring():
            self._camera.start()

        # Retrieves the next available image buffer from the camera. Uses the 'with' context to properly
        # re-queue the buffer to acquire further images.
        with self._camera.fetch() as buffer:
            if buffer is None:  # pragma: no cover
                message = (
                    f"The HarvestersCamera instance for the VideoSystem with id {self._system_id} has failed to grab "
                    f"a frame image from the camera hardware, which is not expected. This indicates initialization or "
                    f"connectivity issues."
                )
                console.error(message=message, error=BrokenPipeError)

            # Retrieves the contents (frame data) from the buffer
            content = buffer.payload.components[0]

            # Collects the information necessary to reshape the originally 1-dimensional frame array into the
            # 2-dimensional array using the correct number and order of color channels.
            width = content.width
            height = content.height
            data_format = content.data_format

            # For monochrome formats, reshapes the 1D array into a 2D array and returns it to caller.
            if data_format in mono_location_formats:
                # Uses copy, which is VERY important. Once the buffer is released, the original 'content' is lost,
                # so NumPy needs to copy the data instead of using the default referencing behavior.
                out_array: NDArray[np.integer[Any]] = content.data.reshape(height, width).copy()
                self._color = False  # Ensures that the color flag is set to False.
                return out_array

            # For color data, evaluates the input format and reshapes the data as necessary.
            if data_format in _color_formats:  # pragma: no cover
                # Reshapes the data into RGB + A format as the first processing step.
                content.data.reshape(
                    height,
                    width,
                    int(content.num_components_per_pixel),  # Sets of R, G, B, and Alpha
                )

                # Swaps every R and B value (RGB → BGR) ot produce BGR / BGRA images. This ensures consistency
                # with the OpenCVCamera API. Note, this is only done if the image data is in the RGB format.
                if data_format in _all_rgb_formats:
                    frame: NDArray[np.integer[Any]] = content[:, :, ::-1].copy()

                self._color = True  # Ensures that the color flag is set to True.

                # Returns the reshaped frame array to the caller
                return frame

            # If the image has an unsupported data format, raises an error
            message = (
                f"The HarvestersCamera instance for the VideoSystem with id {self._system_id} has acquired an image "
                f"with an unsupported data (color) format {data_format}. Currently, only the following unpacked "
                f"families of color formats are supported: Monochrome, RGB, RGBA, BGR, and BGRA."
            )  # pragma: no cover
            console.error(message=message, error=ValueError)  # pragma: no cover
            # This should never be reached, it is here to appease mypy
            raise RuntimeError(ValueError)  # pragma: no cover


class MockCamera:
    """Simulates (mocks) the behavior of the OpenCVCamera and HarvestersCamera classes without the need to interface
    with a physical camera.

    This class is primarily used to test the VideoSystem class functionality. The class fully mimics the behavior of
    other camera interface classes but does not establish a physical connection with any camera hardware.

    Notes:
        This class should not be initialized manually! Use the VideoSystem's add_camera() method to create all camera
        interface instances.

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this camera interface.
        frame_rate: The simulated frame acquisition rate of the camera, in frames per second.
        frame_width: The simulated camera frame width, in pixels.
        frame_height: The simulated camera frame height, in pixels.
        color: The simulated camera frame color mode. If True, the frames are generated using the BGR color mode. If
            False, the frames are generated using the grayscale (monochrome) color mode.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _color: Determines whether to simulate monochrome or RGB frame images.
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
        # Saves class parameters to class attributes
        self._system_id: int = system_id
        self._color: bool = color
        self._frame_rate: int = 30 if frame_rate is None else frame_rate
        self._frame_width: int = 600 if frame_width is None else frame_width
        self._frame_height: int = 400 if frame_height is None else frame_height
        self._camera: bool = False
        self._acquiring: bool = False

        # Creates a random number generator to be used below
        rng = np.random.default_rng(seed=42)  # Specifies a reproducible seed.

        # To allow reproducible testing, the class statically generates a pool of 10 images used during the grab_frame()
        # method calls.
        frames_list: list[NDArray[np.uint8]] = []
        for _ in range(10):
            if self._color:
                frame = rng.integers(0, 256, size=(self._frame_height, self._frame_width, 3), dtype=np.uint8)
                # Ensures the order of the colors is BGR
                bgr_frame: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
                    src=frame, code=cv2.COLOR_RGB2BGR
                )
                frames_list.append(bgr_frame)
            else:
                # grayscale frames have only one channel, so order does not matter
                frames_list.append(
                    rng.integers(0, 256, size=(self._frame_height, self._frame_width, 1), dtype=np.uint8)
                )

        # Casts to a tuple to optimize runtime efficiency
        self._frames: tuple[NDArray[np.uint8], ...] = tuple(frames_list)
        self._current_frame_index: int = 0

        # Cannot be initialized here due to the use of multiprocessing in the VideoSystem class.
        self._timer: PrecisionTimer | None = None

        # Uses the frame_rate to derive the number of microseconds that has to pass between each frame acquisition.
        # This is used to simulate the camera's frame rate during grab_frame() runtime.
        self._time_between_frames: float = 1000 / self._frame_rate

    def connect(self) -> None:
        """Simulates connecting to the camera hardware."""
        self._camera = True

        # Uses millisecond precision, which supports simulating up to 1000 fps. The time has to be initialized here to
        # make the class compatible with the VideoSystem class that uses multiprocessing.
        self._timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    def disconnect(self) -> None:
        """Simulates disconnecting from the camera hardware."""
        self._camera = False
        self._acquiring = False
        self._timer = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the instance is 'connected' to the camera hardware."""
        return self._camera

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently 'acquiring' video frames."""
        return self._acquiring

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
    def frame_pool(self) -> tuple[NDArray[np.uint8], ...]:
        """Returns the pool of camera frames sampled by the grab_frame() method."""
        return self._frames

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
            A NumPy array that stores the frame data. Depending on whether the camera acquires colored or monochrome
            images, the returned arrays have the shape (height, width, channels) or (height, width). Color data uses
            the BGR channel order.

        Raises:
            RuntimeError: If the method is called for a class not currently 'connected' to a camera.
        """
        # Prevents calling this method before connecting to the camera's hardware
        if not self._camera:
            message = (
                f"The MockCamera instance for the VideoSystem with id {self._system_id} is not currently simulating "
                f"connection to the camera hardware, and cannot simulate image acquisition. Call the connect() method "
                f"prior to calling the grab_frame() method."
            )
            console.error(message=message, error=ConnectionError)
            # Fallback to appease mypy, should not be reachable
            raise ConnectionError(message)  # pragma: no cover

        # Flips the acquiring flag the first time this method is called
        if not self._acquiring:
            self._acquiring = True

        # Fallback to appease mypy, the time should always be initialized at this point
        if self._timer is None:
            self._timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # All camera interfaces are designed to block in-place if the frame is not available. Here, this behavior
        # is simulated by using the timer class to 'force' the method to work at a certain frame rate.
        while self._timer.elapsed < self._time_between_frames:
            pass

        # 'Acquires' a frame from the frame pool
        frame = self._frames[self._current_frame_index].copy()

        # Resets the timer to measure the time elapsed since the last frame acquisition.
        self._timer.reset()

        # Increments the frame pool index. When the index reaches the end of the pool, this resets it back to the
        # start of the pool. This simulates the behavior of a cyclic buffer.
        self._current_frame_index = (self._current_frame_index + 1) % _FRAME_POOL_SIZE

        # Returns the acquired frame to caller
        return frame
