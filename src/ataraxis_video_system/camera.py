"""This module provides a unified API that allows other library modules to interface with any supported camera hardware.
Primarily, these interfaces abstract the necessary procedures to connect to the camera and continuously grab the
acquired frames.
"""

import os
from enum import StrEnum
from typing import Any
from pathlib import Path

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Improves OpenCV's performance on Windows.
import cv2
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from harvesters.core import Harvester, ImageAcquirer
from harvesters.util.pfnc import (
    bgr_formats,
    rgb_formats,
    bgra_formats,
    rgba_formats,
    mono_location_formats,
)
from ataraxis_base_utilities import console

# Repackages Harvesters color formats into sets to optimize the efficiency of the HarvestersCamera grab_frames() method:
mono_formats = set(mono_location_formats)
color_formats = set(bgr_formats) | set(rgb_formats) | set(bgra_formats) | set(rgba_formats)
all_rgb_formats = set(rgb_formats) | set(rgba_formats)

# Determines the size of the frame pool used by the MockCamera instances.
_FRAME_POOL_SIZE = 10


class CameraBackends(StrEnum):
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
            argument is not explicitly provided, the instance uses the default frame width of the connected camera.

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
        """Connects to the managed camera hardware."""
        # Prevents re-connecting to an already connected camera
        if self._camera is not None:
            return

        # Instantiates the OpenCV VideoCapture object to acquire images from the camera, using the specified camera ID
        # (index).
        self._camera = cv2.VideoCapture(index=self._camera_index, apiPreference=cv2.CAP_ANY)

        # If necessary, overrides the requested camera acquisition parameters. Note, there is no guarantee that the
        # camera accepts the requested parameters.
        if self._frame_rate != 0.0:
            self._camera.set(propId=cv2.CAP_PROP_FPS, value=float(self._frame_rate))  # pragma: no cover
        if self._frame_width != 0:
            self._camera.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=float(self._frame_width))  # pragma: no cover
        if self._frame_height != 0:
            self._camera.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=float(self._frame_height))  # pragma: no cover

        # Queries the current camera acquisition parameters and stores them in class attributes.
        self._frame_rate = int(self._camera.get(propId=cv2.CAP_PROP_FPS))
        self._frame_width = int(self._camera.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
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

    def grab_frame(self) -> NDArray[np.integer[Any]]:
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
        cti_path: The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, us also acceptable.
            See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.
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
            argument is not explicitly provided, the instance uses the default frame width of the connected camera.

    Attributes:
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this camera interface.
        _camera_index: Stores the index of the camera hardware in the list of all OpenCV-discoverable cameras connected
            to the host-machine.
        _frame_rate: Stores the camera's frame acquisition rate.
        _frame_width: Stores the width of the camera's frames.
        _frame_height: Stores the height of the camera's frames.
        _harvester: Stores the Harvester interface object that discovers and manages the list of accessible GenTL
            cameras.
        _camera: Stores the Harvesters ImageAcquirer object that interfaces with the camera.
    """

    def __init__(
        self,
        system_id: int,
        cti_path: Path,
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

        # Initializes the Harvester class to discover the list of available cameras.
        self._harvester: Harvester = Harvester()
        # Adds the .cti file to the class. This also verifies the file's existence and validity.
        self._harvester.add_file(file_path=str(cti_path), check_existence=True, check_validity=True)
        self._harvester.update()  # Discovers compatible cameras using the GenTL interface specified by the CTI file.

        # Pre-creates the attribute to store the initialized ImageAcquirer object for the connected camera.
        self._camera: ImageAcquirer | None = None

    def __del__(self) -> None:
        """Releases the underlying ImageAcquirer object when the instance is garbage-collected."""
        self.disconnect()  # Releases the camera object
        self._harvester.reset()  # Releases the Harvester class resources

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
        # Precents disconnecting from an already disconnected camera.
        if self._camera is None:
            return

        self._camera.stop()  # Stops image acquisition

        # Discards any unconsumed buffers to ensure proper memory release
        while self._camera.num_holding_filled_buffers != 0:
            _ = self._camera.fetch()  # pragma: no cover

        self._camera.destroy()  # Releases the camera object
        self._camera = None  # Sets the camera object to None

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
            width = content.frame_width
            height = content.frame_height
            data_format = content.data_format

            # For monochrome formats, reshapes the 1D array into a 2D array and returns it to caller.
            if data_format in mono_location_formats:
                # Uses copy, which is VERY important. Once the buffer is released, the original 'content' is lost,
                # so NumPy needs to copy the data instead of using the default referencing behavior.
                out_array: NDArray[np.integer[Any]] = content.data.reshape(height, width).copy()
                return out_array

            # For color data, evaluates the input format and reshapes the data as necessary.
            if data_format in color_formats:  # pragma: no cover
                # Reshapes the data into RGB + A format as the first processing step.
                content.data.reshape(
                    height,
                    width,
                    int(content.num_components_per_pixel),  # Sets of R, G, B, and Alpha
                )

                # Swaps every R and B value (RGB → BGR) ot produce BGR / BGRA images. This ensures consistency
                # with the OpenCVCamera API. Note, this is only done if the image data is in the RGB format.
                if data_format in all_rgb_formats:
                    frame: NDArray[np.integer[Any]] = content[:, :, ::-1].copy()

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
        frame_rate: float | None = None,
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
        self._frame_height: int = 600 if frame_height is None else frame_height
        self._camera: bool = False
        self._acquiring: bool = False

        # Creates a random number generator to be sued below
        rng = np.random.default_rng(seed=42)  # Specifies a reproducible seed.

        # To allow reproducible testing, the class statically generates a pool of 10 images used during the grab_frame()
        # method calls.
        frames_list: list[NDArray[np.uint8]] = []
        for _ in range(10):
            if self._color:
                frame = rng.integers(0, 256, size=(self._frame_height, self._frame_width, 3), dtype=np.uint8)
                bgr_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)  # Ensures the order of the colors is BGR
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
        self._timer = PrecisionTimer("ms")

    def disconnect(self) -> None:
        """Simulates disconnecting from the camera hardware."""
        self._camera = False
        self._acquiring = False
        self._timer = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the instance is 'connected' to the camera hardware."""
        return self._camera is not None

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
            self._timer = PrecisionTimer("ms")

        # All camera interfaces are designed to block in-place if the frame is not available. Here, this behavior
        # is simulated by using the timer class to 'force' the method to work at a certain frame rate.
        while self._timer.elapsed < self._time_between_frames:
            pass

        # 'Acquires' a frame from the frame pool
        frame = self._frames[self._current_frame_index].copy()

        # Resets the timer to measure the time elapsed since the last frame acquisition.
        self._timer.reset()

        # Increments the flame pool index. When the index reaches the end of the pool, this resets it back to the
        # starts of the popol. This simulates the behavior of a cyclic buffer.
        self._current_frame_index = (self._current_frame_index + 1) % _FRAME_POOL_SIZE

        # Returns the acquired frame to caller
        return frame
