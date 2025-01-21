from enum import Enum
from typing import Any
from pathlib import Path

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

class CameraBackends(Enum):
    """Maps valid literal values used to specify Camera class backend when requesting it from the create_camera ()
    method of the VideoSystem class to programmatically callable variables.

    Use this enumeration instead of 'hardcoding' Camera backends where possible to automatically adjust to future API
    changes to this library.

    The backend determines the low-level functioning of the Camera class and is, therefore, very important for
    optimizing video acquisition. It is generally advised to use the 'harvesters' backend with any GeniCam camera
    and only use opencv as a 'fallback' for camera that do not support GeniCam standard.
    """

    HARVESTERS: str
    OPENCV: str
    MOCK: str

class OpenCVCamera:
    """Wraps an OpenCV VideoCapture object and uses it to connect to, manage, and acquire data from the requested
    physical camera.

    This class exposes the necessary API to interface with any OpenCV-compatible camera. Due to the behavior of the
    OpenCV binding, it takes certain configuration parameters during initialization (desired fps and resolution) and
    passes it to the camera binding during connection.

    Notes:
        This class should not be initialized manually! Use the create_camera() method from VideoSystem class to create
        all camera instances.

        After frame-acquisition starts, some parameters, such as the fps or image dimensions, can no longer be altered.
        Commonly altered parameters have been made into initialization arguments to incentivize setting them to the
        desired values before starting frame-acquisition.

    Args:
        camera_id: The unique ID code of the camera instance. This is used to identify the camera and to mark all
            frames acquired from this camera.
        color: Specifies if the camera acquires colored or monochrome images. There is no way to get this
            information via OpenCV, and all images read from VideoCapture use BGR colorspace even for the monochrome
            source. This setting does not control whether the camera acquires colored images. It only controls how
            the class handles the images. Colored images will be saved using the 'BGR' channel order, monochrome
            images will be reduced to using only one channel.
        backend: The integer-code for the backend to use for the connected VideoCapture object. Generally, it
            is advised not to change the default value of this argument unless you know what you are doing.
        camera_index: The index of the camera, relative to all available video devices, e.g.: 0 for the first
            available camera, 1 for the second, etc.
        fps: The desired Frames Per Second to capture the frames at. Note, this depends on the hardware capabilities of
            the camera and is affected by multiple related parameters, such as image dimensions, camera buffer size and
            the communication interface. If not provided (set to None), this parameter will be obtained from the
            connected camera.
        width: The desired width of the camera frames to acquire, in pixels. This will be passed to the camera and
            will only be respected if the camera has the capacity to alter acquired frame resolution. If not provided
            (set to None), this parameter will be obtained from the connected camera.
        height: Same as width, but specifies the desired height of the camera frames to acquire, in pixels. If not
            provided (set to None), this parameter will be obtained from the connected camera.

    Attributes:
        _id: Stores the string-name of the camera.
        _color: Determines whether the camera acquires colored or monochrome images.
        _backend: Stores the code for the backend to be used by the connected VideoCapture object.
        _camera_index: Stores the index of the camera, which is used during connect() method runtime.
        _camera: Stores the OpenCV VideoCapture object that interfaces with the camera.
        _fps: Stores the desired Frames Per Second to capture the frames at.
        _width: Stores the desired width of the camera frames to acquire.
        _height: Stores the desired height of the camera frames to acquire.
        _acquiring: Stores whether the camera is currently acquiring video frames. This is statically set to 'True'
            the first time grab_frames() is called, as it initializes the camera acquisition thread of the binding
            object. If this attribute is True, some parameters, such as the fps, can no longer be altered.
        _backends: A dictionary that maps the meaningful backend names to the codes returned by VideoCapture
            get() method. This is used to convert integer values to meaningful names before returning them to the user.
    """

    _backends: dict[str, float]
    _id: Incomplete
    _color: Incomplete
    _backend: Incomplete
    _camera_index: Incomplete
    _camera: Incomplete
    _fps: Incomplete
    _width: Incomplete
    _height: Incomplete
    _acquiring: bool
    def __init__(
        self,
        camera_id: np.uint8,
        color: bool = True,
        backend: int = ...,
        camera_index: int = 0,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None: ...
    def __del__(self) -> None:
        """Ensures that the camera is disconnected upon garbage collection."""
    def __repr__(self) -> str:
        """Returns a string representation of the OpenCVCamera object."""
    def connect(self) -> None:
        """Initializes the camera VideoCapture object and sets the video acquisition parameters.

        This method has to be called before calling the grab_frames () method. It is used to initialize and prepare the
        camera for image collection.

        Notes:
            While this method passes acquisition parameters, such as fps and frame dimensions, to the camera, there is
            no guarantee they will be set. Cameras with a locked aspect ratio, for example, may not use incompatible
            frame dimensions. Be sure to verify that the desired parameters have been set by using class properties if
            necessary.
        """
    def disconnect(self) -> None:
        """Disconnects from the camera by releasing the VideoCapture object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during the VideoSystem shutdown procedure to properly release
        resources.
        """
    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the camera via a VideoCapture instance."""
    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames.

        This concerns the 'asynchronous' behavior of the wrapped camera object which, after grab_frames() class method
        has been called, continuously acquires and buffers images even if they are not retrieved.
        """
    @property
    def fps(self) -> float | None:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
    @property
    def width(self) -> int | None:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
    @property
    def height(self) -> int | None:
        """Returns the current frame height setting of the camera (in pixels).

        If the camera is connected, this is the actual frame height value the camera is set to produce. If the camera
        is not connected, this is the desired frame height value that will be passed to the camera during connection.
        """
    @property
    def backend(self) -> str:
        """Returns the descriptive string-name for the backend being used by the connected VideoCapture object.

        If the camera is connected, this is the actual backend used to interface with the camera. If the camera
        is not connected, this is the desired backend that will be used to initialize the VideoCapture object.

        Raises:
            ValueError: If the backend code used to retrieve the backend name is not one of the recognized backend
                codes.
        """
    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique identifier code of the camera."""
    def grab_frame(self) -> NDArray[Any]:
        """Grabs the first available frame from the camera buffer and returns it to caller as a NumPy array object.

        This method has to be called repeatedly to acquire new frames from the camera. The first time the method is
        called, the class is switched into the 'acquisition' mode and remains in this mode until the camera is
        disconnected. See the notes below for more information on how 'acquisition' mode works.

        Notes:
            The first time this method is called, the camera initializes image acquisition, which is carried out
            asynchronously. The camera saves the images into its circular buffer (if it supports buffering), and
            calling this method extracts the first image available in the buffer and returns it to caller.

            Due to the initial setup of the buffering procedure, the first call to this method will incur a significant
            delay of up to a few seconds. Therefore, it is advised to call this method ahead of time and either discard
            the first few frames or have some other form of separating initial frames from the frames extracted as
            part of the post-initialization runtime.

            Moreover, it is advised to design video acquisition runtimes around repeatedly calling this method for
            the entire runtime duration to steadily consume the buffered images. This is in contrast to having multiple
            image acquisition 'pulses', which may incur additional overhead.

        Returns:
            A NumPy array with the outer dimensions matching the preset camera frame dimensions. All returned frames
            use the BGR colorspace by default and will, therefore, include three additional color channel dimensions for
            each two-dimensional pixel index.

        Raises:
            RuntimeError: If the camera does not yield an image, or if the method is called for a class not currently
                connected to a camera.
        """

class HarvestersCamera:
    """Wraps a Harvesters ImageAcquirer object and uses it to connect to, manage, and acquire data from the requested
    physical GenTL-compatible camera.

    This class exposes the necessary API to interface with any GenTL-compatible camera. Due to the behavior of the
    accessor library, it takes certain configuration parameters during initialization (desired fps and resolution) and
    passes it to the camera binding during connection.

    Notes:
        This class should not be initialized manually! Use the create_camera() method from VideoSystem class to create
        all camera instances.

        After frame-acquisition starts, some parameters, such as the fps or image dimensions, can no longer be altered.
        Commonly altered parameters have been made into initialization arguments to incentivize setting them to the
        desired values before starting frame-acquisition.

    Args:
        camera_id: The unique ID code of the camera instance. This is used to identify the camera and to mark all
            frames acquired from this camera.
        cti_path: The path to the '.cti' file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire, would
            work as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.
        camera_index: The index of the camera, relative to all available video devices, e.g.: 0 for the first
            available camera, 1 for the second, etc.
        fps: The desired Frames Per Second to capture the frames at. Note, this depends on the hardware capabilities of
            the camera and is affected by multiple related parameters, such as image dimensions, camera buffer size and
            the communication interface. If not provided (set to None), this parameter will be obtained from the
            connected camera.
        width: The desired width of the camera frames to acquire, in pixels. This will be passed to the camera and
            will only be respected if the camera has the capacity to alter acquired frame resolution. If not provided
            (set to None), this parameter will be obtained from the connected camera.
        height: Same as width, but specifies the desired height of the camera frames to acquire, in pixels. If not
            provided (set to None), this parameter will be obtained from the connected camera.

    Attributes:
        _id: Stores the unique identifier code of the camera.
        _camera_index: Stores the index of the camera, which is used during connect() method runtime.
        _camera: Stores the Harvesters ImageAcquirer object that interfaces with the camera.
        _harvester: Stores the Harvester interface object that discovers and manages the list of accessible cameras.
        _fps: Stores the desired Frames Per Second to capture the frames at.
        _width: Stores the desired width of the camera frames to acquire.
        _height: Stores the desired height of the camera frames to acquire.
    """

    _id: Incomplete
    _camera_index: Incomplete
    _camera: Incomplete
    _fps: Incomplete
    _width: Incomplete
    _height: Incomplete
    _harvester: Incomplete
    def __init__(
        self,
        camera_id: np.uint8,
        cti_path: Path,
        camera_index: int = 0,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None: ...
    def __del__(self) -> None:
        """Ensures that the camera is disconnected upon garbage collection."""
    def __repr__(self) -> str:
        """Returns a string representation of the HarvestersCamera object."""
    def connect(self) -> None:
        """Initializes the camera ImageAcquirer object and sets the video acquisition parameters.

        This method has to be called before calling the grab_frames () method. It is used to initialize and prepare the
        camera for image collection. Note, the method does not automatically start acquiring images. Image acquisition
        starts with the first call to the grab_frames () method to make the API consistent across all our camera
        classes.

        Notes:
            While this method passes acquisition parameters, such as fps and frame dimensions, to the camera, there is
            no guarantee they will be set. Cameras with a locked aspect ratio, for example, may not use incompatible
            frame dimensions. Be sure to verify that the desired parameters have been set by using class properties if
            necessary.
        """
    def disconnect(self) -> None:
        """Disconnects from the camera by stopping image acquisition, clearing any unconsumed buffers, and releasing
        the ImageAcquirer object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during the VideoSystem shutdown procedure to properly release
        resources.
        """
    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the camera via an ImageAcquirer instance."""
    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames.

        This concerns the 'asynchronous' behavior of the wrapped camera object which, after grab_frames() class method
        has been called, continuously acquires and buffers images even if they are not retrieved.
        """
    @property
    def fps(self) -> float | None:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
    @property
    def width(self) -> int | None:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
    @property
    def height(self) -> int | None:
        """Returns the current frame height setting of the camera (in pixels).

        If the camera is connected, this is the actual frame height value the camera is set to produce. If the camera
        is not connected, this is the desired frame height value that will be passed to the camera during connection.
        """
    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique byte identifier of the camera."""
    def grab_frame(self) -> NDArray[Any]:
        """Grabs the first available frame from the camera buffer and returns it to caller as a NumPy array object.

        This method has to be called repeatedly to acquire new frames from the camera. The first time the method is
        called, the class is switched into the 'acquisition' mode and remains in this mode until the camera is
        disconnected. See the notes below for more information on how 'acquisition' mode works.

        Notes:
            The first time this method is called, the camera initializes image acquisition, which is carried out
            asynchronously. The camera saves the images into its circular buffer, and calling this method extracts the
            first image available in the buffer and returns it to caller.

            Due to the initial setup of the buffering procedure, the first call to this method will incur a significant
            delay of up to a few seconds. Therefore, it is advised to call this method ahead of time and either discard
            the first few frames or have some other form of separating initial frames from the frames extracted as
            part of the post-initialization runtime.

            Moreover, it is advised to design video acquisition runtimes around repeatedly calling this method for
            the entire runtime duration to steadily consume the buffered images. This is in contrast to having multiple
            image acquisition 'pulses', which may incur additional overhead.

        Returns:
            A NumPy array with the outer dimensions matching the preset camera frame dimensions. The returned frames
            will use either Monochrome, BGRA, or BGR color space. This means that the returned array will use between
            1 and 4 pixel-color channels for each 2-dimensional pixel position. The specific format depends on the
            format used by the camera. All images are converted to BGR to be consistent with OpenCVCamera behavior.

        Raises:
            RuntimeError: If the camera does not yield an image, or if the method is called for a class not currently
                connected to a camera.
        """

class MockCamera:
    """Simulates (mocks) the API behavior and functionality of the OpenCVCamera and HarvestersCamera classes.

    This class is primarily used to test VideoSystem class functionality without using a physical camera, which
    optimizes testing efficiency and speed. The class mimics the behavior of the 'real' camera classes but does not
    establish a physical connection with any camera hardware. The class accepts and returns static values that fully
    mimic the 'real' API.

    Notes:
        This class should not be initialized manually! Use the create_camera() method from VideoSystem class to create
        all camera instances.

        The class uses NumPy to simulate image acquisition where possible, generating 'white noise' images initialized
        with random-generator-derived pixel values. Depending on the 'color' argument, the generated images can be
        monochrome or RGB color images.

    Args:
        camera_id: The unique ID code of the camera instance. This is used to identify the camera and to mark all
            frames acquired from this camera.
        camera_index: The simulated list index of the camera.
        fps: The simulated Frames Per Second of the camera.
        width: The simulated camera frame width.
        height: The simulated camera frame height.
        color: Determines if the camera acquires colored or monochrome images. Colored images will be saved using the
            'BGR' channel order, monochrome images will be reduced to using only one channel.

    Attributes:
        _color: Determine whether the camera should produce monochrome or RGB images.
        _id: Stores the unique byte-code identifier for the camera.
        _camera_index: Stores the simulated index for the camera.
        _camera: A boolean variable used to track whether the camera is 'connected'.
        _fps: Stores the simulated Frames Per Second.
        _width: Stores the simulated camera frame width.
        _height: Stores the simulated camera frame height.
        _acquiring: Stores whether the camera is currently acquiring video frames.
        _frames: Stores the pool of pre-generated frame images used to simulate frame grabbing.
        _current_frame_index: The index of the currently evaluated frame in the pre-generated frame pool. This is used
            to simulate the cyclic buffer used by 'real' camera classes.
        _timer: After the camera is 'connected', this attribute is used to store the timer class that controls the
            output fps rate.
        _time_between_frames: Stores the number of milliseconds that has to pass between acquiring new frames. This is
            used to simulate the real camera fps rate.
    """

    _color: Incomplete
    _id: Incomplete
    _camera_index: Incomplete
    _camera: bool
    _fps: Incomplete
    _width: Incomplete
    _height: Incomplete
    _acquiring: bool
    _frames: Incomplete
    _current_frame_index: int
    _timer: Incomplete
    _time_between_frames: Incomplete
    def __init__(
        self,
        camera_id: np.uint8 = ...,
        camera_index: int = 0,
        fps: float = 30,
        width: int = 600,
        height: int = 600,
        *,
        color: bool = True,
    ) -> None: ...
    def connect(self) -> None:
        """Simulates connecting to the camera, which is a necessary prerequisite to grab frames from the camera."""
    def disconnect(self) -> None:
        """Simulates disconnecting from the camera, which is part of the broader camera shutdown procedure."""
    @property
    def is_connected(self) -> bool:
        """Returns True if the class is 'connected' to the camera."""
    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently 'acquiring' video frames."""
    @property
    def fps(self) -> float:
        """Returns the frames per second (fps) setting of the camera."""
    @property
    def width(self) -> float:
        """Returns the frame width setting of the camera (in pixels)."""
    @property
    def height(self) -> float:
        """Returns the frame height setting of the camera (in pixels)."""
    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique identifier code of the camera."""
    @property
    def frame_pool(self) -> tuple[NDArray[Any], ...]:
        """Returns the tuple that stores the frames that are pooled to produce images during grab_frame() runtime."""
    def grab_frame(self) -> NDArray[np.uint8]:
        """Grabs the first 'available' frame from the camera buffer and returns it to caller as a NumPy array object.

        This method has to be called repeatedly to acquire new frames from the camera. The method is written to largely
        simulate the behavior of the 'real' camera classes.

        Returns:
            A NumPy array with the outer dimensions matching the preset camera frame dimensions. Depending on whether
            the camera is simulating 'color' or 'monochrome' mode, the returned frames will either have one or three
            color channels.

        Raises:
            RuntimeError: If the method is called for a class not currently 'connected' to a camera.
        """
