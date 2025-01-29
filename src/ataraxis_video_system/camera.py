"""This module provides classes that interface with supported Camera backends and an enumeration that contains
supported backend codes.

The classes from this module function as a unified API that allows any other module to work with any supported
camera. They abstract away the necessary procedures to connect to the camera and continuously grab acquired frames.
All 'real' camera backends are written in C and are designed to efficiently integrate into the VideoSystem class.

The classes from this module are not meant to be instantiated or used directly. Instead, they should be created using
the 'create_camera()' method from the VideoSystem class.
"""

import os
from enum import Enum
from typing import Any
from pathlib import Path

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from harvesters.core import Harvester, ImageAcquirer  # type: ignore
from harvesters.util.pfnc import (  # type: ignore
    bgr_formats,
    rgb_formats,
    bgra_formats,
    rgba_formats,
    mono_location_formats,
)
from ataraxis_base_utilities import console


class CameraBackends(Enum):
    """Maps valid literal values used to specify Camera class backend when requesting it from the create_camera ()
    method of the VideoSystem class to programmatically callable variables.

    Use this enumeration instead of 'hardcoding' Camera backends where possible to automatically adjust to future API
    changes to this library.

    The backend determines the low-level functioning of the Camera class and is, therefore, very important for
    optimizing video acquisition. It is generally advised to use the 'harvesters' backend with any GeniCam camera
    and only use opencv as a 'fallback' for camera that do not support GeniCam standard.
    """

    HARVESTERS = "harvesters"
    """
    This is the preferred backend for all cameras that support the GeniCam standard. This includes most scientific and
    industrial machine-vision cameras. This backend is based on the 'harvesters' project and it works with any type of
    GeniCam camera (USB, Ethernet, PCIE). The binding is extremely efficient and can handle large volume of data at a 
    high framerate.
    """
    OPENCV = "opencv"
    """
    This is the 'fallback' backend that should be used with cameras that do not support the GeniCam standard. OpenCV is
    a widely used machine-vision library that offers a flexible camera interface and video-acquisition tools. That said,
    the publicly available OpenCV bindings differ in efficiency for different platforms and camera types and may require
    additional project-specific configuration to work optimally. 
    """
    MOCK = "mock"
    """
    This backend should not be used in production projects. It is used to optimize project testing by providing a 
    Camera class that is not limited by available hardware. This is primarily used to enable parallel testing of 
    VideoSystem methods without making them depend on having a working camera.
    """


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

    # A dictionary that maps backend codes returned by VideoCapture get() method to meaningful names.
    _backends: dict[str, float] = {
        "Any": cv2.CAP_ANY,
        "VFW / V4L (Platform Dependent)": cv2.CAP_VFW,
        "IEEE 1394 / DC 1394 / CMU 1394 / FIREWIRE": cv2.CAP_FIREWIRE,
        "QuickTime": cv2.CAP_QT,
        "Unicap": cv2.CAP_UNICAP,
        "DirectShow": cv2.CAP_DSHOW,
        "PvAPI, Prosilica GigE SDK": cv2.CAP_PVAPI,
        "OpenNI (for Kinect)": cv2.CAP_OPENNI,
        "OpenNI (for Asus Xtion)": cv2.CAP_OPENNI_ASUS,
        "XIMEA Camera API": cv2.CAP_XIAPI,
        "AVFoundation framework iOS": cv2.CAP_AVFOUNDATION,
        "Smartek Giganetix GigEVisionSDK": cv2.CAP_GIGANETIX,
        "Microsoft Media Foundation": cv2.CAP_MSMF,
        "Microsoft Windows Runtime": cv2.CAP_WINRT,
        "Intel Perceptual Computing SDK": cv2.CAP_INTELPERC,
        "OpenNI2 (for Kinect)": cv2.CAP_OPENNI2,
        "OpenNI2 (for Asus Xtion and Occipital Structure sensors)": cv2.CAP_OPENNI2_ASUS,
        "gPhoto2 connection": cv2.CAP_GPHOTO2,
        "GStreamer": cv2.CAP_GSTREAMER,
        "FFMPEG library": cv2.CAP_FFMPEG,
        "OpenCV Image Sequence": cv2.CAP_IMAGES,
        "Aravis SDK": cv2.CAP_ARAVIS,
        "Built-in OpenCV MotionJPEG codec": cv2.CAP_OPENCV_MJPEG,
        "Intel MediaSDK": cv2.CAP_INTEL_MFX,
        "XINE engine (Linux)": cv2.CAP_XINE,
    }

    def __init__(
        self,
        camera_id: np.uint8,
        color: bool = True,
        backend: int = cv2.CAP_ANY,
        camera_index: int = 0,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._id: np.uint8 = camera_id
        self._color: bool = color
        self._backend: int = backend
        self._camera_index: int = camera_index
        self._camera: cv2.VideoCapture | None = None
        self._fps: float | None = fps
        self._width: int | None = width
        self._height: int | None = height
        self._acquiring: bool = False

    def __del__(self) -> None:
        """Ensures that the camera is disconnected upon garbage collection."""
        self.disconnect()

    def __repr__(self) -> str:
        """Returns a string representation of the OpenCVCamera object."""
        representation_string = (
            f"OpenCVCamera(camera_id={self._id}, camera_index={self._camera_index}, fps={self.fps}, "
            f"width={self.width}, height={self.height}, connected={self._camera is not None}, "
            f"acquiring={self._acquiring}, backend = {self.backend})"
        )
        return representation_string

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
        # Only attempts connection if the camera is not already connected
        if self._camera is None:
            # Generates an OpenCV VideoCapture object to acquire images from the camera. Uses the specified backend and
            # camera ID index.
            self._camera = cv2.VideoCapture(index=self._camera_index, apiPreference=int(self._backend))

            # Writes image acquisition parameters to the camera via the object generated above. If any of the
            # acquisition parameters were not provided, skips setting them and instead retrieves them from the
            # connected camera (see below).
            if self._fps is not None:
                self._camera.set(cv2.CAP_PROP_FPS, self._fps)  # pragma: no cover
            if self._width is not None:
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))  # pragma: no cover
            if self._height is not None:
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))  # pragma: no cover

            # Overwrites class attributes with the current properties of the camera. They may differ from the expected
            # result of setting the properties above!
            self._fps = self._camera.get(cv2.CAP_PROP_FPS)
            self._width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._backend = int(self._camera.get(cv2.CAP_PROP_BACKEND))

    def disconnect(self) -> None:
        """Disconnects from the camera by releasing the VideoCapture object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during the VideoSystem shutdown procedure to properly release
        resources.
        """
        # If the camera is already disconnected, returns without doing anything.
        if self._camera is not None:
            self._camera.release()
            self._acquiring = False  # Released camera automatically stops acquiring images
            self._camera = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the camera via a VideoCapture instance."""
        return self._camera is not None

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames.

        This concerns the 'asynchronous' behavior of the wrapped camera object which, after grab_frames() class method
        has been called, continuously acquires and buffers images even if they are not retrieved.
        """
        return self._acquiring

    @property
    def fps(self) -> float | None:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
        return self._fps

    @property
    def width(self) -> int | None:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
        return self._width

    @property
    def height(self) -> int | None:
        """Returns the current frame height setting of the camera (in pixels).

        If the camera is connected, this is the actual frame height value the camera is set to produce. If the camera
        is not connected, this is the desired frame height value that will be passed to the camera during connection.
        """
        return self._height

    @property
    def backend(self) -> str:
        """Returns the descriptive string-name for the backend being used by the connected VideoCapture object.

        If the camera is connected, this is the actual backend used to interface with the camera. If the camera
        is not connected, this is the desired backend that will be used to initialize the VideoCapture object.

        Raises:
            ValueError: If the backend code used to retrieve the backend name is not one of the recognized backend
                codes.
        """
        backend_code = self._backend

        for name, code in self._backends.items():
            if code == backend_code:
                return name

        message = (
            f"Unknown backend code {backend_code} encountered when retrieving the backend name used by the "
            f"OpenCV-managed camera with id {self._id}. Recognized backend codes are: "
            f"{(self._backends.values())}"
        )
        console.error(message=message, error=ValueError)
        # Fallback to appease mypy, should not be reachable
        raise ValueError("Unknown backend code")  # pragma: no cover

    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique identifier code of the camera."""
        return self._id

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
        if self._camera:
            # If necessary, ensures that the 'acquisition' mode flag is True.
            if not self._acquiring:
                self._acquiring = True

            ret, frame = self._camera.read()
            if not ret:
                message = (
                    f"The OpenCV-managed camera with id {self._id} did not yield an image, "
                    f"which is not expected. This may indicate initialization or connectivity issues."
                )
                console.error(message=message, error=RuntimeError)

            if not self._color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert BGR to Monochrome if needed

            return frame
        message = (
            f"The OpenCV-managed camera with id {self._id} is not connected, and cannot "
            f"yield images. Call the connect() method of the class prior to calling the grab_frame() method."
        )
        console.error(message=message, error=RuntimeError)
        # Fallback to appease mypy, should not be reachable
        raise RuntimeError(message)  # pragma: no cover


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

    def __init__(
        self,
        camera_id: np.uint8,
        cti_path: Path,
        camera_index: int = 0,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._id: np.uint8 = camera_id
        self._camera_index: int = camera_index
        self._camera: ImageAcquirer | None = None
        self._fps: float | None = fps
        self._width: int | None = width
        self._height: int | None = height

        # Initializes the Harvester class to discover the list of available cameras.
        self._harvester = Harvester()
        self._harvester.add_file(file_path=str(cti_path))  # Adds the .cti file to the class
        self._harvester.update()  # Discovers compatible cameras using the input .cti file interface

    def __del__(self) -> None:
        """Ensures that the camera is disconnected upon garbage collection."""
        self.disconnect()  # Releases the camera object
        self._harvester.reset()  # Releases the Harvester class resources

    def __repr__(self) -> str:
        """Returns a string representation of the HarvestersCamera object."""
        representation_string = (
            f"HarvestersCamera(camera_id={self._id}, camera_index={self._camera_index}, fps={self.fps}, "
            f"width={self.width}, height={self.height}, connected={self._camera is not None}, "
            f"acquiring={self.is_acquiring})"
        )
        return representation_string

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
        # Only attempts connection if the camera is not already connected
        if self._camera is None:
            # Generates a Harvester ImageAcquirer camera interface object using the provided camera ID as the list_index
            # input.
            self._camera = self._harvester.create(search_key=self._camera_index)

            # Writes image acquisition parameters to the camera via the object generated above.
            if self._width is not None:
                self._camera.remote_device.node_map.Width.value = self._width
            if self._height is not None:
                self._camera.remote_device.node_map.Height.value = self._height
            # Since the newest version of Harvesters checks inputs for validity, the fps have to be set last, as
            # the maximum fps is affected by frame width and height
            if self._fps is not None:
                self._camera.remote_device.node_map.AcquisitionFrameRate.value = self._fps

            # Overwrites class attributes with the current properties of the camera. They may differ from the expected
            # result of setting the properties above!
            # noinspection PyProtectedMember
            self._fps = self._camera.remote_device.node_map.AcquisitionFrameRate.value
            # noinspection PyProtectedMember
            self._width = self._camera.remote_device.node_map.Width.value
            # noinspection PyProtectedMember
            self._height = self._camera.remote_device.node_map.Height.value

    def disconnect(self) -> None:
        """Disconnects from the camera by stopping image acquisition, clearing any unconsumed buffers, and releasing
        the ImageAcquirer object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during the VideoSystem shutdown procedure to properly release
        resources.
        """
        # If the camera is already disconnected, returns without doing anything.
        if self._camera is not None:
            self._camera.stop()  # Stops image acquisition

            # Discards any unconsumed buffers to ensure proper memory release
            while self._camera.num_holding_filled_buffers != 0:
                _ = self._camera.fetch()  # pragma: no cover

            self._camera.destroy()  # Releases the camera object
            self._camera = None  # Sets the camera object to None

    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the camera via an ImageAcquirer instance."""
        return self._camera is not None

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently acquiring video frames.

        This concerns the 'asynchronous' behavior of the wrapped camera object which, after grab_frames() class method
        has been called, continuously acquires and buffers images even if they are not retrieved.
        """
        if self._camera is not None:
            return bool(self._camera.is_acquiring())
        return False  # If the camera is not connected, it cannot be acquiring images.

    @property
    def fps(self) -> float | None:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
        return self._fps

    @property
    def width(self) -> int | None:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
        return self._width

    @property
    def height(self) -> int | None:
        """Returns the current frame height setting of the camera (in pixels).

        If the camera is connected, this is the actual frame height value the camera is set to produce. If the camera
        is not connected, this is the desired frame height value that will be passed to the camera during connection.
        """
        return self._height

    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique byte identifier of the camera."""
        return self._id

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
        if not self._camera:
            message = (
                f"The Harvesters-managed camera with id {self._id} is not connected and cannot "
                f"yield images. Call the connect() method of the class prior to calling the grab_frame() method."
            )
            console.error(message=message, error=RuntimeError)
            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)  # pragma: no cover

        # If necessary, initializes image acquisition
        if not self._camera.is_acquiring():
            self._camera.start()

        # Retrieves the next available image buffer from the camera. Uses the 'with' context to properly
        # re-queue the buffer to acquire further images.
        with self._camera.fetch() as buffer:
            if buffer is None:  # pragma: no cover
                message = (
                    f"The Harvesters-managed camera with id {self._id} did not yield an image, "
                    f"which is not expected. This may indicate initialization or connectivity issues."
                )
                console.error(message=message, error=RuntimeError)

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
                # so we need to force numpy to copy the data instead of using the default referencing behavior.
                out_array: NDArray[Any] = content.data.reshape(height, width).copy()
                return out_array

            # For color data, evaluates the input format and reshapes the data as necessary.
            # This is excluded from coverage as we do not have a color-capable camera to test this right now
            if (
                data_format in rgb_formats
                or data_format in rgba_formats
                or data_format in bgr_formats
                or data_format in bgra_formats
            ):  # pragma: no cover
                # Reshapes the data into RGB + A format as the first processing step.
                content.data.reshape(
                    height,
                    width,
                    int(content.num_components_per_pixel),  # Sets of R, G, B, and Alpha
                )

                # Swaps every R and B value (RGB → BGR) ot produce BGR / BGRA images. This ensures consistency
                # with our OpenCVCamera API. Uses copy, which is VERY important. Once the buffer is released,
                # the original 'content' is lost, so we need to force numpy to copy the data instead of using
                # the default referencing behavior.
                frame: NDArray[Any] = content[:, :, ::-1].copy()

                # Returns the reshaped frame array to caller
                return frame

            # If the image has an unsupported data format, raises an error
            message = (
                f"The Harvesters-managed camera with id {self._id} yielded an image "
                f"with an unsupported data (color) format {data_format}. If possible, re-configure the "
                f"camera to use one of the supported formats: Monochrome, RGB, RGBA, BGR, BGRA. "
                f"Otherwise, you may need to implement a custom data reshaper algorithm."
            )  # pragma: no cover
            console.error(message=message, error=RuntimeError)  # pragma: no cover
            # This should never be reached, it is here to appease mypy
            raise RuntimeError(message)  # pragma: no cover


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

    def __init__(
        self,
        camera_id: np.uint8 = np.uint8(101),
        camera_index: int = 0,
        fps: float = 30,
        width: int = 600,
        height: int = 600,
        *,
        color: bool = True,
    ) -> None:
        # Saves class parameters to class attributes
        self._color: bool = color
        self._id: np.uint8 = camera_id
        self._camera_index: int = camera_index
        self._camera: bool = False
        self._fps: float = fps
        self._width: float = width
        self._height: float = height
        self._acquiring: bool = False

        # To allow reproducible testing, the class statically generates a pool of 10 images that is drawn from during
        # grab_frame() runtime. This allows simulating different fps values and verifying processed images against the
        # original image pull.
        frames_list: list[NDArray[Any]] = []
        for _ in range(10):
            if self._color:
                frame = np.random.randint(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Ensures the order of the colors is BGR
                frames_list.append(bgr_frame)
            else:
                # grayscale frames have only one channel, so order does not matter
                frames_list.append(np.random.randint(0, 256, size=(self._height, self._width, 1), dtype=np.uint8))

        # Casts to a tuple for efficiency reasons
        self._frames = tuple(frames_list)
        self._current_frame_index: int = 0

        self._timer: PrecisionTimer | None = None

        # Uses the fps to derive the number of microseconds that has to pass between each frame acquisition. This is
        # used to simulate real camera fps during grab_frame() runtime.
        self._time_between_frames: float = 1000 / self._fps

    def connect(self) -> None:
        """Simulates connecting to the camera, which is a necessary prerequisite to grab frames from the camera."""
        self._camera = True

        # Uses millisecond precision, which supports simulating up to 1000 fps. The time has to be initialized here to
        # make the class compatible with the VideoSystem, class (due to multiprocessing backend).
        self._timer = PrecisionTimer("ms")

    def disconnect(self) -> None:
        """Simulates disconnecting from the camera, which is part of the broader camera shutdown procedure."""
        self._camera = False
        self._acquiring = False
        self._timer = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the class is 'connected' to the camera."""
        return self._camera

    @property
    def is_acquiring(self) -> bool:
        """Returns True if the camera is currently 'acquiring' video frames."""
        return self._acquiring

    @property
    def fps(self) -> float:
        """Returns the frames per second (fps) setting of the camera."""
        return self._fps

    @property
    def width(self) -> float:
        """Returns the frame width setting of the camera (in pixels)."""
        return self._width

    @property
    def height(self) -> float:
        """Returns the frame height setting of the camera (in pixels)."""
        return self._height

    @property
    def camera_id(self) -> np.uint8:
        """Returns the unique identifier code of the camera."""
        return self._id

    @property
    def frame_pool(self) -> tuple[NDArray[Any], ...]:
        """Returns the tuple that stores the frames that are pooled to produce images during grab_frame() runtime."""
        return self._frames

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
        if not self._camera:
            message = (
                f"The Mocked camera with id {self._id} is not 'connected' and cannot yield images."
                f"Call the connect() method of the class prior to calling the grab_frame() method."
            )
            console.error(message=message, error=RuntimeError)
            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)  # pragma: no cover

        if not self._acquiring:
            self._acquiring = True

        # All our 'real' classes are designed to block in-place if the frame is not available. Here, this behavior
        # is simulated by using the timer class to 'force' the method to work at a certain FPS rate.
        while self._timer is not None and self._timer.elapsed < self._time_between_frames:
            pass

        # Acquires the next frame from the frame pool
        frame = self._frames[self._current_frame_index].copy()

        if self._timer is not None:
            self._timer.reset()  # Resets the timer to measure the time elapsed since the last frame acquisition.

        # Increments the flame pool index. Since the frame pool size is statically set to 10, the maximum retrieval
        # index is 9. Whenever the index reaches 9, it is reset back to 0 (to simulate circular buffer behavior).
        if self._current_frame_index == 9:
            self._current_frame_index = 0
        else:
            self._current_frame_index += 1

        # Returns the acquired frame to caller
        return frame
