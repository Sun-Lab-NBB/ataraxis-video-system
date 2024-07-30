from typing import Any, Optional

import cv2
from harvesters.core import Harvester, ImageAcquirer
from harvesters.util.pfnc import mono_location_formats, rgb_formats, rgba_formats, bgr_formats, bgra_formats
import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console
from enum import Enum
from pathlib import Path


class Backends(Enum):
    """Maps valid literal values used to specify Camera class backend when requesting it from get_camera() function to
    programmatically callable variables.

    Use this enumeration instead of 'hardcoding' Camera backends where possible to automatically adjust to future API
    changes to this library.

    The backend determines the low-level functioning of the Camera class and is, therefore, very important for
    optimizing video acquisition. It is generally advised to use the 'harvesters' backend with any GeniCam camera
    and only use opencv as a 'fallback' for camera that do not support GeniCam standard.
    """

    HARVESTERS: str = "harvesters"
    """
    This is the preferred backend for all cameras that support the GeniCam standard. This includes most scientific and
    industrial machine-vision cameras. This backend is based on the 'harvesters' project and it works with any type of
    GeniCam camera (USB, Ethernet, PCIE). The binding is extremely efficient and can handle large volume of data at a 
    high framerate.
    """
    OPENCV: str = "opencv"
    """
    This is the 'fallback' backend that should be used with cameras that do not support the GeniCam standard. OpenCV is
    a widely used machine-vision library that offers a flexible camera interface and video-acquisition tools. That said,
    the publicly available OpenCV bindings differ in efficiency for different platforms and camera types and may require
    additional project-specific configuration to work optimally. 
    """
    MOCK: str = "mock"
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
        This class should not be initialized manually! Use the create_camera() standalone function to create all camera
        instances.

        After frame-acquisition starts, some parameters, such as the fps or image dimensions, can no longer be altered.
        Commonly altered parameters have been made into initialization arguments to incentivize setting them to the
        desired values before starting frame-acquisition.

    Args:
        backend: The integer-code for the backend to use for the connected VideoCapture object. Generally, it
            is advised not to change the default value of this argument unless you know what you are doing.
        camera_id: The numeric ID of the camera, relative to all available video devices, e.g.: 0 for the first
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
        _backend: Stores the code for the backend to be used by the connected VideoCapture object.
        _camera_id: Stores the numeric camera ID, which is used during connect() method runtime.
        _camera: Stores the OpenCV VideoCapture object that interfaces with the camera.
        _fps: Stores the desired Frames Per Second to capture the frames at.
        _width: Stores the desired width of the camera frames to acquire.
        _height: Stores the desired height of the camera frames to acquire.
        _acquiring: Stores whether the camera is currently acquiring video frames. This is statically set to 'true'
            the first time grab_frames() is called, as it initializes the camera acquisition thread of the binding
            object. If this attribute is true, some parameters, such as the fps, can no longer be altered.
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
        backend: int = cv2.CAP_ANY,
        camera_id: int = 0,
        fps: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._backend: int = backend
        self._camera_id: int = camera_id
        self._camera: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = fps
        self._width: Optional[float] = width
        self._height: Optional[float] = height
        self._acquiring: bool = False

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()

    def __repr__(self) -> str:
        """Returns a string representation of the OpenCVCamera object."""
        representation_string = (
            f"OpenCVCamera(camera_id={self._camera_id}, fps={self.fps}, width={self.width}, "
            f"height={self.height}, connected={self._camera is not None}, acquiring={self._acquiring}, "
            f"backend = {self.backend})"
        )
        return representation_string

    def connect(self) -> None:
        """Initializes the camera VideoCapture object and sets the video acquisition parameters.

        This method has to be called prior to calling grab_frames() method. It is used to initialize and prepare the
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
            self._camera = cv2.VideoCapture(index=self._camera_id, apiPreference=self._backend)

            # Writes image acquisition parameters to the camera via the object generated above. If any of the
            # acquisition parameters were not provided, skips setting them and instead retrieves them from the
            # connected camera (see below).
            if self._fps is not None:
                self._camera.set(cv2.CAP_PROP_FPS, self._fps)
            if self._width is not None:
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, round(self._width))
            if self._height is not None:
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, round(self._height))

            # Overwrites class attributes with the current properties of the camera. They may differ from the expected
            # result of setting the properties above!
            self._fps = self._camera.get(cv2.CAP_PROP_FPS)
            self._width = self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            self._height = self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self._backend = self._camera.get(cv2.CAP_PROP_BACKEND)

    def disconnect(self) -> None:
        """Disconnects from the camera by releasing the VideoCapture object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during VideoSystem shutdown procedure to properly release
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
    def fps(self) -> float:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
        return self._fps

    @property
    def width(self) -> float:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
        return self._width

    @property
    def height(self) -> float:
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
            f"OpenCV-managed camera with id {self._camera_id}. Recognized backend codes are: "
            f"{(self._backends.values())}"
        )
        console.error(message=message, error=ValueError)
        # Fallback to appease mypy, should not be reachable
        raise ValueError("Unknown backend code")  # pragma: no cover

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
            delay up to a few seconds. Therefore, it is advised to call this method ahead of time and either discard
            the first few frames or have some other form of separating initial frames from the frames extracted as
            part of the post-initialization runtime.

            Moreover, it is advised to design video acquisition runtimes around repeatedly calling this method for
            the entire runtime duration to steadily consume the buffered images. This is in contrast to having multiple
            image acquisition 'pulses', which may incur additional overhead.

        Returns:
            A NumPy array with the outer dimensions matching the preset camera frame dimensions. All returned frames
            use the BGR colorspace by default and will, therefore, include 3 additional color channel dimensions for
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
                    f"The OpenCV-managed camera with id {self._camera_id} did not yield an image, "
                    f"which is not expected. This may indicate initialization or connectivity issues."
                )
                console.error(message=message, error=RuntimeError)
            return frame
        else:
            message = (
                f"The OpenCV-managed camera with id {self._camera_id} is not connected and cannot yield images."
                f"Call the connect() method of the class prior to calling the grab_frame() method."
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
        This class should not be initialized manually! Use the create_camera() standalone function to create all camera
        instances.

        After frame-acquisition starts, some parameters, such as the fps or image dimensions, can no longer be altered.
        Commonly altered parameters have been made into initialization arguments to incentivize setting them to the
        desired values before starting frame-acquisition.

    Args:
        cti_path: The path to the '.cti' file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire, would
            work as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.
        camera_id: The numeric ID of the camera, relative to all available video devices, e.g.: 0 for the first
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
        _camera_id: Stores the numeric camera ID, which is used during connect() method runtime.
        _camera: Stores the Harvesters ImageAcquirer object that interfaces with the camera.
        _harvester: Stores the Harvester interface object that discovers and manages the list of accessible cameras.
        _fps: Stores the desired Frames Per Second to capture the frames at.
        _width: Stores the desired width of the camera frames to acquire.
        _height: Stores the desired height of the camera frames to acquire.
    """

    def __init__(
        self,
        cti_path: Path,
        camera_id: int = 0,
        fps: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._camera_id: int = camera_id
        self._camera: Optional[ImageAcquirer] = None
        self._fps: Optional[float] = fps
        self._width: Optional[float] = width
        self._height: Optional[float] = height

        # Initializes the Harvester class to discover the list of available cameras.
        self._harvester = Harvester()
        self._harvester.add_cti_file(file_path=str(cti_path))  # Adds the .cti file to the class
        self._harvester.update_device_info_list()  # Discovers compatible cameras using the input .cti file interface

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()  # Releases the camera object
        self._harvester.reset()  # Releases the Harvester class resources

    def __repr__(self) -> str:
        """Returns a string representation of the HarvestersCamera object."""
        representation_string = (
            f"HarvestersCamera(camera_id={self._camera_id}, fps={self.fps}, width={self.width}, "
            f"height={self.height}, connected={self._camera is not None}, acquiring={self.is_acquiring})"
        )
        return representation_string

    def connect(self) -> None:
        """Initializes the camera ImageAcquirer object and sets the video acquisition parameters.

        This method has to be called prior to calling grab_frames() method. It is used to initialize and prepare the
        camera for image collection. Note, the method does not automatically start acquiring images. Image acquisition
        starts with the first call to grab_frames() method to make the API consistent across all our camera classes.

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
            self._camera = self._harvester.create_image_acquirer(list_index=self._camera_id)

            # Writes image acquisition parameters to the camera via the object generated above.
            if self._fps is not None:
                # noinspection PyProtectedMember
                self._camera._device.node_map.AcquisitionFrameRate.value = self._fps
            if self._width is not None:
                # noinspection PyProtectedMember
                self._camera._device.node_map.Width.value = int(self._width)
            if self._height is not None:
                # noinspection PyProtectedMember
                self._camera._device.node_map.Height.value = int(self._height)

            # Overwrites class attributes with the current properties of the camera. They may differ from the expected
            # result of setting the properties above!
            # noinspection PyProtectedMember
            self._fps = self._camera._device.node_map.AcquisitionFrameRate.value
            # noinspection PyProtectedMember
            self._width = self._camera._device.node_map.Width.value
            # noinspection PyProtectedMember
            self._height = self._camera._device.node_map.Height.value

    def disconnect(self) -> None:
        """Disconnects from the camera by stopping image acquisition, clearing any unconsumed buffers, and releasing
        the ImageAcquirer object.

        After calling this method, it will be impossible to grab new frames until the camera is (re)connected to via the
        connect() method. Make sure this method is called during VideoSystem shutdown procedure to properly release
        resources.
        """

        # If the camera is already disconnected, returns without doing anything.
        if self._camera is not None:
            self._camera.stop_image_acquisition()  # Stops image acquisition

            # Discards any unconsumed buffers to ensure proper memory release
            while self._camera.num_holding_filled_buffers != 0:
                _ = self._camera.fetch_buffer()

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
            return self._camera.is_acquiring_images()
        else:
            return False  # If the camera is not connected, it cannot be acquiring images.

    @property
    def fps(self) -> float:
        """Returns the current frames per second (fps) setting of the camera.

        If the camera is connected, this is the actual fps value the camera is set to produce. If the camera is not
        connected, this is the desired fps value that will be passed to the camera during connection.
        """
        return self._fps

    @property
    def width(self) -> float:
        """Returns the current frame width setting of the camera (in pixels).

        If the camera is connected, this is the actual frame width value the camera is set to produce. If the camera
        is not connected, this is the desired frame width value that will be passed to the camera during connection.
        """
        return self._width

    @property
    def height(self) -> float:
        """Returns the current frame height setting of the camera (in pixels).

        If the camera is connected, this is the actual frame height value the camera is set to produce. If the camera
        is not connected, this is the desired frame height value that will be passed to the camera during connection.
        """
        return self._height

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
            delay up to a few seconds. Therefore, it is advised to call this method ahead of time and either discard
            the first few frames or have some other form of separating initial frames from the frames extracted as
            part of the post-initialization runtime.

            Moreover, it is advised to design video acquisition runtimes around repeatedly calling this method for
            the entire runtime duration to steadily consume the buffered images. This is in contrast to having multiple
            image acquisition 'pulses', which may incur additional overhead.

        Returns:
            A NumPy array with the outer dimensions matching the preset camera frame dimensions. The returned frames
            will use one of the default GenTL output formats: Monochrome, RGB, RGBA, or BGR. This means that the
            returned array will use between 1 and 4 pixel-color channels for each 2-dimensional pixel position. The
            specific format depends on the format used by the camera.

        Raises:
            RuntimeError: If the camera does not yield an image, or if the method is called for a class not currently
                connected to a camera.
        """
        if self._camera:
            # If necessary, initializes image acquisition
            if not self._camera.is_acquiring_images():
                self._camera.start_image_acquisition()

            # Retrieves the next available image buffer from the camera. Uses the 'with' context to properly
            # re-queue the buffer to acquire further images.
            with self._camera.fetch_buffer() as buffer:
                if buffer is None:
                    message = (
                        f"The Harvesters-managed camera with id {self._camera_id} did not yield an image, "
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
                    return content.data.reshape(height, width)
                else:
                    # For color data, evaluates the input format and reshapes the data as necessary
                    if (
                        data_format in rgb_formats
                        or data_format in rgba_formats
                        or data_format in bgr_formats
                        or data_format in bgra_formats
                    ):
                        # Reshapes the data into RGB + A format as the first processing step.
                        frame = content.data.reshape(
                            height,
                            width,
                            int(content.num_components_per_pixel),  # Sets of R, G, B, and Alpha
                        )

                        # For BGR formats, swaps every R and B value (RGB -> BGR):
                        if data_format in bgr_formats:
                            frame = content[:, :, ::-1]

                        # Returns the reshaped frame array to caller
                        return frame

                    # If the image ahs an unsupported data format, raises an error
                    else:
                        message = (
                            f"The Harvesters-managed camera with id {self._camera_id} yielded an image with an "
                            f"unsupported data (color) format {data_format}. If possible, re-configure the camera to "
                            f"use one of the supported formats: Monochrome, RGB, RGBA, BGR, BGRA. Otherwise, you may "
                            f"need to implement a custom data reshaper algorithm."
                        )
                        console.error(message=message, error=RuntimeError)
        else:
            message = (
                f"The Harvesters-managed camera with id {self._camera_id} is not connected and cannot yield images."
                f"Call the connect() method of the class prior to calling the grab_frame() method."
            )
            console.error(message=message, error=RuntimeError)
            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)  # pragma: no cover


class MockCamera:
    """Simulates (mocks) the API behavior and functionality of the OpenCVCamera and HarvestersCamera classes.

    This class is primarily used to test VideoSystem class functionality without using a physical camera, which
    optimizes testing efficiency and speed. The class mimics the behavior of the 'real' camera classes, but does not
    establish a physical connection with any camera hardware. The class accepts and returns static values that fully
    mimic the 'real' API.

    Notes:
        This class should not be initialized manually! Use the create_camera() standalone function to create all camera
        instances.

        The class uses NumPy to simulate image acquisition where possible, generating 'white noise' images initialized
        with random-generator-derived pixel values.

    Args:
        camera_id: camera id

    Attributes:
        camera_id: camera id
        _vid: opencv video capture object.
    """

    def __init__(
        self,
        camera_id: int = 0,
        fps: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        *,
        color: bool = True,
    ) -> None:

        # Saves class parameters to class attributes
        self._color = color
        self._camera_id: int = camera_id
        self._camera: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = fps
        self._width: Optional[float] = width
        self._height: Optional[float] = height
        self._acquiring: bool = False

    def connect(self) -> None:
        """Simulates connecting to the camera, which is a necessary prerequisite to grab frames from the camera."""
        self._camera = True

    def disconnect(self) -> None:
        """Simulates disconnecting from the camera, which is part of the broader camera shutdown procedure."""
        self._camera = None
        self._acquiring = False

    @property
    def is_connected(self) -> bool:
        """Returns True if the class is 'connected' to the camera."""
        return self._camera is not None

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
        """Returns the frame height setting of the camera (in pixels).
        """
        return self._height

    def grab_frame(self):
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected or did not yield an image.

        """
        if self._camera:
            if self._color:
                return np.random.randint(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
            else:
                return np.random.randint(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
        else:
            message = (
                f"The Mocked camera with id {self._camera_id} is not 'connected' and cannot yield images."
                f"Call the connect() method of the class prior to calling the grab_frame() method."
            )
            console.error(message=message, error=RuntimeError)
            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)  # pragma: no cover


def get_opencv_ids() -> tuple[str, ...]:
    """Discovers and reports IDs and descriptive information about cameras accessible through the OpenCV library.

    This function can be used to discover camera IDs accessible through our OpenCVCamera class. Subsequently,
    each of the IDs can be passed to the create_camera() function to create an OpenCVCamera class instance to
    interface with the camera. For each working camera, the function produces a string that includes camera ID, image
    width, height, and the fps value to help identifying the cameras.

    Notes:
        Currently, there is no way to get serial numbers or usb port names from OpenCV. Therefore, while this function
        tries to provide some ID information, it likely will not be enough to identify the cameras. Instead, it is
        advised to use the interactive imaging mode with each of the IDs to manually map IDs to cameras based on the
        produced visual stream.

        This function works by sequentially evaluating camera IDs starting from 0 and up to ID 100. The function
        connects to each camera and takes a test image to ensure the camera is accessible, and it should ONLY be called
        when no OpenCVCamera or any other OpenCV-based connection is active. The evaluation sequence will stop
        early if it encounters more than 5 non-functional IDs in a row.

        This function will yield errors from OpenCV which are unfortunately not circumventable at this time. That said,
        since the function is not designed to be used in well-configured production runtimes, this should not be a major
        concern.

    Returns:
         A tuple of strings. Each string contains camera ID, frame width, frame height, and camera fps value.
    """
    non_working_count = 0
    working_ids = []

    # This loop will keep iterating over IDs until it discovers 5 non-working IDs. The loop is designed to evaluate 100
    # IDs at maximum to prevent infinite execution.
    for evaluated_id in range(100):
        # Evaluates each ID by instantiating a video-capture object and reading one image and dimension data from
        # the connected camera (if any was connected).
        camera = cv2.VideoCapture(evaluated_id)

        # If the evaluated camera can be connected and returns images, it's ID is appended to the ID list
        if camera.isOpened() and camera.read()[0]:
            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camera.get(cv2.CAP_PROP_FPS)
            descriptive_string = f"OpenCV Camera ID: {evaluated_id}, Width: {width}, Height: {height}, FPS: {fps}."
            working_ids.append(descriptive_string)
            non_working_count = 0  # Resets non-working count whenever a working camera is found.
        else:
            non_working_count += 1

        # Breaks the loop early if more than 5 non-working IDs are found consecutively
        if non_working_count >= 5:
            break

        camera.release()  # Releases the camera object to recreate it above for the next cycle

    return tuple(working_ids)  # Converts to tuple before returning to caller.


def get_harvesters_ids(cti_path: Path) -> tuple[str, ...]:
    """Discovers and reports IDs and descriptive information about cameras accessible through the Harvesters library.

    Since Harvesters already supports listing valid IDs available through a given .cti interface, this function wraps
    instantiating and using built-in Harvesters functionality to discover and return ID and descriptive information
    about cameras available to the local system. The discovered IDs can later be used with the create_camera()
    function to create HarvestersCamera class for each camera ID that needs to be interfaced with during runtime.

    Notes:
        This function bundles discovered ID (list index) information with the serial number and the camera model to aid
        identifying physical cameras for each ID.

    Args:
        cti_path: The path to the '.cti' file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire, would
            work as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.

    Returns:
        A tuple of strings. Each string contains camera ID, serial number, and model name.
    """

    # Instantiates the class and adds the input .cti file.
    harvester = Harvester()
    harvester.add_cti_file(file_path=str(cti_path))

    # Gets the list of accessible cameras
    harvester.update_device_info_list()

    # Loops over all discovered cameras and parses basic ID information from each camera to generate a descriptive
    # string.
    working_ids = []
    for num, camera_info in enumerate(harvester.device_info_list):
        descriptive_string = (
            f"Harvesters Camera ID: {num}, Serial Number: {camera_info.serial_number}, Model Name: {camera_info.model}."
        )
        working_ids.append(descriptive_string)

    return tuple(working_ids)  # Converts to tuple before returning to caller.


def get_camera(
    camera_id: int = 0, backend: Backends = Backends.OPENCV, cti_path: Optional[Path] = None
) -> OpenCVCamera | HarvestersCamera | MockCamera:
    if backend == Backends.OPENCV:
        return OpenCVCamera(camera_id=camera_id)
