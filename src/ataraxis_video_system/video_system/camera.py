from typing import Any, Optional

import cv2
from harvesters.core import Harvester, ImageAcquirer
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
        This class should not be initialized manually! Use the setup_camera() standalone function to create all camera
        instances.

        After frame-acquisition starts, some parameters, such as the fps or image dimensions, can no longer be altered.
        Commonly altered parameters have been made into initialization arguments to incentivize setting them to the
        desired values before starting frame-acquisition.

    Args:
        camera_id: The numeric ID of the camera, relative to all available video devices, e.g.: 0 for the first
            available camera, 1 for teh second, etc.
        fps: The desired Frames Per Second to capture the frames at. Note, this depends on the hardware capabilities of
            the camera and is affected by multiple related parameters, such as image dimensions, camera buffer size and
            the communication interface.
        width: The desired width of the camera frames to acquire. This will be passed to the camera and will only be
            respected if the camera has the capacity to alter acquired frame resolution.
        height: Same as width, but specifies the desired height of the camera frames to acquire.
        backend: The integer-code for the backend to use for the connected VideoCapture object. Generally, it
            is advised not to change the default value of this argument unless you know what you are doing.

    Attributes:
        _camera_id: Stores the numeric camera ID, which is used during connect() method runtime.
        _camera: Stores the OpenCV VideoCapture object that interfaces with the camera.
        _fps: Stores the desired Frames Per Second to capture the frames at.
        _width: Stores the desired width of the camera frames to acquire.
        _height: Stores the desired height of the camera frames to acquire.
        _acquiring: Stores whether the camera is currently acquiring video frames. This is statically set to 'true'
            the first time grab_frames() is called, as it initializes the camera acquisition thread of the binding
            object. If this attribute is true, some parameters, such as the fps, can no longer be altered.
        _backend: Stores teh code for the backend to be used by the connected VideoCapture object.
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
        self, camera_id: int = 0, fps: float = 30, width: float = 600, height: float = 400, backend: int = cv2.CAP_ANY
    ) -> None:
        # No input checking here as it is assumed that the class is initialized via get_camera() function that performs
        # the necessary input filtering.

        # Saves class parameters to class attributes
        self._camera_id: int = camera_id
        self._camera: Optional[cv2.VideoCapture] = None
        self._fps: float = fps
        self._width: float = width
        self._height: float = height
        self._acquiring: bool = False
        self._backend: int = backend

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()

    def __repr__(self) -> str:
        """Returns a string representation of the OpenCVCamera object."""
        representation_string = (
            f"OpenCVCamera(camera_id={self._camera_id}, fps={self._fps}, width={self._width}, "
            f"height={self._height}, connected={self._camera is not None}, acquiring={self._acquiring})"
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

            # Writes image acquisition parameters to the camera via the object generated above.
            self._camera.set(cv2.CAP_PROP_FPS, self._fps)
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

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
                f"The OpenCV-managed camera with id {self._camera_id} did not yield an image, "
                f"which is not expected. This may indicate initialization or connectivity issues."
            )
            console.error(message=message, error=RuntimeError)
            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)  # pragma: no cover


class MockCamera:
    """Simulates (mocks) the API behavior and functionality of the OpenCVCamera and HarvestersCamera classes.

    This class is primarily used to test VideoSystem functionality without using a physical camera, which allows
    optimizing testing efficiency and speed. The class mimics the behavior of the 'real' camera classes, but does not
    establish a physical connection with any camera hardware. The class accepts and returns static values that fully
    mimic the 'real' API.

    Notes:
        This class should not be initialized manually! Use the setup_camera() standalone function to create all camera
        instances.

        The class uses NumPy to simulate image acquisition where possible, generating 'white noise' images initialized
        with random-generator-derived pixel values.

    Args:
        camera_id: camera id

    Attributes:
        camera_id: camera id
        specs: dictionary holding the specifications of the camera. This includes fps, frame_width, frame_height
        _vid: opencv video capture object.
    """

    def __init__(self, camera_id: int = 0) -> None:
        self.specs = {"fps": 30.0, "frame_width": 640.0, "frame_height": 480.0}
        self.camera_id = camera_id
        self._vid = None

    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
        self._vid = True

    def disconnect(self) -> None:
        """Disconnects from camera."""
        self._vid = None

    @property
    def is_connected(self) -> bool:
        """Whether the camera is connected."""
        return self._vid is not None

    def grab_frame(self):
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected or did not yield an image.

        """
        if self._vid:
            return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        else:
            raise Exception("camera not connected")


class HarvestersCamera:
    pass


#     """A wrapper class for an opencv VideoCapture object.
#
#     Args:
#         camera_id: camera id
#
#     Attributes:
#         _camera_id: camera id
#         _specs: dictionary holding the specifications of the camera. This includes fps, frame_width, frame_height
#         _camera: opencv video capture object.
#     """
#
#     def __init__(self, camera_id: int = 0, cti_path: Optional[Path] = None) -> None:
#         self._specs: dict[str, Any] = {}
#         self._camera_id: int = camera_id
#         self._cti_path = cti_path
#         self._camera: Optional[ImageAcquirer] = None
#         h = Harvester()
#         grabber = h.create_image_acquirer()
#
#     def __del__(self) -> None:
#         """Ensures that camera is disconnected upon garbage collection."""
#         self.disconnect()
#
#     def connect(self) -> None:
#         """Connects to camera and prepares for image collection."""
#         harvester = Harvester()
#         if self._cti_path is not None:
#             try:
#                 # Imports general interface cti file to enable harvester to find and work with the camera
#                 harvester.add_cti_file(file_path=str(self._cti_path))
#             except FileNotFoundError:
#                 pass
#             except OSError:
#                 pass
#
#             # Updates the device list to discover cameras to grab data from
#             harvester.update_device_info_list()
#
#             # Generates an ImageAcquirer object to acquire images from the camera and saves it to class attribute
#             self._camera = harvester.create_image_acquirer(list_index=self._camera_id)


def get_camera(
    camera_id: int = 0, backend: Backends = Backends.OPENCV, cti_path: Optional[Path] = None
) -> OpenCVCamera | HarvestersCamera | MockCamera:
    if backend == Backends.OPENCV:
        return OpenCVCamera(camera_id=camera_id)
