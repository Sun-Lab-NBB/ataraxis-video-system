"""This module contains the main VideoSystem class that contains methods for setting up, running, and tearing down
interactions between Camera and Saver classes.

While Camera and Saver classes provide convenient interface for cameras and saver backends, VideoSystem connects
cameras to savers and manages the flow of frames between them. Each VideoSystem is a self-contained entity that
provides a simple API for recording data from a wide range of cameras as images or videos. The class is written in a
way that maximizes runtime performance.

All user-oriented functionality of this library is available through the public methods of the VideoSystem class.
"""

from queue import Queue
from types import NoneType
from typing import Optional
from pathlib import Path
from threading import Thread
import subprocess
from dataclasses import dataclass
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
    ProcessError,
)
from multiprocessing.managers import SyncManager

import cv2
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from harvesters.core import Harvester  # type: ignore
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import LogPackage, SharedMemoryArray
from ataraxis_time.time_helpers import convert_time, get_timestamp

from .saver import (
    ImageSaver,
    VideoSaver,
    VideoCodecs,
    ImageFormats,
    VideoFormats,
    CPUEncoderPresets,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from .camera import MockCamera, OpenCVCamera, CameraBackends, HarvestersCamera


@dataclass(frozen=True)
class _CameraSystem:
    """Stores a Camera class instance managed by the VideoSystem class, alongside additional runtime parameters.

    This class is used as a container that aggregates all objects and parameters required by the VideoSystem to
    interface with a camera during runtime.
    """

    """Stores the managed camera interface class."""
    camera: OpenCVCamera | HarvestersCamera | MockCamera

    """Determines whether to override (sub-sample) the camera's frame acquisition rate. The override has to be smaller 
    than the camera's native frame rate and can be used to more precisely control the frame rate of cameras that do 
    not support real-time frame rate control."""
    fps_override: int | float

    """Determines whether acquired frames need to be piped to other processes via the output queue, in addition to 
    being sent to the saver process (if any). This is used to additionally process the frames in-parallel with 
    saving them to disk, for example, to analyze the visual stream data for certain trigger events."""
    output_frames: bool

    """Allows adjusting the frame rate at which frames are sent to the output_queue. Frequently, real time processing
    would impose constraints on the input frame rate that will not match the frame acquisition (and saving) frame 
    rate. This setting allows sub-sampling the saved frames to a frame rate required by the output processing 
    pipeline."""
    output_frame_rate: int | float

    """Determines whether to display acquired camera frames to the user via a display UI. The frames are always 
    displayed at a 30 fps rate regardless of the actual frame rate of the camera. Note, this does not interfere with 
    frame acquisition (saving)."""
    display_frames: bool

    """Same as output_frame_rate, but allows limiting the framerate at which the frames are displayed to the user if 
    displaying the frames is enabled. It is highly advise to not set this value above 30 fps."""
    display_frame_rate: int | float


@dataclass(frozen=True)
class _SaverSystem:
    """Stores a Saver class instance managed by the VideoSystem class, alongside additional runtime parameters.

    This class is used as a container that aggregates all objects and parameters required by the VideoSystem to
    interface with a saver during runtime.
    """

    """Stores the managed saver interface class."""
    saver: ImageSaver | VideoSaver

    """Stores the indices of camera objects whose frames will be saved by the included saver instance. The indices 
    have to match the camera object indices inside the _cameras attribute of the VideoSystem instance that manages 
    this SaverSystem."""
    source_ids: tuple[int]


class VideoSystem:
    """Efficiently combines Camera and Saver instances to acquire, display, and save camera frames in real time.

    This class controls the runtime of Camera and Saver instances running on independent cores (processes) to maximize
    the frame throughout. The class achieves two main objectives: It efficiently moves frames acquired by camera(s)
    to the saver(s) that write them to disk and manages runtime behavior of the managed classes. To do so, the class
    initializes and controls multiple subprocesses to ensure frame producer and consumer classes have sufficient
    computational resources. The class abstracts away all necessary steps to set up, execute, and tear down the
    processes through an easy-to-use API.

    Notes:
        Due to using multiprocessing to improve efficiency, this class would typically reserve 1 or 2 logical cores at
        a minimum to run efficiently. Additionally, due to a time-lag of moving frames from a producer process to a
        consumer process, the class will reserve a variable portion of the RAM to buffer the frame images. The reserved
        memory depends on many factors and can only be established empirically.

        The class does not initialize cameras or savers at instantiation. Instead, you have to manually use the
        add_camera and add_image_saver or add_video_saver methods to add cameras and savers to the system. All cameras
        and all savers added to a single VideoSystem will run on the same logical core (one for savers,
        one for cameras). To use more than two logical cores, create additional VideoSystem instances as necessary.

    Args:
        system_id: A unique byte-code identifier for the VideoSystem instance. This is used to identify the system in
            log files and generated video files. Therefore, this ID has to be unique across all concurrently
            active Ataraxis systems that use DataLogger to log data (such as AtaraxisMicroControllerInterface class).
        system_name: A human-readable name used to identify the VideoSystem instance in error messages.
        system_description: A brief description of the VideoSystem instance. This is used when creating the log file
            that stores class runtime parameters and maps id-codes to meaningful names and descriptions to support
            future data processing.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class via its 'input_queue' property.
            This queue is used to buffer and pipe data to be logged to the logger cores.
        output_directory: The path to the output directory which will be used by all Saver class instances to save
            acquired camera frames as images or videos. This argument is required if you intend to add Saver instances
            to this VideoSystem and optional if you do not intend to save frames grabbed by this VideoSystem.
        harvesters_cti_path: The path to the '.cti' file that provides the GenTL Producer interface. This argument is
            required if you intend to interface with cameras using 'Harvesters' backend! It is recommended to use the
            file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire, would work
            as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.

    Attributes:
        _id: Stores the ID code of the VideoSystem instance.
        _name: Stores the human-readable name of the VideoSystem instance.
        _description: Stores the description of the VideoSystem instance.
        _logger_queue: Stores the multiprocessing Queue object used to buffer and pipe data to be logged.
        _output_directory: Stores the path to the output directory.
        _cti_path: Stores the path to the '.cti' file that provides the GenTL Producer interface.
        _cameras: Stores managed CameraSystems.
        _savers: Stores managed SaverSystems.
        _started: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores a SyncManager class instance from which the image_queue and the log file Lock are derived.
        _image_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from the producer to
            the consumer process.
        _output_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from the producer to
            other concurrently active processes.
        _terminator_array: A SharedMemoryArray instance that provides a cross-process communication interface used to
            manage runtime behavior of spawned processes.
        _producer_process: A process that acquires camera frames using managed CameraSystems.
        _consumer_process: A process that saves the acquired frames using managed SaverSystems.
        _watchdog_thread: A thread used to monitor the runtime status of the remote consumer and producer processes.

    Raises:
        TypeError: If any of the provided arguments has an invalid type.
    """

    def __init__(
        self,
        system_id: np.uint8,
        system_name: str,
        system_description: str,
        logger_queue: MPQueue,  # type: ignore
        output_directory: Optional[Path] = None,
        harvesters_cti_path: Optional[Path] = None,
    ):
        # Resolves the system-name first, to use it in further error messages
        if not isinstance(system_name, str):
            message = (
                f"Unable to initialize the VideoSystem class instance. Expected a string for system_name, but got "
                f"{system_name} of type {type(system_name).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Ensures system_id is a byte-convertible integer
        if not isinstance(system_id, np.uint8):
            message = (
                f"Unable to initialize the {system_name} VideoSystem class instance. Expected a uint8 system_id, but "
                f"encountered {system_id} of type {type(system_id).__name__}."
            )
            console.error(message=message, error=TypeError)

        # If harvesters_cti_path is provided, checks if it is a valid file path.
        if (
            harvesters_cti_path is not None
            and not harvesters_cti_path.exists()
            and not harvesters_cti_path.is_file()
            and not harvesters_cti_path.suffix == ".cti"
        ):
            message = (
                f"Unable to initialize the {system_name} VideoSystem class instance. Expected the path to an existing "
                f"file with a '.cti' suffix or None for harvesters_cti_path, but encountered {harvesters_cti_path} of "
                f"type {type(harvesters_cti_path).__name__}. If a valid path was provided, this error is likely due to "
                f"the file not existing or not being accessible."
            )
            console.error(message=message, error=TypeError)

        # Ensures invalid descriptions are converted to an empty string.
        system_description = system_description if isinstance(system_description, str) else ""

        # Saves ID data and the logger queue to class attributes
        self._id: np.uint8 = system_id
        self._name: str = system_name
        self._description = system_description
        self._logger_queue: MPQueue = logger_queue  # type: ignore
        self._output_directory: Path | None = output_directory
        self._cti_path: Path | None = harvesters_cti_path

        # Initializes placeholder variables that will be filled by add_camera() and add_saver() methods.
        self._cameras: list[_CameraSystem] = []
        self._savers: list[_SaverSystem] = []

        self._started: bool = False  # Tracks whether the system has active processes

        # Sets up the assets used to manage acquisition and saver processes. The assets are configured during the
        # start() method runtime, most of them are initialized to placeholder values here.
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._image_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._output_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: SharedMemoryArray | None = None
        self._producer_process: Process | None = None
        self._consumer_process: Process | None = None
        self._watchdog_thread: Thread | None = None

        # If the output directory path is provided, ensures the directory tree exists
        if output_directory is not None:
            ensure_directory_exists(output_directory)

    def __del__(self) -> None:
        """Ensures that all resources are released when the instance is garbage-collected."""
        self.stop()

    def __repr__(self) -> str:
        """Returns a string representation of the VideoSystem class instance."""
        representation_string: str = (
            f"VideoSystem(name={self._name}, running={self._started}, camera_count={len(self._cameras)}, "
            f"saver_count={len(self._savers)})"
        )
        return representation_string

    def add_camera(
        self,
        camera_name: str,
        camera_id: int = 0,
        camera_backend: CameraBackends = CameraBackends.OPENCV,
        output_frames: bool = False,
        output_frame_rate: int | float = 25,
        display_frames: bool = False,
        display_frame_rate: int | float = 25,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        acquisition_frame_rate: Optional[int | float] = None,
        opencv_backend: Optional[int] = None,
        color: Optional[bool] = None,
    ) -> None:
        """Creates a Camera class instance and adds it to the list of cameras managed by the VideoSystem instance.

        This method allows adding Cameras to an initialized VideoSystem instance. Currently, this is the only intended
        way of using Camera classes available through this library. Unlike Saver class instances, which are not
        required for the VideoSystem to function, at least one valid Camera class must be added to the system before
        its start() method is called.

        Args:
            camera_name: The human-readable of the camera. This is used to help identify the camera to the user in
                messages and during real-time frame display.
            camera_id: The numeric ID of the camera, relative to all available video devices, e.g.: 0 for the first
                available camera, 1 for the second, etc. Usually, the camera IDs are assigned by the host-system based
                on the order of their connection.
            camera_backend: The backend to use for the camera class. Currently, all supported backends are derived from
                the CameraBackends enumeration. The preferred backend is 'Harvesters', but we also support OpenCV for
                non-GenTL-compatible cameras.
            output_frames: Determines whether to output acquired frames via the VideoSystem's output_queue. This
                allows real time processing of acquired frames by other concurrent processes.
            output_frame_rate: Allows adjusting the frame rate at which acquired frames are sent to the output_queue.
                Note, the output framerate cannot exceed the native frame rate of the camera, but it can be lower than
                the native acquisition frame rate. The acquisition frame rate is set via the frames_per_second argument
                below.
            display_frames: Determines whether to display acquired frames to the user. This allows visually monitoring
                the camera feed in real time.
            display_frame_rate: Similar to output_frame_rate, determines the frame rate at which acquired frames are
                displayed to the user.
            frame_width: The desired width of the camera frames to acquire, in pixels. This will be passed to the
                camera and will only be respected if the camera has the capacity to alter acquired frame resolution.
                If not provided (set to None), this parameter will be obtained from the connected camera.
            frame_height: Same as width, but specifies the desired height of the camera frames to acquire, in pixels.
                If not provided (set to None), this parameter will be obtained from the connected camera.
            acquisition_frame_rate: How many frames to capture each second (capture speed). Note, this depends on the
                hardware capabilities of the camera and is affected by multiple related parameters, such as image
                dimensions, camera buffer size, and the communication interface. If not provided (set to None), this
                parameter will be obtained from the connected camera. The VideoSystem always tries to set the camera
                hardware to record frames at the requested rate, but contains a fallback that allows down-sampling the
                acquisition rate in software. This fallback is only used when the camera uses a higher framerate than
                the requested value.
            opencv_backend: Optional. The integer-code for the specific acquisition backend (library) OpenCV should
                use to interface with the camera. Generally, it is advised not to change the default value of this
                argument unless you know what you are doing.
            color: A boolean indicating whether the camera acquires colored or monochrome images. This is
                used by OpenCVCamera to optimize acquired images depending on the source (camera) color space. It is
                also used by the MockCamera to enable simulating monochrome and colored images. This option is ignored
                for HarvesterCamera instances as it expects the colorspace to be configured via the camera's API.

        Raises:
            TypeError: If the input arguments are not of the correct type.
            ValueError: If the requested camera_backend is not one of the supported backends. If the chosen backend is
                Harvesters and the .cti path is not provided. If attempting to set camera framerate or frame dimensions
                fails for any reason.
        """
        # Verifies that the input arguments are of the correct type. Note, checks backend-specific arguments in
        # backend-specific clause.
        if not isinstance(camera_name, str):
            message = (
                f"Unable to add the Camera object to the {self._name} VideoSystem. Expected a string for camera_name "
                f"argument, but got {camera_name} of type {type(camera_name).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(camera_id, int):
            message = (
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected an integer "
                f"for camera_id argument, but got {camera_id} of type {type(camera_id).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(acquisition_frame_rate, (int, float, NoneType)):
            message = (
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected an integer, "
                f"float or None for acquisition_frame_rate argument, but got {acquisition_frame_rate} of type "
                f"{type(acquisition_frame_rate).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frame_width, (int, NoneType)):
            message = (
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected an integer "
                f"or None for frame_width argument, but got {frame_width} of type {type(frame_width).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frame_height, (int, NoneType)):
            message = (
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected an integer "
                f"or None for frame_height argument, but got {frame_height} of type {type(frame_height).__name__}."
            )
            raise console.error(error=TypeError, message=message)

        # Ensures that display_frames is boolean. Does the same for output_frames
        display_frames = False if not isinstance(display_frames, bool) else display_frames
        output_frames = False if not isinstance(output_frames, bool) else output_frames

        # Pre-initializes the fps override to 0. A 0 value indicates that the override is not used. It is enabled
        # automatically as a fallback when camera lacks native fps limiting capabilities.
        fps_override: int | float = 0

        # Converts integer frames_per_second inputs to floats, since the Camera classes expect it to be a float.
        if isinstance(acquisition_frame_rate, int):
            acquisition_frame_rate = float(acquisition_frame_rate)

        # OpenCVCamera
        if camera_backend == CameraBackends.OPENCV:
            # If backend preference is None, uses generic preference
            if opencv_backend is None:
                opencv_backend = int(cv2.CAP_ANY)
            # If the backend is not an integer or None, raises an error
            elif not isinstance(opencv_backend, int):
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Expected "
                    f"an integer or None for opencv_backend argument, but got {opencv_backend} of type "
                    f"{type(opencv_backend).__name__}."
                )
                raise console.error(error=TypeError, message=message)

            # Ensures that color is either True or False.
            image_color = False if not isinstance(color, bool) else color

            # Instantiates the OpenCVCamera class object
            camera = OpenCVCamera(
                name=camera_name,
                color=image_color,
                backend=opencv_backend,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=acquisition_frame_rate,
            )

            # Connects to the camera. This both verifies that the camera can be connected to and applies the
            # frame acquisition settings.
            camera.connect()

            # Grabs a test frame from the camera to verify frame acquisition capabilities.
            frame = camera.grab_frame()

            # Verifies the dimensions and the colorspace of the acquired frame
            if frame_height is not None and frame.shape[0] != frame_height:
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Attempted "
                    f"configuring the camera to acquire frames using the provided frame_height {frame_height}, but the "
                    f"camera returned a test frame with height {frame.shape[0]}. This indicates that the camera "
                    f"does not support the requested frame height and width combination."
                )
                raise console.error(error=ValueError, message=message)
            if frame_width is not None and frame.shape[1] != frame_width:
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Attempted "
                    f"configuring the camera to acquire frames using the provided frame_width {frame_width}, but the "
                    f"camera returned a test frame with width {frame.shape[1]}. This indicates that the camera "
                    f"does not support the requested frame height and width combination."
                )
                raise console.error(error=ValueError, message=message)
            if color and frame.shape[2] <= 1:
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Attempted "
                    f"configuring the camera to acquire colored frames, but the camera returned a test frame with "
                    f"monochrome colorspace. This indicates that the camera does not support acquiring colored frames."
                )
                raise console.error(error=ValueError, message=message)
            elif len(frame.shape) != 2:
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Attempted "
                    f"configuring the camera to acquire monochrome frames, but the camera returned a test frame with "
                    f"BGR colorspace. This likely indicates an OpenCV backend error, since it is unlikely that the "
                    f"camera does not support monochrome colorspace."
                )
                raise console.error(error=ValueError, message=message)

            # If the camera failed to set the requested frame rate, but it is possible to correct the fps via software,
            # enables fps override. Software correction requires that the native fps is higher than the desired fps,
            # as it relies on discarding excessive frames.
            if acquisition_frame_rate is not None and camera.fps > acquisition_frame_rate:
                fps_override = acquisition_frame_rate
            elif acquisition_frame_rate is not None and camera.fps < acquisition_frame_rate:
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Attempted "
                    f"configuring the camera to acquire frames at the rate of {acquisition_frame_rate} frames per "
                    f"second, but the camera automatically adjusted the framerate to {camera.fps}. This indicates that "
                    f"the camera does not support the requested framerate."
                )
                raise console.error(error=ValueError, message=message)

            # Disconnects from the camera to free the resources to be used by the remote producer process, once it is
            # instantiated.
            camera.disconnect()

        # HarvestersCamera
        elif camera_backend == CameraBackends.HARVESTERS:
            # Ensures that the cti_path is provided
            if self._cti_path is None:
                message = (
                    f"Unable to add a {camera_name} HarvestersCamera to the {self._name} VideoSystem. Expected the "
                    f"VideoSystem's cti_path attribute to be a Path object pointing to the '.cti' file, but got None "
                    f"instead. Make sure you provide a valid '.cti' file as harvesters_cit_file argument when "
                    f"initializing the VideoSystem instance."
                )
                console.error(error=ValueError, message=message)
                # Fallback to appease mypy, should not be reachable
                raise ValueError(message)  # pragma: no cover

            # Instantiates and returns the HarvestersCamera class object
            camera = HarvestersCamera(
                name=camera_name,
                cti_path=self._cti_path,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=acquisition_frame_rate,
            )

            # Connects to the camera. This both verifies that the camera can be connected to and applies the camera
            # settings.
            camera.connect()

            # This is only used to verify that the frame acquisition works as expected. Unlike OpenCV, Harvesters
            # raises errors if the camera does not support any of the input settings.
            camera.grab_frame()

            # Disconnects from the camera to free the resources to be used by the remote producer process, once it is
            # instantiated.
            camera.disconnect()

        # MockCamera
        elif camera_backend == CameraBackends.MOCK:
            # Ensures that mock_color is either True or False.
            mock_color = False if not isinstance(color, bool) else color

            # Unlike real cameras, MockCamera cannot retrieve fps and width / height from hardware memory.
            # Therefore, if either of these variables is None, it is initialized to hardcoded defaults
            if frame_height is None:
                frame_height = 400
            if frame_width is None:
                frame_width = 600
            if acquisition_frame_rate is None:
                acquisition_frame_rate = 30

            # Instantiates the MockCamera class object
            camera = MockCamera(
                name=camera_name,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=acquisition_frame_rate,
                color=mock_color,
            )

            # Since MockCamera is implemented in software only, there is no need to check for hardware-dependent
            # events after camera initialization (such as whether the camera is connectable and can be configured to
            # use a specific fps).

        # If the input backend does not match any of the supported backends, raises an error
        else:
            message = (
                f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
                f"camera_backend argument {camera_backend} of type {type(camera_backend).__name__}. "
                f"camera_backend has to be one of the options available from the CameraBackends enumeration."
            )
            raise console.error(error=ValueError, message=message)

        # If the output_frame_rate argument is not an integer or floating value, defaults to using the same framerate
        # as the camera. This has to be checked after the camera has been verified and its fps has been confirmed.
        if not isinstance(output_frame_rate, (int, float)) or not 0 <= output_frame_rate <= camera.fps:
            message = (
                f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
                f"output_frame_rate argument {output_frame_rate} of type {type(output_frame_rate).__name__}. "
                f"Output framerate override has to be an integer or floating point number that does not exceed the "
                f"camera acquisition framerate ({camera.fps})."
            )
            raise console.error(error=TypeError, message=message)

        # Same as above, but for display frame_rate
        if not isinstance(display_frame_rate, (int, float)) or not 0 <= output_frame_rate <= camera.fps:
            message = (
                f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
                f"display_frame_rate argument {display_frame_rate} of type {type(display_frame_rate).__name__}. "
                f"Display framerate override has to be an integer or floating point number that does not exceed the "
                f"camera acquisition framerate ({camera.fps})."
            )
            raise console.error(error=TypeError, message=message)

        # If the camera class was successfully instantiated, packages the class alongside additional parameters into a
        # CameraSystem object and appends it to the camera list.
        self._cameras.append(
            _CameraSystem(
                camera=camera,
                fps_override=fps_override,
                output_frames=output_frames,
                output_frame_rate=output_frame_rate,
                display_frames=display_frames,
                display_frame_rate=display_frame_rate,
            )
        )

    def add_image_saver(
        self,
        source_ids: tuple[int],
        image_format: ImageFormats = ImageFormats.TIFF,
        tiff_compression_strategy: int = cv2.IMWRITE_TIFF_COMPRESSION_LZW,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        png_compression: int = 1,
        thread_count: int = 5,
    ) -> None:
        """Creates an ImageSaver class instance and adds it to the list of savers managed by the VideoSystem instance.

        This method allows adding ImageSavers to an initialized VideoSystem instance. Currently, this is the only
        intended way of using ImageSaver classes available through this library. ImageSavers are not required for the
        VideoSystem to function properly and, therefore, this method does not need to be called unless you need to save
        the camera frames acquired during the runtime of this VideoSystem as images.

        Notes:
            This method is specifically designed to add ImageSavers. If you need to add a VideoSaver (to save frames as
            a video), use the add_video_saver() method instead.

            ImageSavers can reach the highest image saving speed at the cost of using considerably more disk space than
            efficient VideoSavers. Overall, it is highly recommended to use VideoSavers with hardware_encoding where
            possible to optimize the disk space usage and still benefit from a decent frame saving speed.

        Args:
            source_ids: The list of Camera object indices whose frames will be saved by this ImageSaver. The indices
                are based on the order the cameras were / will be added to the VideoSystem instance with index 0 being
                the first added camera. This argument is very important as it directly determines what frames are saved
                and by what savers. If you do not need to save any frames, do not add ImageSaver or VideoSaver instances
                at all.
            image_format: The format to use for the output images. Use ImageFormats enumeration
                to specify the desired image format. Currently, only 'TIFF', 'JPG', and 'PNG' are supported.
            tiff_compression_strategy: The integer-code that specifies the compression strategy used for .tiff image
                files. Has to be one of the OpenCV 'IMWRITE_TIFF_COMPRESSION_*' constants. It is recommended to use
                code 1 (None) for lossless and fastest file saving or code 5 (LZW) for a good speed-to-compression
                balance.
            jpeg_quality: An integer value between 0 and 100 that controls the 'loss' of the JPEG compression. A higher
                value means better quality, less data loss, bigger file size, and slower processing time.
            jpeg_sampling_factor: An integer-code that specifies how JPEG encoder samples image color-space. Has to be
                one of the OpenCV 'IMWRITE_JPEG_SAMPLING_FACTOR_*' constants. It is recommended to use code 444 to
                preserve the full color-space of the image if your application requires this. Another popular option is
                422, which results in better compression at the cost of color coding precision.
            png_compression: An integer value between 0 and 9 that specifies the compression used for .png image files.
                Unlike JPEG, PNG files are always lossless. This value controls the trade-off between the compression
                ratio and the processing time.
            thread_count: The number of writer threads to be used by the saver class. Since ImageSaver uses the
                C-backed OpenCV library, it can safely process multiple frames at the same time via multithreading.
                This controls the number of simultaneously saved images the class instance will support.

        Raises:
            TypeError: If the input arguments are not of the correct type.
        """

        # Verifies that the input arguments are of the correct type.
        if not isinstance(self._output_directory, Path):
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected a valid Path object to "
                f"be provided to the VideoSystem's output_directory argument at initialization, but instead "
                f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
                f"you intend to save camera frames."
            )
            console.error(error=TypeError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise TypeError(message)  # pragma: no cover
        if not isinstance(image_format, ImageFormats):
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected an ImageFormats "
                f"instance for image_format argument, but got {image_format} of type {type(image_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(tiff_compression_strategy, int):
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected an integer for "
                f"tiff_compression_strategy argument, but got {tiff_compression_strategy} of type "
                f"{type(tiff_compression_strategy).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(jpeg_quality, int) or not 0 <= jpeg_quality <= 100:
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected an integer between 0 "
                f"and 100 for jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
            )
            console.error(error=TypeError, message=message)
        if jpeg_sampling_factor not in [
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]:
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected one of the "
                f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_*' constants for jpeg_sampling_factor argument, but got "
                f"{jpeg_sampling_factor} of type {type(jpeg_sampling_factor).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(png_compression, int) or not 0 <= png_compression <= 9:
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected an integer between 0 "
                f"and 9 for png_compression argument, but got {png_compression} of type "
                f"{type(png_compression).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(thread_count, int) or not 0 < thread_count:
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected an integer greater "
                f"than 0 for thread_count argument, but got {thread_count} of type {type(thread_count).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(source_ids, list) or not all(isinstance(camera_id, int) for camera_id in source_ids):
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected a list of integers "
                f"for source_ids argument, but got {source_ids} of type {type(source_ids).__name__}."
            )
            console.error(error=TypeError, message=message)

        if not isinstance(source_ids, tuple) or not all(isinstance(source_id, int) for source_id in source_ids):
            message = (
                f"Unable to add the ImageSaver object to the {self._name} VideoSystem. Expected a tuple of one or "
                f"more integers as source_ids argument, but got {source_ids} of type {type(source_ids).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Configures, initializes, and adds the ImageSaver object to the saver list.
        saver = ImageSaver(
            output_directory=self._output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression_strategy,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=jpeg_sampling_factor,
            png_compression=png_compression,
            thread_count=thread_count,
        )
        self._savers.append(_SaverSystem(saver=saver, source_ids=source_ids))

    def add_video_saver(
        self,
        source_ids: tuple[int],
        hardware_encoding: bool = False,
        video_format: VideoFormats = VideoFormats.MP4,
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: GPUEncoderPresets | CPUEncoderPresets = CPUEncoderPresets.SLOW,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ) -> None:
        """Creates a VideoSaver class instance and adds it to the list of savers managed by the VideoSystem instance.

        This method allows adding VideoSavers to an initialized VideoSystem instance. Currently, this is the only
        intended way of using VideoSaver classes available through this library. VideoSavers are not required for the
        VideoSystem to function properly and, therefore, this method does not need to be called unless you need to save
        the camera frames acquired during the runtime of this VideoSystem as a video.

        Notes:
            VideoSavers rely on third-party software FFMPEG to encode the video frames using GPUs or CPUs. Make sure
            it is installed on the host system and available from Python shell. See https://www.ffmpeg.org/download.html
            for more information.

            This method is specifically designed to add VideoSavers. If you need to add an Image Saver (to save frames
            as standalone images), use the add_image_saver() method instead.

            VideoSavers are the generally recommended saver type to use for most applications. It is also highly advised
            to use hardware encoding if it is available on the host system (requires Nvidia GPU).

        Args:
            source_ids: The list of Camera object indices whose frames will be saved by this VideoSaver. The indices
                are based on the order the cameras were / will be added to the VideoSystem instance with index 0 being
                the first added camera. This argument is very important as it directly determines what frames are saved
                and by what savers. If you do not need to save any frames, do not add ImageSaver or VideoSaver instances
                at all.l
            hardware_encoding: Determines whether to use GPU (hardware) encoding or CPU (software) encoding. It is
                almost always recommended to use the GPU encoding for considerably faster encoding with almost no
                quality loss. However, GPU encoding is only supported by modern Nvidia GPUs.
            video_format: The container format to use for the output video file. Use VideoFormats enumeration to
                specify the desired container format. Currently, only 'MP4', 'MKV', and 'AVI' are supported.
            video_codec: The codec (encoder) to use for generating the video file. Use VideoCodecs enumeration to
                specify the desired codec. Currently, only 'H264' and 'H265' are supported.
            preset: The encoding preset to use for the video file. Use GPUEncoderPresets or CPUEncoderPresets
                enumerations to specify the preset. Note, you have to select the correct preset enumeration based on
                whether hardware encoding is enabled (GPU) or not (CPU)!
            input_pixel_format: The pixel format used by input data. Use InputPixelFormats enumeration to specify the
                pixel format of the frame data that will be passed to this saver. Currently, only 'MONOCHROME', 'BGR',
                and 'BGRA' options are supported. The correct option to choose depends on the configuration of the
                Camera class(es) that acquire frames for this saver.
            output_pixel_format: The pixel format to be used by the output video file. Use OutputPixelFormats
                enumeration to specify the desired pixel format. Currently, only 'YUV420' and 'YUV444' options are
                supported.
            quantization_parameter: The integer value to use for the 'quantization parameter' of the encoder. The
                encoder uses 'constant quantization' to discard the same amount of information from each macro-block of
                the encoded frame, instead of varying the discarded information amount with the complexity of
                macro-blocks. This allows precisely controlling output video size and distortions introduced by the
                encoding process, as the changes are uniform across the whole video. Lower values mean better quality
                (0 is best, 51 is worst). Note, the default value assumes H265 encoder and is likely too low for H264
                encoder. H264 encoder should default to ~25.
            gpu: The index of the GPU to use for hardware encoding. Valid GPU indices can be obtained from 'nvidia-smi'
                command and start with 0 for the first available GPU. This is only used when hardware_encoding is True
                and is ignored otherwise.

        Raises:
            TypeError: If the input arguments are not of the correct type.
            RuntimeError: If the instantiated saver is configured to use GPU video encoding, but the method does not
                detect any available NVIDIA GPUs. If FFMPEG is not accessible from the Python shell.
        """

        # Verifies that the input arguments are of the correct type.
        if not isinstance(self._output_directory, Path):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a valid Path object to "
                f"be provided to the VideoSystem's output_directory argument at initialization, but instead "
                f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
                f"you intend to save camera frames."
            )
            console.error(error=TypeError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise TypeError(message)  # pragma: no cover
        if not isinstance(hardware_encoding, bool):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a boolean for "
                f"hardware_encoding argument, but got {hardware_encoding} of type {type(hardware_encoding).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(video_format, VideoFormats):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a VideoFormats "
                f"instance for video_format argument, but got {video_format} of type {type(video_format).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(video_codec, VideoCodecs):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a VideoCodecs instance "
                f"for video_codec argument, but got {video_codec} of type {type(video_codec).__name__}."
            )
            raise console.error(error=TypeError, message=message)

        # The encoding preset depends on whether the saver is configured to use hardware (GPU) video encoding.
        if hardware_encoding:
            if not isinstance(preset, GPUEncoderPresets):
                message = (
                    f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a "
                    f"GPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
                )
                console.error(error=TypeError, message=message)
        else:
            if not isinstance(preset, CPUEncoderPresets):
                message = (
                    f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a "
                    f"CPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
                )
                console.error(error=TypeError, message=message)

        if not isinstance(input_pixel_format, InputPixelFormats):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected an InputPixelFormats "
                f"instance for input_pixel_format argument, but got {input_pixel_format} of type "
                f"{type(input_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(output_pixel_format, OutputPixelFormats):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected an OutputPixelFormats "
                f"instance for output_pixel_format argument, but got {output_pixel_format} of type "
                f"{type(output_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)

        # While -1 is not explicitly allowed, it is a valid preset to use for 'encoder-default' value. We do not
        # mention it in docstrings, but those who need to know will know.
        if not isinstance(quantization_parameter, int) or not -1 < quantization_parameter <= 51:
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected an integer between 0 "
                f"and 51 for quantization_parameter argument, but got {quantization_parameter} of type "
                f"{type(quantization_parameter).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Since GPU encoding is currently only supported for NVIDIA GPUs, verifies that nvidia-smi is callable
        # for the host system. This is used as a proxy to determine whether the system has an Nvidia GPU:
        if hardware_encoding:
            try:
                # Runs nvidia-smi command, uses check to trigger CalledProcessError exception if runtime fails
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                message = (
                    f"Unable to add the VideoSaver object to the {self._name} VideoSystem. The object is configured to "
                    f"use the GPU video encoding backend, which currently only supports NVIDIA GPUs. Calling "
                    f"'nvidia-smi' to verify the presence of NVIDIA GPUs did not run successfully, indicating that "
                    f"there are no available NVIDIA GPUs on the host system. Use a CPU encoder or make sure nvidia-smi "
                    f"is callable from Python shell."
                )
                console.error(error=RuntimeError, message=message)

        # VideoSavers universally rely on FFMPEG ton be available on the system Path, as FFMPEG is used to encode the
        # videos. Therefore, does a similar check to the one above to make sure that ffmpeg is callable.
        try:
            # Runs nvidia-smi command, uses check to trigger CalledProcessError exception if runtime fails
            subprocess.run(["ffmpeg -version"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. VideoSavers require a "
                f"third-party software, FFMPEG, to be available on the system's Path. Please make sure FFMPEG is "
                f"installed and callable from Python shell. See https://www.ffmpeg.org/download.html for more "
                f"information."
            )
            console.error(error=RuntimeError, message=message)

        if not isinstance(source_ids, tuple) or not all(isinstance(source_id, int) for source_id in source_ids):
            message = (
                f"Unable to add the VideoSaver object to the {self._name} VideoSystem. Expected a tuple of one or "
                f"more integers as source_ids argument, but got {source_ids} of type {type(source_ids).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Configures, initializes, and returns a VideoSaver instance
        saver = VideoSaver(
            output_directory=self._output_directory,
            hardware_encoding=hardware_encoding,
            video_format=video_format,
            video_codec=video_codec,
            preset=preset,
            input_pixel_format=input_pixel_format,
            output_pixel_format=output_pixel_format,
            quantization_parameter=quantization_parameter,
            gpu=gpu,
        )
        self._savers.append(_SaverSystem(saver=saver, source_ids=source_ids))

    def start(self) -> None:
        """Starts the consumer and producer processes of the video system class and begins acquiring camera frames.

        This process begins frame acquisition, but not frame saving. To enable saving acquired frames, call
        start_frame_saving() method. A call to this method is required to make the system operation and should only be
        carried out from the main scope of the runtime context. A call to this method should always be paired with a
        call to the stop() method to properly release the resources allocated to the class.

        Notes:
            By default, this method does not enable saving camera frames to non-volatile memory. This is intentional, as
            in some cases the user may want to see the camera feed, but only record the frames after some initial
            setup. To enable saving camera frames, call the start_frame_saving() method.

        Raises:
            ProcessError: If the method is called outside the '__main__' scope. Also, if this method is called after
                calling the stop() method without first re-initializing the class.
        """

        # if the class is already running, does nothing. This makes it impossible to call start multiple times in a row.
        if self._started:
            return

        # Instantiates an array shared between all processes. This array is used to control all child processes.
        # Index 0 (element 1) is used to issue global process termination command, index 1 (element 2) is used to
        # flexibly enable or disable saving camera frames.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator_array",  # Uses class name with an additional specifier
            prototype=np.zeros(shape=2, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Starts consumer processes first to minimize queue buildup once the producer process is initialized.
        # Technically, due to saving frames being initially disabled, queue buildup is no longer a major concern.
        for camera in self._cameras:
            process = Process(
                target=self._frame_production_loop,
                args=(
                    self._cameras,
                    self._image_queue,
                    self._terminator_array,
                    onset_log,
                    display_frames,
                    fps_override,
                ),
                daemon=True,
            )
            process.start()
            self._consumer_process.append(process)

        # Starts the producer process
        if isinstance(saver, VideoSaver):
            # When the saver class is configured to use the Video backend, it requires additional information to
            # properly encode grabbed frames as a video file. Some of this formation needs to be obtained from a
            # connected camera.
            camera.connect()
            frame_width = camera.width
            frame_height = camera.height

            # For framerate, only retrieves and uses camera framerate if fps_override is not provided. Otherwise, uses
            # the override value. This ensures that the video framerate always matches camera acquisition rate,
            # regardless of the method that enforces the said framerate.
            if fps_override is None:
                frame_rate = camera.fps
            else:
                frame_rate = fps_override

            # Disconnects from the camera, since the producer Process has its own connection subroutine.
            camera.disconnect()

            # Similar to onset_log above, but a separate file to prevent race conditions when multiple savers are used.
            # This is only relevant for ImageSaver(s) (see below), but the interface is kept the same to make it easier
            # to work with this code
            # noinspection PyProtectedMember
            timestamp_log = saver._output_directory.joinpath(f"{self._name}_frame_acquisition_timestamps_saver_1.txt")

            # For VideoSaver, spawns a single process and packages it into a tuple. Since VideoSaver relies on FFMPEG,
            # it automatically scales with available resources without the need for additional Python processes.
            self._consumer_process = (
                Process(
                    target=self._frame_saving_loop,
                    args=(
                        self._savers,
                        self._image_queue,
                        self._terminator_array,
                        timestamp_log,
                        frame_width,
                        frame_height,
                        frame_rate,
                        self._name,
                    ),
                    daemon=True,
                ),
            )
        else:
            # For ImageSaver, spawns the requested number of saver processes. ImageSaver-based processed do not need
            # additional arguments required by VideoSaver processes, so instantiation does not require retrieving any
            # camera information.
            processes: list[Process] = []
            for num in range(image_saver_process_count):
                # To prevent race conditions from multiple savers, each saver writes stamps into a separate file. To do
                # so, the name of the log file is incremented by the count of each saver.
                # noinspection PyProtectedMember
                timestamp_log = saver._output_directory.joinpath(
                    f"{self._name}_frame_acquisition_timestamps_saver_{num + 1}.txt"
                )
                Process(
                    target=self._frame_saving_loop,
                    args=(
                        self._savers,
                        self._image_queue,
                        self._terminator_array,
                        timestamp_log,
                    ),
                    daemon=True,
                )

            # Casts the list to tuple and saves it to class attribute.
            self._consumer_process = tuple(processes)

        # Sets the running tracker
        self._started = True

    def stop(self) -> None:
        """Stops all producer and consumer processes and terminates class runtime by releasing all resources.

        While this does not delete the class instance itself, only call this method once, during the general
        termination of the runtime that instantiated the class. This method destroys the shared memory array buffer,
        so it is impossible to call start() after stop() has been called without re-initializing the class.

        Notes:
            The class will be kept alive until all frames buffered to the image_queue are saved. This is an intentional
            security feature that prevents information loss. If you want to override that behavior, you can initialize
            the class with a 'shutdown_timeout' argument to specify a delay after which all consumers will be forcibly
            terminated. Generally, it is highly advised not to tamper with this feature. The class uses the default
            timeout of 10 minutes (600 seconds), unless this is overridden at instantiation.
        """
        # Ensures that the stop procedure is only executed if the class is running
        if not self._started:
            return

        # Sets both variables in the terminator array to 0, which initializes the shutdown procedure. Technically, only
        # the first variable needs to be set to 0 for this, but resetting the array to 0 works too.
        self._terminator_array.write_data(index=(0, 2), data=[0, 0])

        # This statically blocks until either the queue is empty or the timeout (in seconds) is reached.
        wait_timer = PrecisionTimer("s")
        while self._shutdown_timeout is None or wait_timer.elapsed > self._shutdown_timeout:
            if self._image_queue.empty():
                break

        # Shuts down the manager. This terminates the Queue and discards any information buffered in the queue if it
        # is not saved. This has to be executed for the processes to be able to join, but after allowing them to
        # gracefully empty the queue through the timeout functionality.
        self._mp_manager.shutdown()

        # Joins the producer and consumer processes
        self._producer_process.join()
        for process in self._consumer_process:
            process.join()

        # Disconnects from and destroys the terminator array buffer. This step destroys the shared memory buffer,
        # making it impossible to call start() again.
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

        # Sets running tracker
        self._started = False

    @property
    def started(self) -> bool:
        """Returns true if the system has been started and has active daemon processes connected to cameras and
        saver."""
        return self._started

    @property
    def name(self) -> str:
        """Returns the name of the VideoSystem class instance."""
        return self._name

    @property
    def description(self) -> str:
        """Returns the description of the VideoSystem class instance."""
        return self._description

    @property
    def id_code(self) -> np.uint8:
        """Returns the unique identifier code assigned to the VideoSystem class instance."""
        return self._id

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing Queue object used by the system's producer process to send frames to other
        concurrently active processes."""
        return self._output_queue

    @staticmethod
    def get_opencv_ids() -> tuple[str, ...]:
        """Discovers and reports IDs and descriptive information about cameras accessible through the OpenCV library.

        This method can be used to discover camera IDs accessible through the OpenCV backend. Next, each of the IDs can
        be used via the add_camera() method to add the specific camera to a VideoSystem instance.

        Notes:
            Currently, there is no way to get serial numbers or usb port names from OpenCV. Therefore, while this method
            tries to provide some ID information, it likely will not be enough to identify the cameras. Instead, it is
            advised to test each of the IDs with 'interactive-run' CLI command to manually map IDs to cameras based
            on the produced visual stream.

            This method works by sequentially evaluating camera IDs starting from 0 and up to ID 100. The method
            connects to each camera and takes a test image to ensure the camera is accessible, and it should ONLY be
            called when no OpenCVCamera or any other OpenCV-based connection is active. The evaluation sequence will
            stop early if it encounters more than 5 non-functional IDs in a row.

            This method will yield errors from OpenCV, which are not circumventable at this time. That said,
            since the method is not designed to be used in well-configured production runtimes, this is not
            a major concern.

        Returns:
             A tuple of strings. Each string contains camera ID, frame width, frame height, and camera fps value.
        """
        non_working_count = 0
        working_ids = []

        # This loop will keep iterating over IDs until it discovers 5 non-working IDs. The loop is designed to
        # evaluate 100 IDs at maximum to prevent infinite execution.
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

    @staticmethod
    def get_harvesters_ids(cti_path: Path) -> tuple[str, ...]:
        """Discovers and reports IDs and descriptive information about cameras accessible through the Harvesters
        library.

        Since Harvesters already supports listing valid IDs available through a given .cti interface, this method
        uses built-in Harvesters methods to discover and return camera ID and descriptive information.
        The discovered IDs can later be used via the add_camera() method to add the specific camera to a VideoSystem
        instance.

        Notes:
            This method bundles discovered ID (list index) information with the serial number and the camera model to
            aid identifying physical cameras for each ID.

        Args:
            cti_path: The path to the '.cti' file that provides the GenTL Producer interface. It is recommended to use
                the file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire,
                would work as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more
                details.

        Returns:
            A tuple of strings. Each string contains camera ID, serial number, and model name.
        """

        # Instantiates the class and adds the input .cti file.
        harvester = Harvester()
        harvester.add_file(file_path=str(cti_path))

        # Gets the list of accessible cameras
        harvester.update()

        # Loops over all discovered cameras and parses basic ID information from each camera to generate a descriptive
        # string.
        working_ids = []
        for num, camera_info in enumerate(harvester.device_info_list):
            descriptive_string = (
                f"Harvesters Camera ID: {num}, Serial Number: {camera_info.serial_number}, "
                f"Model Name: {camera_info.model}."
            )
            working_ids.append(descriptive_string)

        return tuple(working_ids)  # Converts to tuple before returning to caller.

    @staticmethod
    def _frame_display_loop(display_queue: Queue, camera_name: str) -> None:  # type: ignore
        """Continuously fetches frame images from display_queue and displays them via OpenCV imshow() method.

        This method runs in a thread as part of the _produce_images_loop() runtime. It is used to display
        frames as they are grabbed from the camera and passed to the multiprocessing queue. This allows visually
        inspecting the frames as they are processed.

        Notes:
            Since the method uses OpenCV under-the-hood, it repeatedly releases GIL as it runs. This makes it
            beneficial to have this functionality as a sub-thread, instead of realizing it at the same level as the
            rest of the image production loop code.

            This thread runs until it is terminated through the display window GUI or passing a non-NumPy-array
            object (e.g.: integer -1) through the display_queue.

        Args:
            display_queue: A multithreading Queue object that is used to buffer grabbed frames to de-couple display from
                acquisition. It is expected that the queue yields frames as NumPy ndarray objects. If the queue yields a
                non-array object, the thread terminates.
            camera_name: The name of the camera which produces displayed images. This is used to generate a descriptive
                window name for the display GUI.
        """

        # Initializes the display window using 'normal' mode to support user-controlled resizing.
        window_name = f"{camera_name} Frames."
        cv2.namedWindow(winname=window_name, flags=cv2.WINDOW_NORMAL)

        # Runs until manually terminated by the user through GUI or programmatically through the thread kill argument.
        while True:
            # This can be blocking, since the loop is terminated by passing 'None' through the queue
            frame = display_queue.get()

            # Programmatic termination is done by passing a non-numpy-array input through the queue
            if not isinstance(frame, np.ndarray):
                display_queue.task_done()  # If the thread is terminated, ensures join() will work as expected
                break

            # Displays the image using the window created above
            cv2.imshow(winname=window_name, mat=frame)

            # Manual termination is done through window GUI
            if cv2.waitKey(1) & 0xFF == 27:
                display_queue.task_done()  # If the thread is terminated, ensures join() will work as expected
                break

            display_queue.task_done()  # Ensures each get() is paired with a task_done() call once display cycle is over

        # Cleans up after runtime by destroying the window. Specifically, targets the window created by this thread to
        # avoid interfering with any other windows.
        cv2.destroyWindow(window_name)

    @staticmethod
    def _frame_production_loop(
        video_system_id: int,
        cameras: tuple[_CameraSystem, ...],
        image_queue: MPQueue,  # type: ignore
        output_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
    ) -> None:
        """Continuously grabs frames from each managed camera and queues them up to be saved by the consumer process.

        This method loops while the first element in terminator_array (index 0) is zero. It continuously grabs
        frames from each managed camera but only queues them up to be saved by the consumer process if the second
        element in terminator_array (index 1) is not zero.

        This method also displays the acquired frames for the cameras that requested this functionality in a separate
        display thread (one per each camera).

        Notes:
            This method should be executed by the producer Process. It is not intended to be executed by the main
            process where VideoSystem is instantiated.

            In addition to sending data to the consumer process, this method also outputs the frame data to other
            concurrent processes via the output_queue. This is only done for cameras that explicitly request this
            feature.

            This method also generates and logs the acquisition onset date and time, which is used to align different
            log data sources to each other when post-processing acquired data.

        Args:
            video_system_id: The unique byte-code identifier of the VideoSystem instance. This is used to identify the
                VideoSystem when logging data.
            cameras: The list of CameraSystem instances managed by this VideoSystem instance.
            image_queue: A multiprocessing queue that buffers and pipes acquired frames to the consumer process.
            output_queue: A multiprocessing queue that buffers and pipes acquired frames to other concurrently active
                processes (not managed by this VideoSystem instance).
            logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
                This queue is used to buffer and pipe data to be logged to the logger cores. This method only logs the
                onset of frame data acquisition, which is used to align different log sources to each other during
                post-processing. The saver loop is used to log frame acquisition times, only preserving the frames that
                are saved to disk.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
        """
        # Connects to the terminator array. This has to be done before preparing camera systems in case connect()
        # method fails for any camera objects.
        terminator_array.connect()

        # Creates a timer that time-stamps acquired frames. This information is crucial for later alignment of multiple
        # data sources. This timer is shared between all managed cameras.
        stamp_timer: PrecisionTimer = PrecisionTimer("us")

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later time
        # readouts.
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)
        stamp_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # Sends the onset data to the logger queue. The time_stamp of 0 is universally interpreted as the timer
        # onset. The serialized data has to be the byte-converted onset date and time using UTC timezone.
        package = LogPackage(source_id=video_system_id, time_stamp=0, serialized_data=onset)
        logger_queue.put(package)

        # Loops over cameras and precreates the necessary objects to control the flow of acquired frames between
        # various VideoSystem components
        camera_dict = {}
        for camera in cameras:
            # Extracts the camera name. This is used to generate dictionary entries for each camera.
            camera_name = camera.camera.name

            # For each camera configured to display frames, creates a worker thread and queue object that handles
            # displaying the frames.
            if camera.display_frames:
                # Creates queue and thread for this camera
                display_queue = Queue()
                display_thread = Thread(target=VideoSystem._frame_display_loop, args=(display_queue, camera_name))
                display_thread.start()

                # Converts the timeout between showing two consecutive frames from frames_per_second to
                # microseconds_per_frame. For this, first divides one second by the number of frames (to get seconds per
                # frame) and then translates that into microseconds. This gives the timeout between two consecutive
                # frame acquisitions.
                show_time = convert_time(
                    time=float(1 / camera.display_frame_rate),  # type: ignore
                    from_units="s",
                    to_units="us",
                    convert_output=True,
                )
            else:
                display_queue = None
                display_thread = None
                show_time = None

            if camera.fps_override:
                # Calculates the microseconds per frame for the acquisition framerate control.
                frame_time = convert_time(
                    time=float(1 / camera.camera.fps),  # type: ignore
                    from_units="s",
                    to_units="us",
                    convert_output=True,
                )
            else:
                frame_time = -1.0  # This is essentially similar to None

            # Calculates the timeout in microseconds per frame, for outputting the frames to other processes.
            if camera.output_frames:
                output_time = convert_time(
                    time=camera.output_frame_rate,  # type: ignore
                    from_units="s",
                    to_units="us",
                    convert_output=True,
                )
            else:
                output_time = None

            # Initializes the high-precision timer used to override managed camera frame rate. This is used to adjust
            # the framerate for cameras that do not support hardware frame rate control.
            acquisition_timer: PrecisionTimer = PrecisionTimer("us")

            # Also initializes a timer to limit displayed frames to 30 fps. This is used to save computational
            # resources, as displayed framerate does not need to be as high as acquisition framerate. This becomes very
            # relevant for displaying large frames at high speeds, as the imshow backend sometimes gets overwhelmed,
            # leading to queue backlogging and displayed data lagging behind the real stream.
            show_timer: PrecisionTimer = PrecisionTimer("us")

            # Also initializes the timer used to limit the framerate at which frames are added to the output queue.
            output_timer: PrecisionTimer = PrecisionTimer("us")

            # Fills the camera dictionary with the necessary data for each camera:
            # Display
            camera_dict[camera_name]["display_queue"] = display_queue
            camera_dict[camera_name]["display_thread"] = display_thread
            camera_dict[camera_name]["show_time"] = show_time
            camera_dict[camera_name]["show_timer"] = show_timer

            # Acquisition
            camera_dict[camera_name]["acquisition_timer"] = acquisition_timer
            camera_dict[camera_name]["frame_time"] = frame_time

            # Output
            camera_dict[camera_name]["output_timer"] = output_timer
            camera_dict[camera_name]["output_time"] = output_time

            camera.camera.connect()  # Connects to the hardware of each camera.

        # Sets the 3d index value of the terminator_array to 1 to indicate that all CameraSystems have been started.
        terminator_array.write_data(index=3, data=np.uint8(1))

        # The loop runs until the VideoSystem is terminated by setting the first element (index 0) of the array to 1
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Loops over each camera during each acquisition cycle:
            for camera_index, camera_system in enumerate(cameras):
                # Extracts the camera class and acquisition runtime parameters from the camera_dictionary
                camera = camera_system.camera
                camera_name = camera.name

                # Display
                show_time = camera_dict[camera_name]["show_time"]
                show_timer = camera_dict[camera_name]["show_timer"]
                display_queue = camera_dict[camera_name]["display_queue"]

                # Acquisition
                frame_time = camera_dict[camera_name]["frame_time"]
                acquisition_timer = camera_dict[camera_name]["acquisition_timer"]

                # Output
                output_timer = camera_dict[camera_name]["output_timer"]
                output_time = camera_dict[camera_name]["output_time"]

                # Grabs the first available frame as a numpy ndarray
                frame = camera.grab_frame()
                frame_stamp = stamp_timer.elapsed  # Generates the time-stamp for the acquired frame

                # If the software framerate override is enabled, this loop is further limited to acquire frames at the
                # specified rate, which is helpful for some cameras that do not have a built-in acquisition control
                # functionality. If the acquisition timeout has not passed and is enabled, skips the rest of the
                # processing runtime
                if acquisition_timer.elapsed < frame_time:
                    continue

                # Bundles frame data, camera index, and acquisition timestamp relative to the onset and passes them to
                # the multiprocessing Queue that delivers the data to the consumer process that saves it to disk. This
                # is only executed if frame saving is enabled via the terminator_array variable using index 1.
                if terminator_array.read_data(index=1):
                    image_queue.put((frame, camera_index, frame_stamp))
                    acquisition_timer.reset()  # Resets the acquisition timer after saving each frame

                # If ths currently processed camera is configured to output frame data via the output_queue, the output
                # timeout has expired, and frame output is enabled via the terminator_array index 2 value, sends the
                # data to the queue. Does not include frame acquisition timestamp, as this data is usually not
                # necessary for real time processing.
                if terminator_array.read_data(index=2) and output_timer.elapsed >= output_time:
                    output_queue.put((frame, camera_index))  # type: ignore
                    output_timer.reset()  # Resets the output timer

                # If the process is configured to display acquired frames, queues each frame to be displayed. Note, this
                # does not depend on whether the frame is buffered for saving. The display frame limiting is
                # critically important, as it prevents the display thread from being overwhelmed, causing displayed
                # stream to lag behind the saved stream.
                if display_queue is not None and show_timer.elapsed >= show_time:
                    display_queue.put(frame)  # type: ignore
                    show_timer.reset()  # Resets the display timer

        # Once the loop above is escaped, releases all resources and terminates the Process.
        for camera in cameras:
            # Extracts the camera name. This is used to generate dictionary entries for each camera.
            camera_name = camera.camera.name

            # Extracts Queue and Thread used for displaying images
            display_queue = camera_dict[camera_name]["display_queue"]
            display_thread = camera_dict[camera_name]["display_thread"]

            # Terminates the display thread
            display_queue.put(None)  # type: ignore
            # Waits for the thread to close
            display_thread.join()  # type: ignore

            camera.camera.disconnect()  # Disconnects from the camera

        # Once all cameras are disconnected, also disconnects from the shared memory array.
        terminator_array.disconnect()  # Disconnects from the terminator array

    @staticmethod
    def _frame_saving_loop(
        video_system_id: int,
        savers: tuple[_SaverSystem, ...],
        image_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
        frame_width: int,
        frame_height: int,
        video_frames_per_second: float,
    ) -> None:
        """Continuously grabs frames from the image_queue and saves them as standalone images or video file, depending
        on the saver class backend.

        This method loops while the first element in terminator_array (index 0) is nonzero. It continuously grabs
        and saves frames buffered through image_queue. The method also logs frame acquisition timestamps, which are
        buffered with each frame data and ID. This method is meant to be run as a process, and it will create an
        infinite loop if run on its own.

        Notes:
            If Saver class is configured to use Image backend and multiple saver processes were requested during
            VideoSystem instantiation, this loop will be used by multiple processes at the same time. This increases
            saving throughput at the expense of using more resources. This may also affect the order of entries in the
            frame acquisition log.

            For Video encoder, the class requires additional information about the encoded data, including the
            identifier to use for the video file. Otherwise, all additional setup / teardown steps are resolved
            automatically as part of this method's runtime.

            This method's main loop will be kept alive until the image_queue is empty. This is an intentional security
            feature that ensures all buffered images are processed before the saver is terminated. To override this
            behavior, you will need to use the process kill command, but it is strongly advised not to tamper
            with this feature.

            This method expects that image_queue buffers 3-element tuples that include frame data, frame id and
            frame acquisition time relative to the onset point in microseconds.

        Args:
            savers: One of the supported Saver classes that is used to save buffered camera frames by interfacing with
                the OpenCV or FFMPEG libraries.
            image_queue: A multiprocessing queue that buffers and pipes frames acquired by the producer process.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
            frame_width: Only for VideoSaver classes. Specifies the width of the frames to be saved, in pixels.
                This has to match the width reported by the Camera class that produces the frames.
            frame_height: Same as above, but specifies the height of the frames to be saved, in pixels.
            video_frames_per_second: Only for VideoSaver classes. Specifies the desired frames-per-second of the
                encoded video file.
        """
        # Connects to the terminator array to manage loop runtime
        terminator_array.connect()

        frame_counters = []
        for saver in savers:
            # Video savers require additional setup before they can save 'live' frames. Image savers are ready to save
            # frames with no additional setup
            if isinstance(saver.saver, VideoSaver):
                video_id = ""
                for num, source_id in enumerate(saver.source_ids):
                    if num != len(saver.source_ids) - 1:
                        video_id += str(source_id) + "_"
                    else:
                        video_id += str(source_id)

                # Initializes the video encoding by instantiating the FFMPEG process that does the encoding. For this, it
                # needs some additional information. Subsequently, Video and Image savers can be accessed via the same
                # frame-saving API.
                saver.saver.create_live_video_encoder(
                    frame_width=frame_width,
                    frame_height=frame_height,
                    video_id=video_id,
                    video_frames_per_second=video_frames_per_second,
                )

                frame_counters.append(1)

        # The loop runs continuously until the first (index 0) element of the array is set to 0. Note, the loop is
        # kept alive until all frames in the queue are processed!
        while terminator_array.read_data(index=0, convert_output=True) or not image_queue.empty():
            for saver_index, saver_system in enumerate(savers):
                saver = saver_system.saver

                # Grabs the image bundled with its ID and acquisition time from the queue and passes it to Saver class.
                # This relies on both Image and Video savers having the same 'live' frame saving API. Uses a
                # non-blocking binding to prevent deadlocking the loop, as it does not use the queue-based termination
                # method for reliability reasons.
                try:
                    frame, camera_id, frame_time = image_queue.get_nowait()
                except Exception:
                    # The only expected exception here is when queue is empty. This block overall makes the code
                    # repeatedly cycle through the loop, allowing flow control statements to properly terminate the
                    # 'while' loop if necessary and the queue is empty.
                    continue

                # Casts the frame_id to string using enough padding to represent a 64-bit integer
                frame_id = f"{frame_counters[saver_index]:020d}"
                saver.save_frame(frame_id, frame)

                # Packages and sends the data to Logger.
                package = LogPackage(
                    video_system_id,
                    time_stamp=frame_time,
                    serialized_data=np.array(
                        object=[np.uint8(camera_id), np.uint64(frame_counters[saver_index])], dtype=np.uint8
                    ),
                )
                logger_queue.put(package)

        # Terminates the video encoder as part of the shutdown procedure
        for saver in savers:
            if isinstance(saver.saver, VideoSaver):
                saver.saver.terminate_live_encoder()
            else:
                saver.saver.shutdown()

        # Once the loop above is escaped, releases all resources and terminates the Process.
        terminator_array.disconnect()

    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the producer and consumer processes are alive
        during runtime.

        This function will raise a RuntimeError if it detects that a monitored process has prematurely shut down. It
        will verify process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

            if not self._started:
                continue

            # Producer process
            if self._producer_process is not None and not self._producer_process.is_alive():
                message = (
                    f"The producer process for the {self._name} VideoSystem with id {self._id} has been prematurely "
                    f"shut down. This likely indicates that the process has encountered a runtime error that "
                    f"terminated the process."
                )
                console.error(message=message, error=RuntimeError)

            # Consumer process:
            if self._consumer_process is not None and not self._consumer_process.is_alive():
                message = (
                    f"The consumer process for the {self._name} VideoSystem with id {self._id} has been prematurely "
                    f"shut down. This likely indicates that the process has encountered a runtime error that "
                    f"terminated the process."
                )
                console.error(message=message, error=RuntimeError)

    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames.

        Does not interfere with grabbing and displaying the frames to user, this process is only stopped when the main
        stop() method is called.
        """
        if self._started:
            self._terminator_array.write_data(index=1, data=0)

    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames.

        The frames are grabbed and (optionally) displayed to user after the main start() method is called, but they
        are not initially written to non-volatile memory. The call to this method additionally enables saving the
        frames to non-volatile memory.
        """
        if self._started:
            self._terminator_array.write_data(index=1, data=1)
