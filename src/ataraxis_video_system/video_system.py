"""This module provides the main VideoSystem class that contains methods for setting up, running, and tearing down
interactions between Camera and Saver classes.

While Camera and Saver classes provide an interface for cameras and saver backends, VideoSystem connects cameras to
savers and manages the flow of frames between them. Each VideoSystem is a self-contained entity that provides a simple
API for recording data from a wide range of cameras as images or videos. The class is written in a way that maximizes
runtime performance.

All user-oriented functionality of this library is available through the public methods of the VideoSystem class.
"""

import os
import sys
from queue import Queue
from types import NoneType
from typing import Any
from pathlib import Path
import warnings
from threading import Thread
import subprocess
from dataclasses import dataclass
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from harvesters.core import Harvester  # type: ignore
from numpy.lib.npyio import NpzFile
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, LogPackage, SharedMemoryArray
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

    """Determines whether to save the frames acquired by this camera to disk. Setting this to True only means that 
    the frames will be submitted to the saving queue. There has to also be a Saver configured to grab and save these 
    frames."""
    save_frames: bool

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


class VideoSystem:
    """Efficiently combines a Camera and a Saver instance to acquire, display, and save camera frames to disk.

    This class controls the runtime of Camera and Saver instances running on independent cores (processes) to maximize
    the frame throughout. The class achieves two main objectives: It efficiently moves frames acquired by the camera
    to the saver that writes them to disk and manages runtime behavior of wrapped Camera and Saver classes. To do so,
    the class initializes and controls multiple subprocesses to ensure frame producer and consumer classes have
    sufficient computational resources. The class abstracts away all necessary steps to set up, execute, and tear down
    the processes through an easy-to-use API.

    Notes:
        Due to using multiprocessing to improve efficiency, this class reserves up to two logical cores to run
        efficiently. Additionally, due to a time-lag of moving frames from the producer process to the
        consumer process, the class will reserve a variable portion of the RAM to buffer the frames. The reserved
        memory depends on many factors and can only be established empirically.

        The class does not initialize cameras or savers at instantiation. Instead, you have to manually use the
        add_camera() and add_image_saver() or add_video_saver() methods to add camera and saver to the system. To use
        more than one camera and one saver, create additional VideoSystem instances as necessary.

    Args:
        system_id: A unique byte-code identifier for the VideoSystem instance. This is used to identify the system in
            log files and generated video files. Therefore, this ID has to be unique across all concurrently
            active Ataraxis systems that use DataLogger to log data (such as AtaraxisMicroControllerInterface class).
            Note, this ID is used to identify all frames saved by the same VideoSystem and to name log files generated
            by the VideoSystem.
        data_logger: An initialized DataLogger instance used to log the timestamps for all frames saved by this
            VideoSystem instance. The DataLogger itself is NOT managed by this instance and will need to be activated
            separately. This instance only extracts the necessary information to pipe the data to the logger.
        output_directory: The path to the output directory which will be used by the Saver class to save
            acquired camera frames as images or videos. This argument is required if you intend to add a Saver instance
            to this VideoSystem and optional if you do not intend to save frames grabbed by this VideoSystem.
        harvesters_cti_path: The path to the '.cti' file that provides the GenTL Producer interface. This argument is
            required if you intend to interface with cameras using 'Harvesters' backend! It is recommended to use the
            file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire, would work
            as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details.

    Attributes:
        _id: Stores the ID code of the VideoSystem instance.
        _logger_queue: Stores the multiprocessing Queue object used to buffer and pipe data to be logged.
        _log_directory: Stores the Path to the log folder directory.
        _output_directory: Stores the path to the output directory.
        _cti_path: Stores the path to the '.cti' file that provides the GenTL Producer interface.
        _camera: Stores managed CameraSystem.
        _saver: Stores managed SaverSystem.
        _started: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores a SyncManager class instance from which the image_queue and the log file Lock are derived.
        _image_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from the producer to
            the consumer process.
        _output_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from the producer to
            other concurrently active processes.
        _terminator_array: A SharedMemoryArray instance that provides a cross-process communication interface used to
            manage runtime behavior of spawned processes.
        _producer_process: A process that acquires camera frames using managed CameraSystem.
        _consumer_process: A process that saves the acquired frames using managed SaverSystem.
        _watchdog_thread: A thread used to monitor the runtime status of the remote consumer and producer processes.

    Raises:
        TypeError: If any of the provided arguments has an invalid type.
    """

    def __init__(
        self,
        system_id: np.uint8,
        data_logger: DataLogger,
        output_directory: Path | None = None,
        harvesters_cti_path: Path | None = None,
    ):
        # Has to be set first to avoid stop method errors
        self._started: bool = False  # Tracks whether the system has active processes
        # The manager is created early in the __init__ phase to support del-based cleanup
        self._mp_manager: SyncManager = multiprocessing.Manager()

        # Ensures system_id is a byte-convertible integer
        if not isinstance(system_id, np.uint8):
            message = (
                f"Unable to initialize the VideoSystem class instance. Expected a uint8 integer as system_id argument, "
                f"but encountered {system_id} of type {type(system_id).__name__}."
            )
            console.error(message=message, error=TypeError)

        # If harvesters_cti_path is provided, checks if it is a valid file path.
        if harvesters_cti_path is not None and (
            not harvesters_cti_path.exists() or harvesters_cti_path.suffix != ".cti"
        ):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected the path to an existing "
                f"file with a '.cti' suffix or None for harvesters_cti_path, but encountered {harvesters_cti_path} of "
                f"type {type(harvesters_cti_path).__name__}. If a valid path was provided, this error is likely due to "
                f"the file not existing or not being accessible."
            )
            console.error(message=message, error=TypeError)

        # Ensures that the data_logger is an initialized DataLogger instance.
        if not isinstance(data_logger, DataLogger):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected an initialized "
                f"DataLogger instance for 'data_logger' argument, but encountered {data_logger} of type "
                f"{type(data_logger).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Ensures that the output_directory is either a Path instance or None:
        if output_directory is not None and not isinstance(output_directory, Path):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected a Path instance or None "
                f"for 'output_directory' argument, but encountered {output_directory} of type "
                f"{type(output_directory).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Saves ID data and the logger queue to class attributes
        self._id: np.uint8 = system_id
        self._logger_queue: MPQueue = data_logger.input_queue  # type: ignore
        self._log_directory: Path = data_logger.output_directory
        self._output_directory: Path | None = output_directory
        self._cti_path: Path | None = harvesters_cti_path

        # Initializes placeholder variables that will be filled by add_camera() and add_saver() methods.
        self._camera: _CameraSystem | None = None
        self._saver: ImageSaver | VideoSaver | None = None

        # Sets up the assets used to manage acquisition and saver processes. The assets are configured during the
        # start() method runtime, most of them are initialized to placeholder values here.
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
        self._mp_manager.shutdown()

    def __repr__(self) -> str:
        """Returns a string representation of the VideoSystem class instance."""

        if self._camera is not None:
            camera_name = str(type(self._camera.camera).__name__)
        else:
            camera_name = "None"

        if self._saver is not None:
            saver_name = str(type(self._saver).__name__)
        else:
            saver_name = "None"

        representation_string: str = (
            f"VideoSystem(id={self._id}, started={self._started}, camera={camera_name}, saver={saver_name})"
        )
        return representation_string

    def add_camera(
        self,
        save_frames: bool,
        camera_index: int = 0,
        camera_backend: CameraBackends = CameraBackends.OPENCV,
        output_frames: bool = False,
        output_frame_rate: float = 25,
        display_frames: bool = False,
        display_frame_rate: float = 25,
        frame_width: int | None = None,
        frame_height: int | None = None,
        acquisition_frame_rate: float | None = None,
        opencv_backend: int | None = None,
        color: bool | None = None,
    ) -> None:
        """Creates a Camera class instance and adds it to the VideoSystem instance.

        This method allows adding Cameras to an initialized VideoSystem instance. Currently, this is the only intended
        way of using Camera classes available through this library. Unlike Saver class instances, which are not
        required for the VideoSystem to function, a valid Camera class must be added to the system before its start()
        method is called. The only exception to this rule is when using the encode_video_from_images() method, which
        requires a VideoSaver and does not require the start() method to be called.

        Notes:
            Calling this method multiple times replaces the existing Camera class instance with a new one.

        Args:
            save_frames: Determines whether frames acquired by this camera should be saved to disk as images or video.
                If this is enabled, there has to be a valid Saver class instance configured to save frames from this
                Camera.
            camera_index: The index of the camera, relative to all available video devices, e.g.: 0 for the first
                available camera, 1 for the second, etc. Usually, the host-system assigns the camera indices
                based on the order of their connection.
            camera_backend: The backend to use for the camera class. Currently, all supported backends are derived from
                the CameraBackends enumeration. The preferred backend is 'Harvesters', but we also support OpenCV for
                non-GenTL-compatible cameras.
            output_frames: Determines whether to output acquired frames via the VideoSystem's output_queue. This
                allows real-time processing of acquired frames by other concurrent processes.
            output_frame_rate: Allows adjusting the frame rate at which acquired frames are sent to the output_queue.
                Note, the output framerate cannot exceed the native frame rate of the camera, but it can be lower than
                the native acquisition frame rate. The acquisition frame rate is set via the frames_per_second argument
                below.
            display_frames: Determines whether to display acquired frames to the user. This allows visually monitoring
                the camera feed in real time. Note, frame displaying may be broken for some macOS versions. This error
                originates from OpenCV and not this library and is usually fixed in a timely manner by OpenCV
                developers.
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

        if not isinstance(camera_index, int):
            message = (
                f"Unable to add a Camera to the VideoSystem with id {self._id}. Expected an "
                f"integer for camera_id argument, but got {camera_index} of type {type(camera_index).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(acquisition_frame_rate, (int, float, NoneType)):
            message = (
                f"Unable to add a Camera to the VideoSystem with id {self._id}. Expected an "
                f"integer, float or None for acquisition_frame_rate argument, but got {acquisition_frame_rate} of type "
                f"{type(acquisition_frame_rate).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(frame_width, (int, NoneType)):
            message = (
                f"Unable to add a Camera to the VideoSystem with id {self._id}. Expected an "
                f"integer or None for frame_width argument, but got {frame_width} of type {type(frame_width).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(frame_height, (int, NoneType)):
            message = (
                f"Unable to add a Camera to the VideoSystem with id {self._id}. Expected an "
                f"integer or None for frame_height argument, but got {frame_height} of type "
                f"{type(frame_height).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Ensures that display_frames is boolean. Does the same for output_frames
        display_frames = False if not isinstance(display_frames, bool) else display_frames
        output_frames = False if not isinstance(output_frames, bool) else output_frames

        # Disables frame displaying on macOS until OpenCV backend issues are fixed
        if display_frames and "darwin" in sys.platform:
            warnings.warn(
                message=(
                    f"Displaying frames is currently not supported for Apple Silicon devices. See ReadMe for details."
                )
            )
            display_frames = False

        # Pre-initializes the fps override to 0. A 0 value indicates that the override is not used. It is enabled
        # automatically as a fallback when the camera lacks native fps limiting capabilities.
        fps_override: int | float = 0

        # Converts integer frames_per_second inputs to floats, since the Camera classes expect it to be a float.
        if isinstance(acquisition_frame_rate, int):
            acquisition_frame_rate = float(acquisition_frame_rate)

        # Presets the variable type
        camera: OpenCVCamera | HarvestersCamera | MockCamera

        # OpenCVCamera
        if camera_backend == CameraBackends.OPENCV:
            # If the backend preference is None, uses generic preference
            if opencv_backend is None:
                opencv_backend = int(cv2.CAP_ANY)
            # If the backend is not an integer or None, raises an error
            elif not isinstance(opencv_backend, int):
                message = (
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Expected an integer or None for opencv_backend argument, but got {opencv_backend} of type "
                    f"{type(opencv_backend).__name__}."
                )
                console.error(error=TypeError, message=message)

            # Ensures that color is either True or False.
            image_color = False if not isinstance(color, bool) else color

            # Instantiates the OpenCVCamera class object
            camera = OpenCVCamera(
                camera_id=self._id,
                color=image_color,
                backend=opencv_backend,
                camera_index=camera_index,
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
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Attempted configuring the camera to acquire frames using the provided frame_height "
                    f"{frame_height}, but the camera returned a test frame with height {frame.shape[0]}. This "
                    f"indicates that the camera does not support the requested frame height and width combination."
                )
                console.error(error=ValueError, message=message)
            if frame_width is not None and frame.shape[1] != frame_width:
                message = (
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Attempted configuring the camera to acquire frames using the provided frame_width {frame_width}, "
                    f"but the camera returned a test frame with width {frame.shape[1]}. This indicates that the camera "
                    f"does not support the requested frame height and width combination."
                )
                console.error(error=ValueError, message=message)
            if color and frame.shape[2] <= 1:  # pragma: no cover
                message = (
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Attempted configuring the camera to acquire colored frames, but the camera returned a test frame "
                    f"with monochrome colorspace. This indicates that the camera does not support acquiring colored "
                    f"frames."
                )
                console.error(error=ValueError, message=message)
            elif not color and len(frame.shape) != 2:  # pragma: no cover
                message = (
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Attempted configuring the camera to acquire monochrome frames, but the camera returned a test "
                    f"frame with BGR colorspace. This likely indicates an OpenCV backend error, since it is unlikely "
                    f"that the camera does not support monochrome colorspace."
                )
                console.error(error=ValueError, message=message)

            # If the camera failed to set the requested frame rate, but it is possible to correct the fps via software,
            # enables fps override. Software correction requires that the native fps is higher than the desired fps,
            # as it relies on discarding excessive frames.
            if acquisition_frame_rate is not None and camera.fps > acquisition_frame_rate:  # type: ignore
                fps_override = acquisition_frame_rate  # pragma: no cover
            elif acquisition_frame_rate is not None and camera.fps < acquisition_frame_rate:  # type: ignore
                message = (
                    f"Unable to add the OpenCVCamera to the VideoSystem with id {self._id}. "
                    f"Attempted configuring the camera to acquire frames at the rate of {acquisition_frame_rate} "
                    f"frames per second, but the camera automatically adjusted the framerate to {camera.fps}. This "
                    f"indicates that the camera does not support the requested framerate."
                )
                console.error(error=ValueError, message=message)

            # Disconnects from the camera to free the resources to be used by the remote producer process, once it is
            # instantiated.
            camera.disconnect()

        # HarvestersCamera
        elif camera_backend == CameraBackends.HARVESTERS:
            # Ensures that the cti_path is provided
            if self._cti_path is None:
                message = (
                    f"Unable to add HarvestersCamera to the VideoSystem with id {self._id}. "
                    f"Expected the VideoSystem's cti_path attribute to be a Path object pointing to the '.cti' file, "
                    f"but got None instead. Make sure you provide a valid '.cti' file as harvesters_cit_file argument "
                    f"when initializing the VideoSystem instance."
                )
                console.error(error=ValueError, message=message)
                # Fallback to appease mypy, should not be reachable
                raise ValueError(message)  # pragma: no cover

            # Instantiates and returns the HarvestersCamera class object
            camera = HarvestersCamera(
                camera_id=self._id,
                cti_path=self._cti_path,
                camera_index=camera_index,
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
                camera_id=self._id,
                camera_index=camera_index,
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
                f"Unable to instantiate the Camera due to encountering an unsupported "
                f"camera_backend argument {camera_backend} of type {type(camera_backend).__name__}. "
                f"camera_backend has to be one of the options available from the CameraBackends enumeration."
            )
            console.error(error=ValueError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise ValueError(message)  # pragma: no cover

        # If the output_frame_rate argument is not an integer or floating value, defaults to using the same framerate
        # as the camera. This has to be checked after the camera has been verified and its fps has been confirmed.
        if output_frames and (
            not isinstance(output_frame_rate, (int, float)) or not 0 <= output_frame_rate <= camera.fps  # type: ignore
        ):
            message = (
                f"Unable to instantiate the Camera due to encountering an unsupported "
                f"output_frame_rate argument {output_frame_rate} of type {type(output_frame_rate).__name__}. "
                f"Output framerate override has to be an integer or floating point number that does not exceed the "
                f"camera acquisition framerate ({camera.fps})."
            )
            console.error(error=TypeError, message=message)

        # Same as above, but for display frame_rate
        if display_frames and (
            not isinstance(display_frame_rate, (int, float)) or not 0 <= output_frame_rate <= camera.fps  # type: ignore
        ):
            message = (
                f"Unable to instantiate the Camera due to encountering an unsupported "
                f"display_frame_rate argument {display_frame_rate} of type {type(display_frame_rate).__name__}. "
                f"Display framerate override has to be an integer or floating point number that does not exceed the "
                f"camera acquisition framerate ({camera.fps})."
            )
            console.error(error=TypeError, message=message)

        # Ensures that save_frames is either True or False.
        save_frames = save_frames if isinstance(save_frames, bool) else False

        # If the camera class was successfully instantiated, packages the class alongside additional parameters into a
        # CameraSystem object and appends it to the camera list.
        self._camera = _CameraSystem(
            camera=camera,
            save_frames=save_frames,
            fps_override=fps_override,
            output_frames=output_frames,
            output_frame_rate=output_frame_rate,
            display_frames=display_frames,
            display_frame_rate=display_frame_rate,
        )

    def add_image_saver(
        self,
        image_format: ImageFormats = ImageFormats.TIFF,
        tiff_compression_strategy: int = cv2.IMWRITE_TIFF_COMPRESSION_LZW,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        png_compression: int = 1,
        thread_count: int = 5,
    ) -> None:
        """Creates an ImageSaver class instance and adds it to the VideoSystem instance.

        This method allows adding an ImageSaver to an initialized VideoSystem instance. Currently, this is the only
        intended way of using ImageSaver class available through this library. ImageSaver is not required for the
        VideoSystem to function and, therefore, this method does not need to be called unless you need to save
        the camera frames acquired during the runtime of this VideoSystem as images.

        Notes:
            Calling this method multiple times will replace the existing Saver (Image or Video) with the new one.

            This method is specifically designed to add ImageSaver instances. If you need to add a VideoSaver
            (to save frames as a video), use the add_video_saver() method instead.

            ImageSavers can reach the highest image saving speed at the cost of using considerably more disk space than
            efficient VideoSavers. Overall, it is highly recommended to use VideoSavers with hardware_encoding where
            possible to optimize the disk space usage and still benefit from a decent frame saving speed.

        Args:
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
                preserve the full color space of the image if your application requires this. Another popular option is
                422, which results in better compression at the cost of color coding precision.
            png_compression: An integer value between 0 and 9 that specifies the compression used for .png image files.
                Unlike JPEG, PNG files are always lossless. This value controls the trade-off between the compression
                ratio and the processing time. Compression level of 0 means uncompressed and 9 means maximum
                compression.
            thread_count: The number of writer threads to be used by the saver class. Since ImageSaver uses the
                C-backed OpenCV library, it can safely process multiple frames at the same time via multithreading.
                This controls the number of simultaneously saved images the class instance will support.

        Raises:
            TypeError: If the input arguments are not of the correct type.
        """
        if not isinstance(self._output_directory, Path):
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected a valid Path "
                f"object to be provided to the VideoSystem's output_directory argument at initialization, but instead "
                f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
                f"you intend to save camera frames."
            )
            console.error(error=TypeError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise TypeError(message)  # pragma: no cover
        if not isinstance(image_format, ImageFormats):
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected an ImageFormats "
                f"instance for image_format argument, but got {image_format} of type {type(image_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(tiff_compression_strategy, int):
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected an integer for "
                f"tiff_compression_strategy argument, but got {tiff_compression_strategy} of type "
                f"{type(tiff_compression_strategy).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(jpeg_quality, int) or not 0 <= jpeg_quality <= 100:
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected an integer "
                f"between 0 and 100 for jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
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
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected one of the "
                f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_*' constants for jpeg_sampling_factor argument, but got "
                f"{jpeg_sampling_factor} of type {type(jpeg_sampling_factor).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(png_compression, int) or not 0 <= png_compression <= 9:
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected an integer "
                f"between 0 and 9 for png_compression argument, but got {png_compression} of type "
                f"{type(png_compression).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(thread_count, int) or thread_count <= 0:
            message = (
                f"Unable to add the ImageSaver object to the VideoSystem with id {self._id}. Expected an integer "
                f"greater than 0 for thread_count argument, but got {thread_count} of type "
                f"{type(thread_count).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Configures, initializes, and adds the ImageSaver object to the saver list.
        self._saver = ImageSaver(
            output_directory=self._output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression_strategy,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=jpeg_sampling_factor,
            png_compression=png_compression,
            thread_count=thread_count,
        )

    def add_video_saver(
        self,
        hardware_encoding: bool = False,
        video_format: VideoFormats = VideoFormats.MP4,
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: GPUEncoderPresets | CPUEncoderPresets = CPUEncoderPresets.SLOW,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ) -> None:
        """Creates a VideoSaver class instance and adds it to the VideoSystem instance.

        This method allows adding a VideoSaver to an initialized VideoSystem instance. Currently, this is the only
        intended way of using VideoSaver class available through this library. VideoSavers are not required for the
        VideoSystem to function and, therefore, this method does not need to be called unless you need to save
        the camera frames acquired during the runtime of this VideoSystem as a video. This method can also be used to
        initialize the VideoSaver that will be used to convert images to a video file via the encode_video_from_images()
        method.

        Notes:
            Calling this method multiple times will replace the existing Saver (Image or Video) with the new one.

            VideoSavers rely on third-party software FFMPEG to encode the video frames using GPUs or CPUs. Make sure
            it is installed on the host system and available from Python shell. See https://www.ffmpeg.org/download.html
            for more information.

            This method is specifically designed to add VideoSaver instances. If you need to add an ImageSaver (to save
            frames as standalone images), use the add_image_saver() method instead.

            VideoSavers are the generally recommended saver type to use for most applications. It is also highly advised
            to use hardware encoding if it is available on the host system (requires Nvidia GPU).

        Args:
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
                Camera class(es) that acquires frames for this saver.
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
            gpu: The index of the GPU to use for hardware encoding. Valid GPU indices can be obtained from the
                'nvidia-smi' command and start with 0 for the first available GPU. This is only used when
                hardware_encoding is True and is ignored otherwise.

        Raises:
            TypeError: If the input arguments are not of the correct type.
            RuntimeError: If the instantiated saver is configured to use GPU video encoding, but the method does not
                detect any available NVIDIA GPUs. If FFMPEG is not accessible from the Python shell.
        """
        if not isinstance(self._output_directory, Path):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a valid Path "
                f"object to be provided to the VideoSystem's output_directory argument at initialization, but instead "
                f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
                f"you intend to save camera frames."
            )
            console.error(error=TypeError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise TypeError(message)  # pragma: no cover
        if not isinstance(hardware_encoding, bool):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a boolean for "
                f"hardware_encoding argument, but got {hardware_encoding} of type {type(hardware_encoding).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(video_format, VideoFormats):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a VideoFormats "
                f"instance for video_format argument, but got {video_format} of type {type(video_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(video_codec, VideoCodecs):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a VideoCodecs "
                f"instance for video_codec argument, but got {video_codec} of type {type(video_codec).__name__}."
            )
            console.error(error=TypeError, message=message)

        # The encoding preset depends on whether the saver is configured to use hardware (GPU) video encoding.
        if hardware_encoding:
            if not isinstance(preset, GPUEncoderPresets):
                message = (
                    f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a "
                    f"GPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
                )
                console.error(error=TypeError, message=message)
        elif not isinstance(preset, CPUEncoderPresets):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected a "
                f"CPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
            )
            console.error(error=TypeError, message=message)

        if not isinstance(input_pixel_format, InputPixelFormats):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected an "
                f"InputPixelFormats instance for input_pixel_format argument, but got {input_pixel_format} of type "
                f"{type(input_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(output_pixel_format, OutputPixelFormats):
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected an "
                f"OutputPixelFormats instance for output_pixel_format argument, but got {output_pixel_format} of type "
                f"{type(output_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)

        # While -1 is not explicitly allowed, it is a valid preset to use for 'encoder-default' value. We do not
        # mention it in docstrings, but those who need to know will know.
        if not isinstance(quantization_parameter, int) or not -1 < quantization_parameter <= 51:
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. Expected an integer "
                f"between 0 and 51 for quantization_parameter argument, but got {quantization_parameter} of type "
                f"{type(quantization_parameter).__name__}."
            )
            console.error(error=TypeError, message=message)

        # VideoSavers universally rely on FFMPEG ton be available on the system Path, as FFMPEG is used to encode the
        # videos. Therefore, does a similar check to the one above to make sure that ffmpeg is callable.
        try:
            # Runs ffmpeg version command, uses check to trigger CalledProcessError exception if runtime fails
            subprocess.run(args=["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:  # pragma: no cover
            message = (
                f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. VideoSavers require a "
                f"third-party software, FFMPEG, to be available on the system's Path. Please make sure FFMPEG is "
                f"installed and callable from Python shell. See https://www.ffmpeg.org/download.html for more "
                f"information."
            )
            console.error(error=RuntimeError, message=message)

        # Since GPU encoding is currently only supported for NVIDIA GPUs, verifies that nvidia-smi is callable
        # for the host system. This is used as a proxy to determine whether the system has an Nvidia GPU:
        if hardware_encoding:
            try:
                # Runs nvidia-smi command, uses check to trigger CalledProcessError exception if runtime fails
                subprocess.run(
                    args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except Exception:  # pragma: no cover
                message = (
                    f"Unable to add the VideoSaver object to the VideoSystem with id {self._id}. The object is "
                    f"configured to use the GPU video encoding backend, which currently only supports NVIDIA GPUs. "
                    f"Calling 'nvidia-smi' to verify the presence of NVIDIA GPUs did not run successfully, indicating "
                    f"that there are no available NVIDIA GPUs on the host system. Use a CPU encoder or make sure "
                    f"nvidia-smi is callable from Python shell."
                )
                console.error(error=RuntimeError, message=message)

        # Configures, initializes, and returns a VideoSaver instance
        self._saver = VideoSaver(
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

    def start(self) -> None:
        """Starts the consumer and producer processes of the VideoSystem instance and begins acquiring camera frames.

        This process starts acquiring frames but does not save them! To enable saving acquired frames, call the
        start_frame_saving() method. A call to this method should always be paired with a call to the stop() method to
        properly release the resources allocated to the class.

        Notes:
            By default, this method does not enable saving camera frames to non-volatile memory. This is intentional, as
            in some cases the user may want to see the camera feed but only record the frames after some initial
            setup. To enable saving camera frames, call the start_frame_saving() method.

        Raises:
            RuntimeError: If starting the consumer or producer processes stalls or fails. If the camera is configured to
            save frames, but there is no Saver. If there is no Camera to acquire frames.
        """
        # Skips re-starting the system if it is already started.
        if self._started:
            return

        # This timer is used to forcibly terminate processes that stall at initialization.
        initialization_timer = PrecisionTimer(precision="s")

        # Prevents starting the system if there is no Camera
        if self._camera is None:
            message = (
                f"Unable to start the VideoSystem with id {self._id}. The VideoSystem must be equipped with a Camera "
                f"before it can be started. Use add_camera() method to add a Camera class to the VideoSystem. If you "
                f"need to convert a directory of images to video, use the encode_video_from_images() method instead."
            )
            console.error(error=RuntimeError, message=message)

        # If the camera is configured to save frames, ensures it is matched with a Saver class instance.
        elif self._camera.save_frames and self._saver is None:
            message = (
                f"Unable to start the VideoSystem with id {self._id}. The managed Camera is configured to save frames "
                f"and has to be matched to a Saver instance, but no Saver was added. Use add_image_saver() or "
                f"add_video_saver() method to add a Saver instance to save camera frames."
            )
            console.error(error=RuntimeError, message=message)

        # Instantiates an array shared between all processes. This array is used to control all child processes.
        # Index 0 (element 1) is used to issue global process termination command
        # Index 1 (element 2) is used to flexibly enable or disable saving camera frames.
        # Index 2 (element 3) is used to flexibly enable or disable outputting camera frames to other processes.
        # Index 3 (element 4) is used to track VideoSystem initialization status.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._id}_terminator_array",  # Uses class id with an additional specifier
            prototype=np.zeros(shape=4, dtype=np.uint8),
            exist_ok=True,  # Automatically recreates the buffer if it already exists
        )  # Instantiation automatically connects the main process to the array.

        # Only starts the consumer process if the managed camera is configured to save frames.
        if self._saver is not None:
            # Starts the consumer process first to minimize queue buildup once the producer process is initialized.
            # Technically, due to saving frames being initially disabled, queue buildup is no longer a major concern,
            # but this safety-oriented initialization order is still preserved.
            self._consumer_process = Process(
                target=self._frame_saving_loop,
                args=(
                    self._id,
                    self._saver,
                    self._camera,
                    self._image_queue,
                    self._logger_queue,
                    self._terminator_array,
                ),
                daemon=True,
            )
            self._consumer_process.start()

            # Waits for the process to report that it has successfully initialized.
            initialization_timer.reset()
            while self._terminator_array.read_data(index=3) != 2:  # pragma: no cover
                # If the process takes too long to initialize or dies, raises an error.
                if initialization_timer.elapsed > 10 or not self._consumer_process.is_alive():
                    message = (
                        f"Unable to start the VideoSystem with id {self._id}. The consumer process has "
                        f"unexpectedly shut down or stalled for more than 10 seconds during initialization. This "
                        f"likely indicates a problem with the Saver (Video or Image) class managed by the process."
                    )

                    # Reclaims all committed resources before terminating with an error.
                    self._terminator_array.write_data(index=0, data=np.uint8(1))
                    self._consumer_process.join()
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()

                    console.error(error=RuntimeError, message=message)

        # Starts the producer process
        self._producer_process = Process(
            target=self._frame_production_loop,
            args=(
                self._id,
                self._camera,
                self._image_queue,
                self._output_queue,
                self._logger_queue,
                self._terminator_array,
            ),
            daemon=True,
        )
        self._producer_process.start()

        # Waits for the process to report that it has successfully initialized.
        initialization_timer.reset()
        while self._terminator_array.read_data(index=3) != 1:  # pragma: no cover
            # If the process takes too long to initialize or dies, raises an error.
            if initialization_timer.elapsed > 10 or not self._producer_process.is_alive():
                message = (
                    f"Unable to start the VideoSystem with id {self._id}. The producer process has "
                    f"unexpectedly shut down or stalled for more than 10 seconds during initialization. This likely "
                    f"indicates a problem with initializing the Camera class controlled by the process or the frame "
                    f"display thread."
                )

                # Reclaims all committed resources before terminating with an error.
                self._terminator_array.write_data(index=0, data=np.uint8(1))
                if self._consumer_process is not None:
                    self._consumer_process.join()
                self._producer_process.join()
                self._terminator_array.disconnect()
                self._terminator_array.destroy()

                console.error(error=RuntimeError, message=message)

        # Creates ands tarts the watchdog thread
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        # Sets the _started flag, which also activates watchdog monitoring.
        self._started = True

    def stop(self) -> None:
        """Stops the producer and consumer processes and releases all resources.

        The instance will be kept alive until all frames buffered to the image_queue are saved. This is an intentional
        security feature that prevents information loss.

        Notes:
            This method waits for at most 10 minutes for the output_queue and the image_queue to become empty. If the
            queues are not empty by that time, the method forcibly terminates the instance and discards any unprocessed
            data.
        """
        # This timer is used to forcibly terminate the process that gets stuck in the shutdown sequence.
        shutdown_timer = PrecisionTimer(precision="s")

        # Ensures that the stop procedure is only executed if the class is running
        if not self._started or self._terminator_array is None:
            # Terminator array cannot be None if the process has started, so this check is to appease mypy.
            return

        # This inactivates the watchdog thread monitoring, ensuring it does not err when the processes are terminated.
        self._started = False

        # Sets the global shutdown flag to true
        self._terminator_array.write_data(index=0, data=np.uint8(1))

        # Delays for 2 seconds to allow the consumer process to terminate its runtime
        shutdown_timer.delay_noblock(delay=2)

        # Waits until both multiprocessing queues made by the instance are empty. This is aborted if the shutdown
        # stalls at this step for 10+ minutes
        while not self._image_queue.empty() or self._output_queue.empty():
            # Prevents being stuck in this loop.
            if shutdown_timer.elapsed > 600:
                break

        # Joins the producer and consumer processes
        if self._producer_process is not None:
            self._producer_process.join(timeout=20)
        if self._consumer_process is not None:
            self._consumer_process.join(timeout=20)

        # Joins the watchdog thread
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=20)

        # Disconnects from and destroys the terminator array buffer. This step destroys the shared memory buffer.
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

    @property
    def started(self) -> bool:
        """Returns true if the system has been started and has active daemon processes connected to cameras and
        saver.
        """
        return self._started

    @property
    def system_id(self) -> np.uint8:
        """Returns the unique identifier code assigned to the VideoSystem class instance."""
        return self._id

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing Queue object used by the system's producer process to send frames to other
        concurrently active processes.
        """
        return self._output_queue

    @staticmethod
    def get_opencv_ids() -> tuple[str, ...]:
        """Discovers and reports IDs and descriptive information about cameras accessible through the OpenCV library.

        This method can be used to discover camera IDs accessible through the OpenCV backend. Next, each of the IDs can
        be used via the add_camera() method to add the specific camera to a VideoSystem instance.

        Notes:
            Currently, there is no way to get serial numbers or usb port names from OpenCV. Therefore, while this method
            tries to provide some ID information, it likely will not be enough to identify the cameras. Instead, it is
            advised to test each of the IDs with the 'interactive-run' CLI command to manually map IDs to cameras based
            on the produced visual stream.

            This method works by sequentially evaluating camera IDs starting from 0 and up to ID 100. The method
            connects to each camera and takes a test image to ensure the camera is accessible, and it should ONLY be
            called when no OpenCVCamera or any other OpenCV-based connection is active. The evaluation sequence will
            stop early if it encounters more than five non-functional IDs in a row.

        Returns:
             A tuple of strings. Each string contains camera ID, frame width, frame height, and camera fps value.
        """
        # Disables OpenCV error logging temporarily
        prev_log_level = cv2.getLogLevel()
        cv2.setLogLevel(0)

        try:
            non_working_count = 0
            working_ids = []

            # This loop will keep iterating over IDs until it discovers 5 non-working IDs. The loop is designed to
            # evaluate 100 IDs at maximum to prevent infinite execution.
            for evaluated_id in range(100):
                # Evaluates each ID by instantiating a video-capture object and reading one image and dimension data
                # from the connected camera (if any was connected).
                camera = cv2.VideoCapture(evaluated_id)

                # If the evaluated camera can be connected and returns images, it's ID is appended to the ID list
                if camera.isOpened() and camera.read()[0]:
                    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = camera.get(cv2.CAP_PROP_FPS)
                    descriptive_string = (
                        f"OpenCV Camera ID: {evaluated_id}, Width: {width}, Height: {height}, FPS: {fps}."
                    )
                    working_ids.append(descriptive_string)
                    non_working_count = 0  # Resets non-working count whenever a working camera is found.
                else:
                    non_working_count += 1

                # Breaks the loop early if more than 5 non-working IDs are found consecutively
                if non_working_count >= 5:
                    break

                camera.release()  # Releases the camera object to recreate it above for the next cycle

            return tuple(working_ids)  # Converts to tuple before returning to caller.

        finally:
            # Restores previous log level
            cv2.setLogLevel(prev_log_level)

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

        # Resets the harvester instance after retrieving discoverable IDs
        harvester.remove_file(file_path=str(cti_path))
        harvester.reset()

        return tuple(working_ids)  # Converts to tuple before returning to caller.

    @staticmethod
    def _frame_display_loop(
        display_queue: Queue,  # type: ignore
        camera_id: int,
    ) -> None:  # pragma: no cover
        """Continuously fetches frame images from display_queue and displays them via the OpenCV imshow () method.

        This method runs in a thread as part of the _produce_images_loop() runtime. It is used to display
        frames as they are grabbed from the camera and passed to the multiprocessing queue. This allows visually
        inspecting the frames as they are processed.

        Notes:
            Since the method uses OpenCV under-the-hood, it repeatedly releases GIL as it runs. This makes it
            beneficial to have this functionality as a sub-thread instead of realizing it at the same level as the
            rest of the image production loop code.

            This thread runs until it is terminated through the display window GUI or passing a non-NumPy-array
            object (e.g.: integer -1) through the display_queue.

        Args:
            display_queue: A multithreading Queue object that is used to buffer grabbed frames to de-couple display from
                acquisition. It is expected that the queue yields frames as NumPy ndarray objects. If the queue yields a
                non-array object, the thread terminates.
            camera_id: The unique ID of the camera which produces displayed images. This is used to generate a
                descriptive window name for the display GUI. Currently, this ID is the same as the VideoSystem id
        """
        # Initializes the display window using 'normal' mode to support user-controlled resizing.
        window_name = f"VideoSystem {camera_id} Frames."
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
        cv2.destroyAllWindows()

    @staticmethod
    def _frame_production_loop(
        video_system_id: np.uint8,
        camera_system: _CameraSystem,
        image_queue: MPQueue,  # type: ignore
        output_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
    ) -> None:  # pragma: no cover
        """Continuously grabs frames from the input camera and queues them up to be saved by the consumer process.

        This method loops while the first element in terminator_array (index 0) is zero. It continuously grabs
        frames from the camera but only queues them up to be saved by the consumer process if the second
        element in terminator_array (index 1) is not zero.

        This method also displays the acquired frames for the cameras that request this functionality in a separate
        display thread.

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
            camera_system: The CameraSystem class that stores the managed camera with some additional parameters.
            image_queue: A multiprocessing queue that buffers and pipes acquired frames to the consumer process.
            output_queue: A multiprocessing queue that buffers and pipes acquired frames to other concurrently active
                processes (not managed by this VideoSystem instance).
            logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
                This queue is used to buffer and pipe data to be logged to the logger cores.
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
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)  # type: ignore
        stamp_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # Sends the onset data to the logger queue. The time_stamp of 0 is universally interpreted as the timer
        # onset. The serialized data has to be the byte-converted onset date and time using UTC timezone.
        package = LogPackage(source_id=video_system_id, time_stamp=np.uint64(0), serialized_data=onset)
        logger_queue.put(package)

        # Extracts the Camera class from the CameraSystem wrapper
        camera = camera_system.camera

        # If the camera is configured to display frames, creates a worker thread and queue object that handles
        # displaying the frames.
        show_time = None
        show_timer = None
        display_queue = None
        display_thread = None
        if camera_system.display_frames:
            # Creates queue and thread for the camera
            display_queue = Queue()
            display_thread = Thread(target=VideoSystem._frame_display_loop, args=(display_queue, video_system_id))
            display_thread.start()

            # Converts the timeout between showing two consecutive frames from frames_per_second to
            # microseconds_per_frame. For this, first divides one second by the number of frames (to get seconds per
            # frame) and then translates that into microseconds. This gives the timeout between two consecutive
            # frame acquisitions.
            # noinspection PyTypeChecker
            show_time = convert_time(
                time=float(1 / camera_system.display_frame_rate),
                from_units="s",
                to_units="us",
                convert_output=True,
            )
            show_timer = PrecisionTimer("us")

        # If the camera cannot produce the required fps natively, but it is possible to subsample the fps via software,
        # sets up the necessary assets
        frame_time = None
        frame_timer = None
        if camera_system.save_frames and camera_system.fps_override > 0:
            # noinspection PyTypeChecker
            frame_time = convert_time(
                time=float(1 / camera_system.fps_override),
                from_units="s",
                to_units="us",
                convert_output=True,
            )
            frame_timer = PrecisionTimer("us")

        # Calculates the timeout in microseconds per frame for outputting the frames to other processes if the camera
        # supports this functionality.
        output_time = None
        output_timer = None
        if camera_system.output_frames:
            # noinspection PyTypeChecker
            output_time = convert_time(
                time=float(1 / camera_system.output_frame_rate),
                from_units="s",
                to_units="us",
                convert_output=True,
            )
            output_timer = PrecisionTimer("us")

        camera.connect()  # Connects to the hardware of the camera.

        # Sets the 3d index value of the terminator_array to 1 to indicate that all CameraSystems have been started.
        terminator_array.write_data(index=3, data=np.uint8(1))

        try:
            # The loop runs until the VideoSystem is terminated by setting the first element (index 0) of the array to 1
            while not terminator_array.read_data(index=0):
                # Grabs the first available frame as a numpy ndarray. For some backends, this blocks until the frame is
                # available if it is called to early. For other backends, it returns the same frame as grabbed during
                # the previous call.
                frame = camera.grab_frame()
                frame_stamp = stamp_timer.elapsed  # Generates the time-stamp for the acquired frame

                # If the camera is configured to display acquired frames, queues each frame to be displayed. This
                # does not depend on whether the frame is buffered for saving.
                if display_queue is not None and show_timer.elapsed >= show_time:
                    show_timer.reset()  # Resets the display timer
                    display_queue.put(frame)

                # If the software framerate override is enabled, this loop is further limited to acquire frames at
                # the specified rate, which is helpful for some cameras that do not have a built-in acquisition
                # control functionality. If the acquisition timeout has not passed and is enabled, skips the rest
                # of the processing runtime
                if camera_system.fps_override != 0 and frame_timer.elapsed >= frame_time:  # type: ignore
                    continue
                elif camera_system.fps_override != 0:
                    # Resets the frame acquisition timer before processing the frame so that the wait time for the next
                    # frame is 'absorbed' into processing the frame.
                    frame_timer.reset()  # type: ignore

                # Bundles frame data and acquisition timestamp relative to the onset and passes them
                # to the multiprocessing Queue that delivers the data to the consumer process that saves it to disk.
                # This is only executed if frame saving is enabled via the terminator_array variable using index 1
                # (globally) and if the specific Camera instance is configured to save frames at all.
                if terminator_array.read_data(index=1) == 1 and camera_system.save_frames:
                    image_queue.put((frame, frame_stamp))

                # If the camera is configured to output frame data via the output_queue and the necessary conditions
                # are fulfilled, sends the frame data to the output_queue.
                if (
                    terminator_array.read_data(index=2) == 1
                    and camera_system.output_frames
                    and output_timer.elapsed >= output_time  # type: ignore
                ):
                    # Resets the output timer
                    output_timer.reset()  # type: ignore
                    output_queue.put(frame)

        except Exception as e:
            # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
            # before re-raising the exception to terminate the process.
            sys.stderr.write(str(f"Error in the VideoSystem {video_system_id} producer Proces: {e}"))
            sys.stderr.flush()
            raise e

        # Ensures proper cleanup even if the loop runs into an error
        finally:
            # Disconnects from the shared memory array
            terminator_array.disconnect()
            camera.disconnect()  # Disconnects from the camera

            # Releases all resources and terminates the Process.
            # Terminates the display thread
            if display_queue is not None:
                display_queue.put(None)
            # Waits for the thread to close
            if display_thread is not None:
                display_thread.join()

    @staticmethod
    def _frame_saving_loop(
        video_system_id: np.uint8,
        saver: ImageSaver | VideoSaver,
        camera_system: _CameraSystem,
        image_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
    ) -> None:  # pragma: no cover
        """Continuously grabs frames from the image_queue and saves them as standalone images or video file, depending
        on the input saver instance.

        This method loops while the first element in terminator_array (index 0) is nonzero. It continuously grabs
        frames from image_queue and uses the saver instance to write them to disk.

        This method also logs the acquisition time for each saved frame via the logger_queue instance.

        Notes:
            This method's main loop will be kept alive until the image_queue is empty. This is an intentional security
            feature that ensures all buffered images are processed before the saver is terminated. To override this
            behavior, you will need to use the process kill command, but it is strongly advised not to tamper
            with this feature.

            This method expects that image_queue buffers 2-element tuples that include frame data and frame acquisition
            time relative to the onset point in microseconds.

        Args:
            video_system_id: The unique byte-code identifier of the VideoSystem instance. This is used to identify the
                VideoSystem when logging data.
            saver: The VideoSaver or ImageSaver instance to use for saving frames.
            camera_system: The CameraSystem instance that stores the camera that will be acquiring frames.
            image_queue: A multiprocessing queue that buffers and pipes acquired frames to the consumer process.
            logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
                This queue is used to buffer and pipe data to be logged to the logger cores.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
        """
        # Connects to the terminator array used to manage the loop runtime
        terminator_array.connect()

        # Both saver classes have an additional setup process that has to be carried out before frames can be submitted
        # for saving,
        if isinstance(saver, VideoSaver):
            # For video saver, uses camera data to initialize the video container
            saver.create_live_video_encoder(
                frame_width=camera_system.camera.width,  # type: ignore
                frame_height=camera_system.camera.height,  # type: ignore
                video_id=f"{video_system_id:03d}",  # Uses the index of the video system with padding
                video_frames_per_second=camera_system.camera.fps,  # type: ignore
            )

        # For ImageSavers, the only setup consists of calling the 'live' image saver creation method to initialize
        # the threads and local queue objects.
        else:
            saver.create_live_image_saver()

        # Sets the 3d index value of the terminator_array to 2 to indicate that all SaverSystems have been started.
        terminator_array.write_data(index=3, data=np.uint8(2))

        data_placeholder = np.array([], dtype=np.uint8)

        try:
            # This loop will run until the global shutdown command is issued (via variable under index 0) and until the
            # image_queue is empty.
            while not terminator_array.read_data(index=0, convert_output=True) or not image_queue.empty():
                # Grabs the image bundled with its acquisition time from the queue and passes it to Saver class.
                try:
                    frame: NDArray[Any]
                    frame_time: int
                    frame, frame_time = image_queue.get_nowait()
                except Exception:
                    # Cycles the loop if the queue is empty
                    continue

                # Sends the frame to be saved by the saver
                saver.save_frame(frame)  # Same API for Image and Video savers.

                # Logs frame acquisition data. For this, uses an empty numpy array as payload, as we only care about the
                # acquisition timestamps at this time.
                package = LogPackage(
                    video_system_id,
                    time_stamp=np.uint64(frame_time),
                    serialized_data=np.array(object=data_placeholder, dtype=np.uint8),
                )
                logger_queue.put(package)

        except Exception as e:
            # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
            # before re-raising the exception to terminate the process.
            sys.stderr.write(str(f"Error in the VideoSystem {video_system_id} consumer Proces: {e}"))
            sys.stderr.flush()
            raise e

        # Ensures proper resource cleanup even during error shutdowns
        finally:
            # Disconnects from the shared memory array
            terminator_array.disconnect()

            # Carries out the necessary shut-down procedures:
            if isinstance(saver, VideoSaver):
                saver.terminate_live_encoder()
            else:
                saver.terminate_image_saver()

    def _watchdog(self) -> None:  # pragma: no cover
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

            # The watchdog functionality only kicks-in after the VideoSystem has been started
            if not self._started:
                continue

            # Checks if producer is alive
            error = False
            producer = False
            if self._producer_process is not None and not self._producer_process.is_alive():
                error = True
                producer = True

            # Checks if Consumer is alive
            if self._consumer_process is not None and not self._consumer_process.is_alive():
                error = True

            # If either consumer or producer is dead, ensures proper resource reclamation before terminating with an
            # error
            if error:
                # Reclaims all committed resources before terminating with an error.
                self._terminator_array.write_data(index=0, data=np.uint8(1))  # type: ignore
                if self._consumer_process is not None:
                    self._consumer_process.join()
                if self._producer_process is not None:
                    self._producer_process.join()
                if self._terminator_array is not None:
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()
                self._started = False  # The code above is equivalent to stopping the instance

                if producer:
                    message = (
                        f"The producer process for the VideoSystem with id {self._id} has been prematurely "
                        f"shut down. This likely indicates that the process has encountered a runtime error that "
                        f"terminated the process."
                    )
                    console.error(message=message, error=RuntimeError)

                else:
                    message = (
                        f"The consumer process for the VideoSystem with id {self._id} has been prematurely "
                        f"shut down. This likely indicates that the process has encountered a runtime error that "
                        f"terminated the process."
                    )
                    console.error(message=message, error=RuntimeError)

    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames.

        Does not interfere with grabbing and displaying the frames to the user, this process is only stopped when the
        main stop() method is called.
        """
        if self._started and self._terminator_array is not None:
            # noinspection PyTypeChecker
            self._terminator_array.write_data(index=1, data=0)

    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames.

        The frames are grabbed and (optionally) displayed to the user after the main start() method is called, but they
        are not initially saved to disk. The call to this method additionally enables saving the frames to disk
        """
        if self._started and self._terminator_array is not None:
            # noinspection PyTypeChecker
            self._terminator_array.write_data(index=1, data=1)

    def stop_frame_output(self) -> None:
        """Disables outputting frame data via the instance output_queue."""
        if self._started and self._terminator_array is not None:
            # noinspection PyTypeChecker
            self._terminator_array.write_data(index=2, data=0)

    def start_frame_output(self) -> None:
        """Enables outputting frame data via the instance output_queue.

        Some cameras can be configured to additionally share acquired frame data with other concurrently active
        processes. When the VideoSystem starts, this functionality is not enabled by default and has to be enabled
        separately via this method.
        """
        if self._started and self._terminator_array is not None:
            # noinspection PyTypeChecker
            self._terminator_array.write_data(index=2, data=1)

    def extract_logged_data(self) -> tuple[np.uint64, ...]:
        """Extracts the frame acquisition timestamps from the .npz log archive generated by the VideoSystem instance
        during runtime

        This method reads the compressed '.npz' archives generated by the VideoSystem and, if the system saves any
        frames acquired by the camera, extracts a tuple of acquisition timestamps. The order of timestamps in
        the tuple is sequential and matches the order of frame acquisition.

        Notes:
            This method should be used as a convenience abstraction for the inner workings of the DataLogger class that
            decodes frame timestamp data from log files for further user-defined processing.

            Since version 1.1.0 this function is largely a wrapper around the multiprocessing-safe global
            'extract_logged_video_system_data' function exposed by the library. The extraction logic did not change, but
            having an instance-independent extraction function helps parallelize the data acquired by many sources that
            use Ataraxis code.


        Returns:
            A tuple that stores the frame acquisition timestamps, where each timestamp is a 64-bit unsigned numpy
            integer and specifies the number of microseconds since the UTC epoch onset.

        Raises:
            ValueError: If the .npz archive for the VideoSystem instance does not exist.
        """

        # Generates the log file path using the video system ID. Assumes that the log has been compressed to the .npz
        # format before calling this method
        log_path = self._log_directory.joinpath(f"{self._id}_log.npz")

        # If a compressed log archive does not exist, raises an error
        if not log_path.exists():
            error_message = (
                f"Unable to extract frame data for VideoSystem with id {self._id} from the log file. "
                f"This likely indicates that the logs have not been compressed via DataLogger's compress_logs() method "
                f"and are not available for processing. Call log compression method before calling this method."
            )
            console.error(message=error_message, error=ValueError)

        # After verifying that the compressed log archive exists, it is safe to call the global function that contains
        # the extraction logic. This makes this method a wrapper for the multiprocessing-safe extraction function that
        # can be used directly.
        return extract_logged_video_system_data(log_path=log_path)

    def encode_video_from_images(
        self, directory: Path, target_fps: int, video_name: str, cleanup: bool = False
    ) -> None:
        """Converts a set of images acquired via the ImageSaver class into a video file.

        Use this method to post-process image frames acquired in real time into a storage-efficient video file.
        Primarily, this is helpful for runtimes that require the fastest possible saving speed (achieved by saving raw
        frame data without processing) but still need to optimize acquired data for long-term storage.

        Notes:
            The video is written to the output directory of the VideoSystem class, regardless of where the source images
            are stored.

            The dimensions of the video are determined automatically from the first image passed to the encoder. The
            method expects all frames in the directory to have the same dimensions.

        Args:
            directory: The directory where the images are stored.
            target_fps: The framerate to use for the created video.
            video_name: The ID or name to use for the generated video file. Do not include the extension in this name,
                as it will be resolved and appended automatically, based on the VideoSaver configuration.
            cleanup: Determines whether to clean up (delete) source images after the video creation. The cleanup is
                only carried out after the FFMPEG process terminates with a success code. Make sure to test your
                pipeline before enabling this option, as this method does not verify the encoded video for corruption.

        Raises:
            TypeError: If the Saver managed by the VideoSystem is not a VideoSaver.
            Exception: If there are no images with supported file-extensions in the specified directory.
        """

        # Prevents running this method if the saver is not a VideoSaver
        if not isinstance(self._saver, VideoSaver):
            message = (
                f"The VideoSystem with ID {self._id} is unable to encode a directory of images as a video file. The "
                f"VideoSystem requires a VideoSaver class to generate video files. Call the add_video_saver() method "
                f"before calling this method()."
            )
            console.error(message=message, error=TypeError)

        # Calls the conversion method.
        self._saver.create_video_from_image_folder(  # type: ignore
            image_directory=directory, video_frames_per_second=target_fps, video_id=video_name, cleanup=cleanup
        )

    @property
    def output_directory(self) -> Path | None:
        """Returns the path to the directory where the Saver managed by the VideoSystem outputs acquired frames as
        images or video file.

        If the VideoSystem does not have a Saver, it returns None to indicate that there is no valid output directory.
        """
        if self._saver is not None:
            # noinspection PyProtectedMember
            return self._saver._output_directory
        else:
            return None

    @property
    def log_path(self) -> Path:
        """Returns the path to the compressed .npz log archive that would be generated for the VideoSystem by the
        DataLogger instance given to the class at initialization.

        Primarily, this path should be used as an argument to the instance-independent
        'extract_logged_video_system_data' data extraction function.
        """
        return self._log_directory.joinpath(f"{self._id}_log.npz")


def extract_logged_video_system_data(log_path: Path) -> tuple[np.uint64, ...]:
    """Extracts the frame acquisition timestamps from the .npz log archive generated by the VideoSystem instance
    during runtime.

    This function reads the '.npz' archive generated by the DataLogger 'compress_logs' method for a VideoSystem
    instance and, if the system saved any frames acquired by the camera, extracts the tuple of frame timestamps.
    The order of timestamps in the tuple is sequential and matches the order of frame acquisition, and the timestamps
    are given as microseconds elapsed since the UTC epoch onset.

    This function is process- and thread-safe and can be pickled. It is specifically designed to be executed in-parallel
    for many concurrently used VideoSystems, but it can also be used standalone. If you have an initialized
    VideoSystem instance, it is recommended to use its 'extract_logged_data' method instead, as it automatically
    resolves the log_path argument.

    Notes:
        This function should be used as a convenience abstraction for the inner workings of the DataLogger class that
        decodes frame timestamp data from log files for further user-defined processing.

        The function assumes that it is given an .npz archive generated for a VideoSystem instance and WILL behave
        unexpectedly if it is instead given an archive generated by another Ataraxis class, such as
        MicroControllerInterface.

    Args:
        log_path: The path to the .npz archive file that stores the logged data generated by the VideoSystem
            instance during runtime.

    Returns:
        A tuple that stores the frame acquisition timestamps, where each timestamp is a 64-bit unsigned numpy
        integer and specifies the number of microseconds since the UTC epoch onset.

    Raises:
        ValueError: If the .npz archive for the VideoSystem instance does not exist.
    """
    # If a compressed log archive does not exist, raises an error
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        error_message = (
            f"Unable to extract VideoSystem frame timestamp data from the log file {log_path}. "
            f"This likely indicates that the logs have not been compressed via DataLogger's compress_logs() method "
            f"and are not available for processing. Call log compression method before calling this method. Valid "
            f"'log_path' arguments must point to an .npz archive file."
        )
        console.error(message=error_message, error=ValueError)

    # Loads the archive into RAM
    archive: NpzFile = np.load(file=log_path)

    # Precreates the list to store the extracted data.
    frame_data = []

    # Locates the logging onset timestamp. The onset is used to convert the timestamps for logged frame data into
    # absolute UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in
    # microseconds relative to the onset timestamp.
    timestamp_offset = 0
    onset_us = np.uint64(0)
    timestamp: np.uint64
    for number, item in enumerate(archive.files):
        message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

        # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
        # message starting at index 1. If timestamp value is 0, the message contains the onset timestamp value
        # stored as 8-byte payload. Index 0 stores the source ID (uint8 value)
        if np.uint64(message[1:9].view(np.uint64)[0]) == 0:
            # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
            onset_us = np.uint64(message[9:].view("<i8")[0].copy())

            # Breaks the loop onc the onset is found. Generally, the onset is expected to be found very early into
            # the loop
            timestamp_offset = number  # Records the item number at which the onset value was found.
            break

    # Once the onset has been discovered, loops over all remaining messages and extracts frame data.
    for item in archive.files[timestamp_offset + 1 :]:
        message = archive[item]

        # Extracts the elapsed microseconds since timestamp and uses it to calculate the global timestamp for the
        # message, in microseconds since epoch onset.
        elapsed_microseconds = np.uint64(message[1:9].view(np.uint64)[0].copy())
        timestamp = onset_us + elapsed_microseconds

        # Iteratively fills the list with data. Frame stamp messages do not have a payload, they only contain the
        # source ID and the acquisition timestamp. This gives them the length of 9 bytes.
        if len(message) == 9:
            frame_data.append(timestamp)

    # Closes the archive to free up memory
    archive.close()

    # Returns the extracted data
    return tuple(frame_data)
