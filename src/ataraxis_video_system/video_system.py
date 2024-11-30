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
import datetime
from datetime import timezone
from threading import Thread
import subprocess
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
    ProcessError,
)
from multiprocessing.managers import SyncManager
from dataclasses import dataclass

import cv2
import numpy as np
from ataraxis_time import PrecisionTimer
from harvesters.core import Harvester  # type: ignore
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray
from ataraxis_time.time_helpers import convert_time

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
    """Stores a Camera class instance managed by a VideoSystem class, alongside additional runtime parameters.

    This class is used as a container that aggregates all objects and parameters required by the VideoSystem to
    interface with a camera.
    """

    """Stores the managed camera interface class."""
    camera: OpenCVCamera | HarvestersCamera | MockCamera

    """Determines whether to override (sub-sample) the camera's frame acquisition rate. The override has to be smaller 
    than the camera's native frame rate and can be used to more precisely control the frame rate of cameras that do 
    not support real-time frame rate control."""
    fps_override: int | float

    """Determines whether to display acquired camera frames to the user via a display UI. The frames are always 
    displayed at a 30 fps rate regardless of the actual frame rate of the camera. Note, this does not interfere with 
    frame acquisition (saving)."""
    display_frames: bool

    """Determines whether acquired frames need to be piped to other processes via the output queue, in addition to 
    being sent to the saver process (if any). This is used to additionally process the frames in-parallel with 
    saving them to disk, for example, to analyze the visual stream data for certain trigger events."""
    output_frames: bool


@dataclass(frozen=True)
class _SaverSystem:
    """Stores a Saver class instance managed by a VideoSystem class, alongside additional runtime parameters.

    This class is used as a container that aggregates all objects and parameters required by the VideoSystem to
    interface with a saver.
    """

    """Stores the managed saver interface class."""
    saver: ImageSaver | VideoSaver

    """Stores the indices of camera objects whose frames will be saved by the included saver instance. The indices 
    have to match the camera object indices inside the _cameras attribute of the VideoSystem instance that manages 
    this SaverSystem."""
    source_ids: list[int]


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
            log files and generated video files.
        system_name: A human-readable name used to identify the VideoSystem instance in error messages.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class via its 'input_queue' property.
            This queue is used to buffer and pipe data to be logged to the logger cores.
        system_description: A brief description of the VideoSystem instance. This is used when creating the log file
            that stores class runtime parameters and maps id-codes to meaningful names and descriptions to support
            future data processing.

    Attributes:
        _id: Stores the ID code of the VideoSystem instance.
        _name: Stores the human-readable name of the VideoSystem instance.
        _description: Stores the description of the VideoSystem instance.
        _logger_queue: Stores the multiprocessing Queue object used to buffer and pipe data to be logged.
        _cameras: Stores managed CameraSystems.
        _savers: Stores managed SaverSystems.
        _started: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores a SyncManager class instance from which the image_queue and the log file Lock are derived.
        _image_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from producer to
            consumer processes.
        _terminator_array: A SharedMemoryArray instance that provides a cross-process communication interface used to
            manage runtime behavior of spawned processes.
        _producer_process: A process that acquires camera frames using managed CameraSystems.
        _consumer_process: A process that saves the acquired frames using managed SaverSystems.
        _watchdog_thread: A thread used to monitor the runtime status of the remote consumer and producer processes.

    Raises:
        TypeError: If any of the provided arguments has an invalid type.
    """

    def __init__(
        self, system_id: np.uint8, system_name: str, logger_queue: MPQueue, system_description=""  # type: ignore
    ):
        # Resolves the system-name first, to use it in further error messages
        if not isinstance(system_name, str):
            raise TypeError(
                f"Unable to initialize the VideoSystem class instance. Expected a string for system_name, but got "
                f"{system_name} of type {type(system_name).__name__}."
            )

        # Ensures system_id is a byte-convertible integer
        if not isinstance(system_id, np.uint8):
            raise TypeError(
                f"Unable to initialize the {system_name} VideoSystem class instance. Expected a uint8 system_id, but "
                f"encountered {system_id} of type {type(system_id).__name__}."
            )

        # Ensures invalid descriptions are converted to an empty string.
        system_description = system_description if isinstance(system_description, str) else ""

        # Saves ID data and the logger queue to class attributes
        self._id: np.uint8 = system_id
        self._name: str = system_name
        self._description = system_description
        self._logger_queue: MPQueue = logger_queue  # type: ignore

        # Initializes placeholder variables that will be filled by add_camera() and add_saver() methods.
        self._cameras: list[_CameraSystem] = []
        self._savers: list[_SaverSystem] = []

        self._started: bool = False  # Tracks whether the system has active processes

        # Sets up the assets used to manage acquisition and saver processes. The assets are configured during the
        # start() method runtime, most of them are initialized to placeholder values here.
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._image_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: SharedMemoryArray | None = None
        self._producer_process: Process | None = None
        self._consumer_process: Process | None = None
        self._watchdog_thread: None | Thread = None

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
        camera_backend: CameraBackends = CameraBackends.OPENCV,
        camera_id: int = 0,
        display_frames: bool = False,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        frames_per_second: Optional[int | float] = None,
        fps_override: Optional[int | float] = None,
        opencv_backend: Optional[int] = None,
        cti_path: Optional[Path] = None,
        color: Optional[bool] = None,
    ) -> None:
        """Creates and returns a Camera class instance that uses the specified camera backend.

        This method centralizes Camera class instantiation. It contains methods for verifying the input information
        and instantiating the specialized Camera class based on the requested camera backend. All Camera classes from
        this library have to be initialized using this method.

        Notes:
            While the method contains many arguments that allow to flexibly configure the instantiated camera, the only
            crucial ones are camera name, backend, and the numeric ID of the camera. Everything else is automatically
            queried from the camera, unless provided.

        Args:
            camera_name: The string-name of the camera. This is used to help identify the camera and to mark all
                frames acquired from this camera.
            camera_id: The numeric ID of the camera, relative to all available video devices, e.g.: 0 for the first
                available camera, 1 for the second, etc. Generally, the cameras are ordered based on the order they
                were connected to the host system.
            camera_backend: The backend to use for the camera class. Currently, all supported backends are derived from
                the CameraBackends enumeration. The preferred backend is 'Harvesters', but we also support OpenCV for
                non-GenTL-compatible cameras.
            frame_width: The desired width of the camera frames to acquire, in pixels. This will be passed to the
                camera and will only be respected if the camera has the capacity to alter acquired frame resolution.
                If not provided (set to None), this parameter will be obtained from the connected camera.
            frame_height: Same as width, but specifies the desired height of the camera frames to acquire, in pixels.
                If not provided (set to None), this parameter will be obtained from the connected camera.
            frames_per_second: The desired Frames Per Second to capture the frames at. Note, this depends on the
                hardware capabilities OF the camera and is affected by multiple related parameters, such as image
                dimensions, camera buffer size, and the communication interface. If not provided (set to None), this
                parameter will be obtained from the connected camera.
            fps_override: The number of frames to grab from the camera per second. This argument allows optionally
                overriding the frames per second (fps) parameter of the Camera class. When provided, the frame
                acquisition process will only trigger the frame grab procedure at the specified interval. Generally,
                this override should only be used for cameras with a fixed framerate or cameras that do not have
                framerate control capability at all. For cameras that support onboard framerate control, setting the
                fps through the Camera class will almost always be better. The override will not be able to exceed the
                acquisition speed enforced by the camera hardware.
            opencv_backend: Optional. The integer-code for the specific acquisition backend (library) OpenCV should
                use to interface with the camera. Generally, it is advised not to change the default value of this
                argument unless you know what you are doing.
            cti_path: The path to the '.cti' file that provides the GenTL Producer interface. It is recommended to use
                the file supplied by your camera vendor if possible, but a general Producer, such as mvImpactAcquire,
                would work as well. See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more
                details. Note, cti_path is only necessary for Harvesters backend, but it is REQUIRED for that backend.
            color: A boolean indicating whether the camera acquires colored or monochrome images. This is
                used by OpenCVCamera to optimize acquired images depending on the source (camera) color space. It is
                also used by the MockCamera to enable simulating monochrome and colored images.
            display_frames: Determines whether to display acquired frames to the user. This allows visually monitoring
                the camera feed in real time, which is frequently desirable in scientific experiments.

        Raises:
            TypeError: If the input arguments are not of the correct type.
            ValueError: If the requested camera_backend is not one of the supported backends. If the input cti_path does
                not point to a '.cti' file.
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
                f"Unable to add the {camera_name} object to the {self._name} VideoSystem. Expected an integer for "
                f"camera_id argument, but got {camera_id} of type {type(camera_id).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frames_per_second, (int, float, NoneType)):
            message = (
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected an integer, "
                f"float or None for frames_per_second argument, but got {frames_per_second} of type "
                f"{type(frames_per_second).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if fps_override is not None and fps_override < 1:
            raise ValueError(
                f"Unable to add the {camera_name} Camera object to the {self._name} VideoSystem. Expected a positive "
                f"integer or None for fps_override, but got {fps_override} of type {type(fps_override).__name__}."
            )
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

        # Ensures that display_frames is either True or False.
        display_frames = True if display_frames is not None else False

        # Converts integer frames_per_second inputs to floats, since the Camera classes expect it to be a float.
        if isinstance(frames_per_second, int):
            frames_per_second = float(frames_per_second)

        # OpenCVCamera
        if camera_backend == CameraBackends.OPENCV:
            # If backend preference is None, uses generic preference
            if opencv_backend is None:
                opencv_backend = int(cv2.CAP_ANY)

            # If the backend is still not an integer, raises an error
            if not isinstance(opencv_backend, int):
                message = (
                    f"Unable to add the {camera_name} OpenCVCamera object to the {self._name} VideoSystem. Expected "
                    f"an integer or None for opencv_backend argument, but got {opencv_backend} of type "
                    f"{type(opencv_backend).__name__}."
                )
                raise console.error(error=TypeError, message=message)

            # Ensures that color is either True or False. Also replaces None fps_override values with 0 to optimize
            # future data handling
            image_color = True if color is not None else False
            fps_override = 0 if fps_override is None else fps_override

            # Instantiates and returns the OpenCVCamera class object
            camera = OpenCVCamera(
                name=camera_name,
                color=image_color,
                backend=opencv_backend,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=frames_per_second,
            )

        # HarvestersCamera
        elif camera_backend == CameraBackends.HARVESTERS:
            # Ensures that the cti_path is a valid Path object and that it points to a '.cti' file.
            if not isinstance(cti_path, Path) or cti_path.suffix != ".cti":
                message = (
                    f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
                    f"pointing to the '.cti' file for cti_path argument, but got {cti_path} of "
                    f"type {type(cti_path).__name__}."
                )
                console.error(error=ValueError, message=message)

            # Instantiates and returns the HarvestersCamera class object
            camera = HarvestersCamera(
                name=camera_name,
                cti_path=cti_path,  # type: ignore
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=frames_per_second,
            )

        # MockCamera
        elif camera_backend == CameraBackends.MOCK:
            # Ensures that mock_color is either True or False.
            mock_color = True if color is not None else False

            # Unlike 'real' cameras, MockCamera cannot retrieve fps and width / height from hardware memory.
            # Therefore, if either of these variables is NoneType, it is initialized to the class default value.
            if isinstance(frame_height, NoneType):
                frame_height = 400
            if isinstance(frame_width, NoneType):
                frame_width = 600
            if isinstance(frames_per_second, NoneType):
                frames_per_second = 30

            # Instantiates and returns the MockCamera class object
            camera = MockCamera(
                name=camera_name,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=frames_per_second,
                color=mock_color,
            )

        # If the input backend does not match any of the supported backends, raises an error
        else:
            message = (
                f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
                f"camera_backend argument {camera_backend} of type {type(camera_backend).__name__}. "
                f"camera_backend has to be one of the options available from the CameraBackends enumeration."
            )
            raise console.error(error=ValueError, message=message)

        # If the camera class was successfully instantiated, packages the class alongside additional parameters into a
        # CameraSystem object and appends it to the cameras list.
        self._cameras.append(_CameraSystem(camera=camera, display_frames=display_frames, fps_override=fps_override))

    def add_image_saver(
        self,
        output_directory: Path,
        image_format: ImageFormats = ImageFormats.TIFF,
        tiff_compression: int = cv2.IMWRITE_TIFF_COMPRESSION_LZW,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        png_compression: int = 1,
        process_count: int = 1,
        thread_count: int = 5,
    ) -> ImageSaver:
        """Creates and returns a Saver class instance configured to save camera frame as independent images.

        This method centralizes Saver class instantiation. It contains methods for verifying the input information
        and instantiating the specialized Saver class to output images. All Saver classes from this library have to be
        initialized using this method or a companion create_video_saver() method.

        Notes:
            While the method contains many arguments that allow to flexibly configure the instantiated saver, the only
            crucial one is the output directory. That said, it is advised to optimize all parameters
            relevant for your chosen backend as needed, as it directly controls the quality, file size and encoding
            speed of the generated file(s).

        Args:
            output_directory: The path to the output directory where the image or video files will be saved after
                encoding.
            image_format: The format to use for the output images. Use ImageFormats enumeration
                to specify the desired image format. Currently, only 'TIFF', 'JPG', and 'PNG' are supported.
            tiff_compression: The integer-code that specifies the compression strategy used for
                Tiff image files. Has to be one of the OpenCV 'IMWRITE_TIFF_COMPRESSION_*' constants. It is recommended
                to use code 1 (None) for lossless and fastest file saving or code 5 (LZW) for a good
                speed-to-compression balance.
            jpeg_quality: An integer value between 0 and 100 that controls the 'loss' of the
                JPEG compression. A higher value means better quality, less data loss, bigger file size, and slower
                processing time.
            jpeg_sampling_factor: An integer-code that specifies how JPEG encoder samples image
                color-space. Has to be one of the OpenCV 'IMWRITE_JPEG_SAMPLING_FACTOR_*' constants. It is recommended
                to use code 444 to preserve the full color-space of the image for scientific applications.
            png_compression: An integer value between 0 and 9 that specifies the compression of
                the PNG file. Unlike JPEG, PNG files are always lossless. This value controls the trade-off between
                the compression ratio and the processing time.
            process_count: The number of ImageSaver processes to use. This parameter is only used when the
                saver class is an instance of ImageSaver. Since each saved image is independent of all other images,
                the performance of ImageSvers can be improved by using multiple processes with multiple threads to
                increase the saving throughput. For most use cases, a single saver process will be enough.
            thread_count: The number of writer threads to be used by the saver class. Since
                ImageSaver uses the c-backed OpenCV library, it can safely process multiple frames at the same time
                via multithreading. This controls the number of simultaneously saved images the class will support.

        Raises:
            TypeError: If the input arguments are not of the correct type.
        """
        # Verifies that the input arguments are of the correct type.
        if not isinstance(output_directory, Path):
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected a Path instance for output_directory "
                f"argument, but got {output_directory} of type {type(output_directory).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(image_format, ImageFormats):
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected an ImageFormats instance for "
                f"image_format argument, but got {image_format} of type {type(image_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(tiff_compression, int):
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected an integer for tiff_compression "
                f"argument, but got {tiff_compression} of type {type(tiff_compression).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not 0 <= jpeg_quality <= 100:
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 100 for "
                f"jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
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
                f"Unable to instantiate an ImageSaver class object. Expected one of the "
                f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_' constants for jpeg_sampling_factor argument, but got "
                f"{jpeg_sampling_factor} of type {type(jpeg_sampling_factor).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not 0 <= png_compression <= 9:
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 9 for "
                f"png_compression argument, but got {png_compression} of type "
                f"{type(png_compression).__name__}."
            )
            console.error(error=TypeError, message=message)
        if process_count < 1:
            raise ValueError(
                f"Unable to instantiate an ImageSaver class object. Expected a positive integer for "
                f"image_saver_process_count, but got {process_count} of type "
                f"{type(process_count).__name__}"
            )
        if not isinstance(thread_count, int):
            message = (
                f"Unable to instantiate an ImageSaver class object. Expected an integer for thread_count "
                f"argument, but got {thread_count} of type {type(thread_count).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Configures, initializes and returns an ImageSaver instance
        return ImageSaver(
            output_directory=output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=jpeg_sampling_factor,
            png_compression=png_compression,
            thread_count=thread_count,
        )

    def add_video_saver(
        self,
        output_directory: Path,
        hardware_encoding: bool = False,
        video_format: VideoFormats = VideoFormats.MP4,
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: GPUEncoderPresets | CPUEncoderPresets = CPUEncoderPresets.SLOW,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ) -> VideoSaver:
        """Creates and returns a Saver class instance configured to save camera frame as video files.

        This method centralizes Saver class instantiation. It contains methods for verifying the input information
        and instantiating the specialized Saver class to output video files. All Saver classes from this library have
        to be initialized using this method or a companion create_image_saver() method.

        Notes:
            While the method contains many arguments that allow to flexibly configure the instantiated saver, the only
            crucial one is the output directory. That said, it is advised to optimize all parameters
            relevant for your chosen backend as needed, as it directly controls the quality, file size and encoding
            speed of the generated file(s).

        Args:
            output_directory: The path to the output directory where the image or video files will be saved after
                encoding.
            hardware_encoding: Only for Video savers. Determines whether to use GPU (hardware) encoding or CPU
                (software) encoding. It is almost always recommended to use the GPU encoding for considerably faster
                encoding with almost no quality loss. GPU encoding is only supported by modern Nvidia GPUs, however.
            video_format: Only for Video savers. The container format to use for the output video. Use VideoFormats
                enumeration to specify the desired container format. Currently, only 'MP4', 'MKV', and 'AVI' are
                supported.
            video_codec: Only for Video savers. The codec (encoder) to use for generating the video file. Use
                VideoCodecs enumeration to specify the desired codec. Currently, only 'H264' and 'H265' are supported.
            preset: Only for Video savers. The encoding preset to use for generating the video file. Use
                GPUEncoderPresets or CPUEncoderPresets enumerations to specify the preset. Note, you have to select the
                correct preset enumeration based on whether hardware encoding is enabled!
            input_pixel_format: Only for Video savers. The pixel format used by input data. This only applies when
                encoding simultaneously acquired frames. When encoding pre-acquire images, FFMPEG will resolve color
                formats automatically. Use InputPixelFormats enumeration to specify the desired pixel format.
                Currently, only 'MONOCHROME' and 'BGR' and 'BGRA' options are supported. The option to choose depends
                on the configuration of the Camera class that was used for frame acquisition.
            output_pixel_format: Only for Video savers. The pixel format to be used by the output video. Use
                OutputPixelFormats enumeration to specify the desired pixel format. Currently, only 'YUV420' and
                'YUV444' options are supported.
            quantization_parameter: Only for Video savers. The integer value to use for the 'quantization parameter'
                of the encoder. The encoder uses 'constant quantization' to discard the same amount of information from
                each macro-block of the frame, instead of varying the discarded information amount with the complexity
                of macro-blocks. This allows precisely controlling output video size and distortions introduced by the
                encoding process, as the changes are uniform across the whole video. Lower values mean better quality
                (0 is best, 51 is worst). Note, the default assumes H265 encoder and is likely too low for H264 encoder.
                H264 encoder should default to ~25.
            gpu: Only for Video savers. The index of the GPU to use for encoding. Valid GPU indices can be obtained
                from 'nvidia-smi' command. This is only used when hardware_encoding is True.

        Raises:
            TypeError: If the input arguments are not of the correct type.
            RuntimeError: If the instantiated saver is configured to use GPU video encoding, but the method does not
                detect any available NVIDIA GPUs.
        """

        # Verifies that the input arguments are of the correct type.
        if not isinstance(output_directory, Path):
            message = (
                f"Unable to instantiate a Saver class object. Expected a Path instance for output_directory argument, "
                f"but got {output_directory} of type {type(output_directory).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(hardware_encoding, bool):
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected a boolean for hardware_encoding "
                f"argument, but got {hardware_encoding} of type {type(hardware_encoding).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(video_format, VideoFormats):
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected a VideoFormats instance for "
                f"video_format argument, but got {video_format} of type {type(video_format).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(video_codec, VideoCodecs):
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected a VideoCodecs instance for "
                f"video_codec argument, but got {video_codec} of type {type(video_codec).__name__}."
            )
            raise console.error(error=TypeError, message=message)

        # Preset source depends on hardware_encoding value
        if hardware_encoding:
            if not isinstance(preset, GPUEncoderPresets):
                message = (
                    f"Unable to instantiate a GPU VideoSaver class object. Expected a GPUEncoderPresets instance "
                    f"for preset argument, but got {preset} of type {type(preset).__name__}."
                )
                console.error(error=TypeError, message=message)
        else:
            if not isinstance(preset, CPUEncoderPresets):
                message = (
                    f"Unable to instantiate a CPU VideoSaver class object. Expected a CPUEncoderPresets instance "
                    f"for preset argument, but got {preset} of type {type(preset).__name__}."
                )
                console.error(error=TypeError, message=message)

        if not isinstance(input_pixel_format, InputPixelFormats):
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected an InputPixelFormats instance for "
                f"input_pixel_format argument, but got {input_pixel_format} of type "
                f"{type(input_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not isinstance(output_pixel_format, OutputPixelFormats):
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected an OutputPixelFormats instance for "
                f"output_pixel_format argument, but got {output_pixel_format} of type "
                f"{type(output_pixel_format).__name__}."
            )
            console.error(error=TypeError, message=message)
        if not -1 < quantization_parameter <= 51:
            message = (
                f"Unable to instantiate a VideoSaver class object. Expected an integer between 0 and 51 for "
                f"quantization_parameter argument, but got {quantization_parameter} of type "
                f"{type(quantization_parameter).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Since GPU encoding is currently only supported for NVIDIA GPUs, verifies that nvidia-smi is callable
        # for the host system. This is used as a proxy to determine whether the system has an Nvidia GPU:
        if hardware_encoding == True:
            try:
                # Runs nvidia-smi command, uses check to trigger CalledProcessError exception if runtime fails
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                message = (
                    f"Unable to instantiate a VideoSaver class object. The object is configured to use the GPU "
                    f"video encoding backend, which currently only supports NVIDIA GPUs. Calling 'nvidia-smi' to "
                    f"verify the presence of NVIDIA GPUs did not run successfully, indicating that there are no "
                    f"available NVIDIA GPUs on the host system. Use a CPU encoder or make sure nvidia-smi is callable "
                    f"from Python shell."
                )
                console.error(error=RuntimeError, message=message)

        # Configures, initializes and returns a VideoSaver instance
        return VideoSaver(
            output_directory=output_directory,
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

        This method is used as a thread target as part of the _produce_images_loop() runtime. It is used to display
        frames as they are grabbed from the camera and passed to the multiprocessing queue. This allows visually
        inspecting the frames as they are processed, which is often desired during scientific experiments.

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
            camera_name: The name of the camera which produces displayed images. This is used to generate a
                descriptive window name for the display GUI.
        """

        # Initializes the display window using 'normal' mode to support user-controlled resizing.
        window_name = f"{camera_name} Camera Feed"
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
        camera: OpenCVCamera | HarvestersCamera | MockCamera,
        image_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
        log_path: Path,
        display_video: bool = False,
        fps: Optional[float] = None,
    ) -> None:
        """Continuously grabs frames from the camera and queues them up to be saved by the consumer processes and
        displayed via the display thread.

        This method loops while the first element in terminator_array (index 0) is nonzero. It continuously grabs
        frames from the camera, but only queues them up to be saved by the consumer processes as long as the second
        element in terminator_array (index 1) is nonzero. This method is meant to be run as a process and will create
        an infinite loop if run on its own.

        Notes:
            The method can be configured with an fps override to manually control the acquisition frame rate. Generally,
            this functionality should be avoided for most scientific and industrial cameras, as they all have a
            built-in frame rate limiter that will be considerably more efficient than the local implementation. For
            cameras without a built-in frame-limiter however, this functionality can be used to enforce a certain
            frame rate via software.

            When enabled, the method writes each frame data, ID, and acquisition timestamp relative to onset time to the
            image_queue as a 3-element tuple.

        Args:
            camera: A supported Camera class instance that is used to interface with the camera that produces frames.
            image_queue: A multiprocessing queue that buffers and pipes acquired frames to consumer processes.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
            log_path: The path to be used for logging frame acquisition times as .txt entries. This method establishes
                and writes the 'onset' point in UTC time to the file. Subsequently, all frame acquisition stamps are
                given in microseconds elapsed since the onset point.
            display_video: Determines whether to display acquired frames to the user through an OpenCV backend.
            fps: Manually overrides camera acquisition frame rate by triggering frame grabbing method at the specified
                interval. The override should be avoided for most higher-end cameras, and their built-in frame limiter
                module should be used instead (fps can be specified when instantiating Camera classes).
        """

        # Creates a timer that time-stamps acquired frames. This information is crucial for later alignment
        # of multiple data sources.
        # noinspection PyTypeChecker
        stamp_timer: PrecisionTimer = PrecisionTimer("us")

        # Also initializes a timer to limit displayed frames to 30 fps. This is to save computational resources, as
        # displayed framerate does not need to be as high as acquisition framerate. This becomes very relevant for
        # displaying large frames at high speeds, as the imshow backend can easily become overwhelmed, leading to
        # queue backlogging and displayed data lagging behind the real stream.
        show_timer: PrecisionTimer = PrecisionTimer("ms")

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later time
        # readouts.
        onset = datetime.datetime.now(timezone.utc)
        stamp_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # If the method is configured to display acquired frames, sets up the display thread and a queue that buffers
        # and pipes the frames to be displayed to the worker thread.
        display_queue: Optional[Queue]  # type: ignore
        if display_video:
            display_queue = Queue()
            display_thread = Thread(target=VideoSystem._frame_display_loop, args=(display_queue, camera.name))
            display_thread.start()
        else:
            display_queue = None
            display_thread = None

        # If the method is configured to manually control the acquisition frame rate, initializes the high-precision
        # timer to control the acquisition and translates the desired frame per second parameter into the precision
        # units of the timer.
        # noinspection PyTypeChecker
        acquisition_timer: PrecisionTimer = PrecisionTimer("us")
        frame_time: float
        if fps is not None:
            # Translates the fps to use the timer-units of the timer. In this case, converts from frames per second to
            # frames per microsecond.
            # noinspection PyTypeChecker
            frame_time = convert_time(time=fps, from_units="s", to_units="us", convert_output=True)  # type: ignore
        else:
            frame_time = -1.0  # Due to the timer comparison below, this is equivalent to setting this to None

        # Connects to the camera and to the terminator array.
        camera.connect()
        terminator_array.connect()

        frame_number = 1  # Tracks the number of acquired frames. This is used to generate IDs for acquired frames.

        # Saves onset data to the log file. Due to how the consumer processes work, this should always be the very
        # first entry to the file. Also includes the name of the camera.
        with open(log_path, mode="wt", buffering=1) as log:
            log.write(f"{camera.name}-{str(onset)}\n")  # Includes a newline to efficiently separate further entries

        # The loop runs until the VideoSystem is terminated by setting the first element (index 0) of the array to 0
        while terminator_array.read_data(index=0, convert_output=True):
            # If the fps override is enabled, this loop is further limited to acquire frames at the specified rate,
            # which is helpful for some cameras that do not have a built-in acquisition control functionality.
            if acquisition_timer.elapsed > frame_time:
                # Grabs the frames as a numpy ndarray
                frame = camera.grab_frame()
                frame_stamp = stamp_timer.elapsed

                # If manual frame rate control is enabled, resets the timer after acquiring each frame. This results in
                # the time spent on further processing below to be 'absorbed' into the between-frame wait time.
                if fps is not None:
                    acquisition_timer.reset()

                # Only buffers the frame for saving if this behavior is enabled through the terminator array
                if terminator_array.read_data(index=1, convert_output=True):
                    # Bundles frame data, ID, and acquisition timestamp relative to onset and passes it to the
                    # multiprocessing Queue that delivers the data to the consumer process that saves it to
                    # non-volatile memory. This is likely the 'slowest' of all processing steps done by this loop.
                    image_queue.put((frame, f"{frame_number:08d}", frame_stamp))

                    # Increments the frame number to continuously update IDs for newly acquired frames
                    frame_number += 1

                # If the process is configured to display acquired frames, queues each frame to be displayed. Note, this
                # does not depend on whether the frame is buffered for saving. The display frame limiting is
                # critically important, as it prevents the display thread from being overwhelmed, causing displayed
                # stream to lag behind the saved stream.
                if display_video and show_timer.elapsed > 33.333:  # Display frames at 30 fps
                    display_queue.put(frame)  # type: ignore
                    show_timer.reset()  # Resets the timer for the next frame

        # Once the loop above is escaped releases all resources and terminates the Process.
        if display_video:
            # Terminates the display thread
            display_queue.put(None)  # type: ignore
            # Waits for the thread to close
            display_thread.join()  # type: ignore

        camera.disconnect()  # Disconnects from the camera
        terminator_array.disconnect()  # Disconnects from the terminator array

    @staticmethod
    def _frame_saving_loop(
        saver: VideoSaver | ImageSaver,
        image_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
        log_path: Path,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        video_frames_per_second: Optional[float] = None,
        video_id: Optional[str] = None,
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
            saver: One of the supported Saver classes that is used to save buffered camera frames by interfacing with
                the OpenCV or FFMPEG libraries.
            image_queue: A multiprocessing queue that buffers and pipes frames acquired by the producer process.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
            log_path: The path to be used for logging frame acquisition times as .txt entries. To minimize the latency
                between grabbing frames, timestamps are logged by consumers, rather than the producer. This method
                creates an entry that bundles each frame ID with its acquisition timestamp and appends it to the log
                file.
            frame_width: Only for VideoSaver classes. Specifies the width of the frames to be saved, in pixels.
                This has to match the width reported by the Camera class that produces the frames.
            frame_height: Same as above, but specifies the height of the frames to be saved, in pixels.
            video_frames_per_second: Only for VideoSaver classes. Specifies the desired frames-per-second of the
                encoded video file.
            video_id: Only for VideoSaver classes. Specifies the unique identifier used as the name of the
                encoded video file.
        """

        # Video savers require additional setup before they can save 'live' frames. Image savers are ready to save
        # frames with no additional setup
        if isinstance(saver, VideoSaver):
            # Initializes the video encoding by instantiating the FFMPEG process that does the encoding. For this, it
            # needs some additional information. Subsequently, Video and Image savers can be accessed via the same
            # frame-saving API.
            saver.create_live_video_encoder(
                frame_width=frame_width,  # type: ignore
                frame_height=frame_height,  # type: ignore
                video_id=video_id,  # type: ignore
                video_frames_per_second=video_frames_per_second,  # type: ignore
            )

        # Connects to the terminator array to manage loop runtime
        terminator_array.connect()

        # Minimizes IO delay by opening the file once and streaming to the file afterward. Uses line buffering
        with open(log_path, mode="at", buffering=1) as log:
            # The loop runs continuously until the first (index 0) element of the array is set to 0. Note, the loop is
            # kept alive until all frames in the queue are processed!
            while terminator_array.read_data(index=0, convert_output=True) or not image_queue.empty():
                # Grabs the image bundled with its ID and acquisition time from the queue and passes it to Saver class.
                # This relies on both Image and Video savers having the same 'live' frame saving API. Uses a
                # non-blocking binding to prevent deadlocking the loop, as it does not use the queue-based termination
                # method for reliability reasons.
                try:
                    frame, frame_id, frame_time = image_queue.get(block=False)
                except Exception:
                    # The only expected exception here is when queue is empty. This block overall makes the code
                    # repeatedly cycle through the loop, allowing flow control statements to properly terminate the
                    # 'while' loop if necessary and the queue is empty.
                    continue

                # This is only executed if the get() above yielded data
                saver.save_frame(frame_id, frame)

                # Bundles frame ID with acquisition time and writes it to the log file. Uses locks to prevent race
                # conditions when multiple ImageSavers are used at the same time. This should not majorly degrade
                # performance, as writing a short text string to file is still much faster than saving an image. For
                # Video savers, this is even less of a concern as there is always one saver at any given time.
                log.write(f"{str(frame_id)}-{str(frame_time)}\n")  # This produces entries like: '0001-18528'

        # Terminates the video encoder as part of the shutdown procedure
        if isinstance(saver, VideoSaver):
            saver.terminate_live_encoder()

        # Once the loop above is escaped, releases all resources and terminates the Process.
        terminator_array.disconnect()

    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the communication process is alive during
        runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):  # type: ignore

            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

            if not self._started:
                continue

            # Only checks that the process is alive if it is started. The shutdown() flips the started tracker
            # before actually shutting down the process, so there should be no collisions here.
            if self._communication_process is not None and not self._communication_process.is_alive():
                message = (
                    f"The communication process for the MicroControllerInterface {self._controller_name} with id "
                    f"{self._controller_id} has been prematurely shut down. This likely indicates that the process has "
                    f"encountered a runtime error that terminated the process."
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
