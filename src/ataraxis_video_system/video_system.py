"""This module contains the main VideoSystem class that contains methods for setting up, running, and tearing down
interactions between Camera and Saver classes.

While Camera and Saver classes provide convenient interface for cameras and saver backends, VideoSystem connects
cameras to savers and manages the flow of frames between them. Each VideoSystem is a self-contained entity that
provides a simple API for recording data from a wide range of cameras as images or videos. The class is written to
maximize runtime performance.

All user-oriented functionality of this library is available through the public methods of the VideoSystem class.
"""

from queue import Queue
from typing import Any, Optional
from types import NoneType
from threading import Thread
import multiprocessing
from multiprocessing import Queue as MPQueue, Process, ProcessError, Lock
from multiprocessing.managers import SyncManager

import cv2
import numpy as np
import keyboard
from ataraxis_base_utilities.console.console_class import LogLevel
from ataraxis_time import PrecisionTimer
from ataraxis_time.time_helpers import convert_time
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray
import datetime

from .camera import HarvestersCamera, OpenCVCamera, MockCamera, CameraBackends
from .saver import (
    SaverBackends,
    ImageFormats,
    VideoFormats,
    VideoCodecs,
    InputPixelFormats,
    OutputPixelFormats,
    CPUEncoderPresets,
    GPUEncoderPresets,
    VideoSaver,
    ImageSaver,
)
from pathlib import Path
from harvesters.core import Harvester


class VideoSystem:
    """Provides a system for efficiently taking, processing, and saving images in real time.

    Args:
        display_frames: whether or not to display a video of the current frames being recorded.
        listen_for_keypress: If true, the video system will stop the image collection when the 'q' key is pressed
                and stop image saving when the 'w' key is pressed.
        shutdown_timeout: The number of seconds after which non-terminated processes will be forcibly terminated.
            The method first attempts to shut the processes gracefully, which may require some time. If you need to shut
            everything down fast and data loss is not an issue, you can specify a timeout, after which the processes
            will be terminated with data loss.

    Attributes:
        _camera: camera for image collection.
        _running: whether or not the video system is running.
        _producer_process: multiprocessing process to control the image collection.
        _consumer_processes: list multiprocessing processes to control image saving.
        _terminator_array: multiprocessing array to keep track of process activity and facilitate safe process
            termination.
        _image_queue: multiprocessing queue to hold images before saving.

    Raises:
        ProcessError: If the function is created not within the '__main__' scope
        ValueError: If the save format is specified to an invalid format.
        ValueError: If a specified tiff_compression_level is not within [0, 9] inclusive.
        ValueError: If a specified jpeg_quality is not within [0, 100] inclusive.
        ProcessError: If the computer does not have enough cpu cores.
    """

    def __init__(
        self,
        camera: HarvestersCamera | OpenCVCamera | MockCamera,
        saver: VideoSaver | ImageSaver,
        system_name: str,
        image_saver_process_count: int = 1,
        fps_override: Optional[int | float] = None,
        shutdown_timeout: Optional[int] = 600,
        *,
        display_frames: bool = True,
        listen_for_keypress: bool = False,
    ):
        # Saves input arguments to class attributes
        self._camera: OpenCVCamera | HarvestersCamera | MockCamera = camera
        self._saver: VideoSaver | ImageSaver = saver
        self._interactive_mode: bool = listen_for_keypress
        self._name: str = system_name
        self._shutdown_timeout: Optional[int] = shutdown_timeout
        self._running: bool = False  # Tracks whether the system has active processes

        # Sets up the multiprocessing Queue, which is used to buffer and pipe images from the producer (camera) to
        # one or more consumers (savers). Uses Manager() instantiation as it has a working qsize() method for all
        # supported platforms.
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._image_queue: MPQueue[Any] = self._mp_manager.Queue()  # type: ignore

        # Uses the saver class output directory to construct the path to the frame acquisition log file. This file
        # is stored in the same directory to which saved frames are written as images or video. The file is used to
        # store the acquisition time for each saved frame.
        # noinspection PyProtectedMember
        log_path = saver._output_directory.joinpath(f"{self._name}_frame_acquisition_log.txt")

        # Instantiates an array shared between all processes. This array is used to control all child processes.
        # Index 0 (element 1) is used to issue global process termination command, index 1 (element 2) is used to
        # flexibly enable or disable saving camera frames.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{system_name}_terminator_array",  # Uses class name with an additional specifier
            prototype=np.array([0, 0], dtype=np.int32),
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the image producer Process. This process continuously executes a loop that conditionally grabs frames
        # from camera, optionally displays them to the user and queues them up to be saved by the consumers.
        self._producer_process: Process = Process(
            target=self._frame_production_loop,
            args=(self._camera, self._image_queue, self._terminator_array, log_path, display_frames, fps_override),
            daemon=True,
        )

        # Instantiates the lock class to be used by consumer processes. Technically, this is only necessary when
        # multiple ImageSavers are used, but this should be a very minor performance concern, so it is used for all
        # use cases.
        lock = self._mp_manager.Lock()

        # Sets up image consumer Process(es), depending on whether the saver class is an ImageSaver or a VideoSaver.
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

            # For VideoSaver, spawns a single process and packages it into a tuple. Since VideoSaver relies on FFMPEG,
            # it automatically scales with available resources without the need for additional Python processes.
            self._consumer_processes = Process(
                target=self._frame_saving_loop,
                args=(
                    self._saver,
                    self._image_queue,
                    self._terminator_array,
                    log_path,
                    lock,
                    frame_width,
                    frame_height,
                    frame_rate,
                    self._name,
                ),
                daemon=True,
            )
        else:
            # For ImageSaver, spawns the requested number of saver processes. ImageSaver-based processed do not need
            # additional arguments required by VideoSaver processes, so instantiation does not require retrieving any
            # camera information.
            processes = [
                Process(
                    target=self._frame_saving_loop,
                    args=(
                        self._saver,
                        self._image_queue,
                        self._terminator_array,
                        log_path,
                        lock,
                    ),
                    daemon=True,
                )
                for _ in range(image_saver_process_count)
            ]

            # Casts the list to tuple and saves it to class attribute.
            self._consumer_processes: tuple[Process, ...] = tuple(processes)

    def __del__(self):
        """Ensures that all resources are released upon garbage collection."""
        self.stop()

    @staticmethod
    def get_opencv_ids() -> tuple[str, ...]:
        """Discovers and reports IDs and descriptive information about cameras accessible through the OpenCV library.

        This method can be used to discover camera IDs accessible through the OpenCV Backend. Subsequently,
        each of the IDs can be passed to the create_camera() method to create an OpenCVCamera class instance to
        interface with the camera. For each working camera, the method produces a string that includes camera ID, image
        width, height, and the fps value to help identifying the cameras.

        Notes:
            Currently, there is no way to get serial numbers or usb port names from OpenCV. Therefore, while this method
            tries to provide some ID information, it likely will not be enough to identify the cameras. Instead, it is
            advised to use the interactive imaging mode with each of the IDs to manually map IDs to cameras based on the
            produced visual stream.

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
        uses built-in Harvesters functionality to discover and return camera ID and descriptive information.
        The discovered IDs can later be used with the create_camera() method to create HarvestersCamera class to
        interface with the desired cameras.

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
        harvester.add_cti_file(file_path=str(cti_path))

        # Gets the list of accessible cameras
        harvester.update_device_info_list()

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
    def create_camera(
        camera_name: str,
        camera_backend: CameraBackends = CameraBackends.OPENCV,
        camera_id: int = 0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        frames_per_second: Optional[int | float] = None,
        opencv_backend: Optional[int] = None,
        cti_path: Optional[Path] = None,
        color: Optional[bool] = None,
    ) -> OpenCVCamera | HarvestersCamera | MockCamera:
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

        Raises:
            TypeError: If the input arguments are not of the correct type.
            ValueError: If the requested camera_backend is not one of the supported backends. If the input cti_path does
                not point to a '.cti' file.
        """

        # Verifies that the input arguments are of the correct type. Note, checks backend-specific arguments in
        # backend-specific clause.
        if not isinstance(camera_name, str):
            message = (
                f"Unable to instantiate a Camera class object. Expected a string for camera_name argument, but "
                f"got {camera_name} of type {type(camera_name).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(camera_id, int):
            message = (
                f"Unable to instantiate a {camera_name} Camera class object. Expected an integer for camera_id "
                f"argument, but got {camera_id} of type {type(camera_id).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frames_per_second, (int, float, NoneType)):
            message = (
                f"Unable to instantiate a {camera_name} Camera class object. Expected an integer, float or None for "
                f"frames_per_second argument, but got {frames_per_second} of type {type(frames_per_second).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frame_width, (int, NoneType)):
            message = (
                f"Unable to instantiate a {camera_name} Camera class object. Expected an integer or None for "
                f"frame_width argument, but got {frame_width} of type {type(frame_width).__name__}."
            )
            raise console.error(error=TypeError, message=message)
        if not isinstance(frame_height, (int, float, NoneType)):
            message = (
                f"Unable to instantiate a {camera_name} Camera class object. Expected an integer or None for "
                f"frame_height argument, but got {frame_height} of type {type(frame_height).__name__}."
            )
            raise console.error(error=TypeError, message=message)

        # Casts frame arguments to floats, as this is what camera classes expect
        frames_per_second = float(frames_per_second)
        frame_width = float(frame_width)
        frame_height = float(frame_height)

        # OpenCVCamera
        if camera_backend == CameraBackends.OPENCV:
            # If backend preference is None, uses generic preference
            if opencv_backend is None:
                opencv_backend = cv2.CAP_ANY

            # Otherwise, if the backend is not an integer, raises an error
            elif not isinstance(opencv_backend, int):
                message = (
                    f"Unable to instantiate a {camera_name} OpenCVCamera class object. Expected an integer or None "
                    f"for opencv_backend argument, but got {opencv_backend} of type {type(opencv_backend).__name__}."
                )
                raise console.error(error=TypeError, message=message)

            # Ensures that color is either True or False.
            image_color = True if color is not None else False

            # Instantiates and returns the OpenCVCamera class object
            return OpenCVCamera(
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
            if not isinstance(cti_path, Path) or cti_path.suffix is not ".cti":
                message = (
                    f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
                    f"pointing to the '.cti' file for cti_path argument, but got {cti_path} of "
                    f"type {type(cti_path).__name__}."
                )
                raise console.error(error=ValueError, message=message)

            # Instantiates and returns the HarvestersCamera class object
            return HarvestersCamera(
                name=camera_name,
                cti_path=cti_path,
                camera_id=camera_id,
                height=frame_height,
                width=frame_width,
                fps=frames_per_second,
            )

        # MockCamera
        elif camera_backend == CameraBackends.MOCK:
            # Ensures that mock_color is either True or False.
            mock_color = True if color is not None else False

            # Instantiates and returns the MockCamera class object
            return MockCamera(
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

    @staticmethod
    def create_saver(
        output_directory: Path,
        saver_backend: SaverBackends = SaverBackends.VIDEO,
        image_format: ImageFormats = ImageFormats.TIFF,
        tiff_compression: int = cv2.IMWRITE_TIFF_COMPRESSION_LZW,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        png_compression: int = 1,
        thread_count: int = 5,
        hardware_encoding: bool = False,
        video_format: VideoFormats = VideoFormats.MP4,
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: GPUEncoderPresets | CPUEncoderPresets = CPUEncoderPresets.SLOW,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ) -> VideoSaver | ImageSaver:
        """Creates and returns a Saver class instance that uses the specified saver backend.

        This method centralizes Saver class instantiation. It contains methods for verifying the input information
        and instantiating the specialized Saver class based on the requested saver backend. All Saver classes from
        this library have to be initialized using this method.

        Notes:
            While the method contains many arguments that allow to flexibly configure the instantiated saver, the only
            crucial ones are the output directory and saver backend. That said, it is advised to optimize all parameters
            relevant for your chosen backend as needed, as it directly controls the quality, file size and encoding
            speed of the generated file(s).

        Args:
            output_directory: The path to the output directory where the image or video files will be saved after
                encoding.
            saver_backend: The backend to use for the saver class. Currently, all supported backends are derived from
                the SaverBackends enumeration. It is advised to use the Video saver backend and, if possible, to
                enable hardware encoding, to save frames as video files. Alternatively, Image backend is also available
                to save frames as image files.
            image_format: Only for Image savers. The format to use for the output images. Use ImageFormats enumeration
                to specify the desired image format. Currently, only 'TIFF', 'JPG', and 'PNG' are supported.
            tiff_compression: Only for Image savers. The integer-code that specifies the compression strategy used for
                Tiff image files. Has to be one of the OpenCV 'IMWRITE_TIFF_COMPRESSION_*' constants. It is recommended
                to use code 1 (None) for lossless and fastest file saving or code 5 (LZW) for a good
                speed-to-compression balance.
            jpeg_quality: Only for Image savers. An integer value between 0 and 100 that controls the 'loss' of the
                JPEG compression. A higher value means better quality, less data loss, bigger file size, and slower
                processing time.
            jpeg_sampling_factor: Only for Image savers. An integer-code that specifies how JPEG encoder samples image
                color-space. Has to be one of the OpenCV 'IMWRITE_JPEG_SAMPLING_FACTOR_*' constants. It is recommended
                to use code 444 to preserve the full color-space of the image for scientific applications.
            png_compression: Only for Image savers. An integer value between 0 and 9 that specifies the compression of
                the PNG file. Unlike JPEG, PNG files are always lossless. This value controls the trade-off between
                the compression ratio and the processing time.
            thread_count: Only for Image savers. The number of writer threads to be used by the saver class. Since
                ImageSaver uses the c-backed OpenCV library, it can safely process multiple frames at the same time
                via multithreading. This controls the number of simultaneously saved images the class will support.
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
            ValueError: If the input saver_backend is not one of the supported options.
        """
        # Verifies that the input arguments are of the correct type. Note, checks backend-specific arguments in
        # backend-specific clause.
        if not isinstance(output_directory, Path):
            message = (
                f"Unable to instantiate a Saver class object. Expected a Path instance for output_directory argument, "
                f"but got {output_directory} of type {type(output_directory).__name__}."
            )
            raise console.error(error=TypeError, message=message)

        # Image Saver
        if saver_backend == SaverBackends.IMAGE:
            # Verifies ImageSaver initialization arguments:
            if not isinstance(image_format, ImageFormats):
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an ImageFormats instance for "
                    f"image_format argument, but got {image_format} of type {type(image_format).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not isinstance(tiff_compression, int):
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an integer for tiff_compression "
                    f"argument, but got {tiff_compression} of type {type(tiff_compression).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not 0 <= jpeg_quality <= 100:
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 100 for "
                    f"jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
                )
                raise console.error(error=TypeError, message=message)
            if not 0 <= jpeg_sampling_factor <= 4:
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 4 for "
                    f"jpeg_sampling_factor argument, but got {jpeg_sampling_factor} of type "
                    f"{type(jpeg_sampling_factor).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not 0 <= png_compression <= 9:
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 9 for "
                    f"png_compression argument, but got {png_compression} of type "
                    f"{type(png_compression).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not isinstance(thread_count, int):
                message = (
                    f"Unable to instantiate an ImageSaver class object. Expected an integer for thread_count "
                    f"argument, but got {thread_count} of type {type(thread_count).__name__}."
                )
                raise console.error(error=TypeError, message=message)

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

        # Video Saver
        elif saver_backend == SaverBackends.VIDEO:
            # Verifies VideoSaver initialization arguments:
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
                    raise console.error(error=TypeError, message=message)
            else:
                if not isinstance(preset, CPUEncoderPresets):
                    message = (
                        f"Unable to instantiate a CPU VideoSaver class object. Expected a CPUEncoderPresets instance "
                        f"for preset argument, but got {preset} of type {type(preset).__name__}."
                    )
                    raise console.error(error=TypeError, message=message)

            if not isinstance(input_pixel_format, InputPixelFormats):
                message = (
                    f"Unable to instantiate a VideoSaver class object. Expected an InputPixelFormats instance for "
                    f"input_pixel_format argument, but got {input_pixel_format} of type "
                    f"{type(input_pixel_format).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not isinstance(output_pixel_format, OutputPixelFormats):
                message = (
                    f"Unable to instantiate a VideoSaver class object. Expected an OutputPixelFormats instance for "
                    f"output_pixel_format argument, but got {output_pixel_format} of type "
                    f"{type(output_pixel_format).__name__}."
                )
                raise console.error(error=TypeError, message=message)
            if not -1 < quantization_parameter <= 51:
                message = (
                    f"Unable to instantiate a VideoSaver class object. Expected an integer between 0 and 51 for "
                    f"quantization_parameter argument, but got {quantization_parameter} of type "
                    f"{type(quantization_parameter).__name__}."
                )
                raise console.error(error=TypeError, message=message)

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

        # If the input backend does not match any of the supported backends, raises an error
        else:
            message = (
                f"Unable to instantiate a Saver class object due to encountering an unsupported "
                f"saver_backend argument {saver_backend} of type {type(saver_backend).__name__}. "
                f"saver_backend has to be one of the options available from the SaverBackends enumeration."
            )
            raise console.error(error=ValueError, message=message)

    @staticmethod
    def _frame_display_loop(display_queue: Queue, camera_name: str) -> None:
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
        image_queue: MPQueue[Any],
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

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later time
        # readouts.
        onset = datetime.datetime.now(datetime.UTC)
        stamp_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # If the method is configured to display acquired frames, sets up the display thread and a queue that buffers
        # and pipes the frames to be displayed to the worker thread.
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
        if fps is not None:
            # Translates the fps to use the timer-units of the timer. In this case, converts from frames per second to
            # frames per microsecond.
            # noinspection PyTypeChecker
            frame_time = convert_time(time=fps, from_units="s", to_units="us", convert_output=True)
        else:
            frame_time = None

        # Connects to the camera and to the terminator array.
        camera.connect()
        terminator_array.connect()

        frame_number = 1  # Tracks the number of acquired frames. This is used to generate IDs for acquired frames.

        # Saves onset data to the log file. Due to how the consumer processes work, this should always be the very
        # first entry to the file. Also includes the name of the camera.
        with open(log_path, mode="wt") as log:
            log.write(f"{camera.name}-{str(onset)}\n")  # Includes a newline to efficiently separate further entries

        # The loop runs until the VideoSystem is terminated by setting the first element (index 0) of the array to 0
        while terminator_array.read_data(index=0, convert_output=True):
            # If the fps override is enabled, this loop is further limited to acquire frames at the specified rate,
            # which is helpful for some cameras that do not have a built-in acquisition control functionality.
            if frame_time is None or acquisition_timer.elapsed > frame_time:
                # Grabs the frames as a numpy ndarray
                frame = camera.grab_frame()
                frame_stamp = stamp_timer.elapsed

                # If manual frame rate control is enabled, resets the timer after acquiring each frame. This results in
                # the time spent on further processing below to be 'absorbed' into the between-frame wait time.
                if fps:
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
                # does not depend on whether the frame is buffered for saving.
                if display_video:
                    display_queue.put(frame)

        # Once the loop above is escaped releases all resources and terminates the Process.

        if display_video:
            display_queue.put(None)  # Terminates the display thread
            display_thread.join()  # Waits for the thread to close
        camera.disconnect()  # Disconnects from the camera
        terminator_array.disconnect()  # Disconnects from the terminator array

    @staticmethod
    def _frame_saving_loop(
        saver: VideoSaver | ImageSaver,
        image_queue: MPQueue[Any],
        terminator_array: SharedMemoryArray,
        log_path: Path,
        lock: Lock,
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
            lock: A multiprocessing Lock instance, that is used to prevent race conditions when multiple ImageSavers
                need to access the frame acquisition log file. This should not majorly detract the benefit of multiple
                ImageSavers as logging is very fast compared to writing frames to disk.
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
                frame_width=frame_width,
                frame_height=frame_height,
                video_id=video_id,
                video_frames_per_second=video_frames_per_second,
            )

        # Connects to the terminator array to manage loop runtime
        terminator_array.connect()

        # The loop runs continuously until the first (index 0) element of the array is set to 0. Note, the loop is
        # kept alive until all frames in the queue are processed!
        while terminator_array.read_data(index=0, convert_output=True) or not image_queue.empty():
            # Grabs the image bundled with its ID and acquisition time from the queue and passes it to Saver class.
            # This relies on both Image and Video savers having the same 'live' frame saving API. Uses a non-blocking
            # binding to prevent deadlocking the loop, as it does not use the queue-based termination method for
            # reliability reasons.
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
            lock.acquire()
            with open(log_path, mode="at") as log:
                log.write(f"{str(frame_id)}-{str(frame_time)}\n")  # This produces entries like: '0001-18528'
            lock.release()

        # Terminates the video encoder as part of the shutdown procedure
        if isinstance(saver, VideoSaver):
            saver.terminate_live_encoder()

        # Once the loop above is escaped, releases all resources and terminates the Process.
        terminator_array.disconnect()

    @staticmethod
    def _empty_function() -> None:
        """A placeholder function used to verify the class is only instantiated inside the main scope of each runtime.

        The function itself does nothing. It is used to enforce that the start() method of the class only triggers
        inside the main scope, to avoid uncontrolled spawning of daemon processes.
        """
        pass

    def start(self) -> None:
        """Starts the consumer and producer processes of the video system class and begins acquiring camera frames.

        This process begins frame acquisition, but not frame saving. To enable saving acquired frames, call
        start_frame_saving() method. A call to this method is required to make the system operation and should only be
        carried out from the main scope of the runtime context. A call to this method should always be paired with a
        call to the stop() method to properly release the resources allocated to the class.

        Notes:
            When the class is configured to run in the interactive mode, calling this method automatically enables
            console output, even if it has been previously disabled. If console class is configured to log
            messages, the messages emitted by this class will be logged.

        Raises:
            ProcessError: If the method is called outside the '__main__' scope.
        """

        # if the class is already running, does nothing. This makes it impossible to call start multiple times in a row.
        if self._running:
            return

        # Ensures that it is onl;y possible to call this method from the main scope. This is to prevent uncontrolled
        # daemonic process spawning behavior.
        try:
            p = Process(target=self._empty_function)
            p.start()
            p.join()
        except RuntimeError:  # pragma: no cover
            message: str = (
                f"The start method of the VideoSystem {self._name} was called outside of '__main__' scope, which is "
                f"not allowed. Make sure that the start() and stop() methods are only called from the main scope."
            )
            console.error(message=message, error=ProcessError)

        # Starts consumer processes first to minimize queue buildup once the producer process is initialized.
        # Technically, due to saving frames being initially disabled, queue buildup is no longer a major concern.
        for process in self._consumer_processes:
            process.start()

        # Starts the producer process
        self._producer_process.start()

        # If the class is configured to run in the interactive mode, pairs interactive hotkeys with specific
        # 'on trigger' events.
        if self._interactive_mode:
            console.enable()  # Interactive mode automatically enables console output
            keyboard.add_hotkey("q", self._on_press_q)
            keyboard.add_hotkey("w", self._on_press_w)
            keyboard.add_hotkey("s", self._on_press_s)
            message = (
                f"Started VideoSystem {self._name} in interactive mode. Press 'q' to stop the system, 'w' to start "
                f"saving acquired camera frames and 's' to stop saving camera frames."
            )
            console.echo(message=message, level=LogLevel.INFO)

        # Sets the running tracker
        self._running = True

    def stop(self) -> None:
        """Stops all producer and consumer processes and terminates class runtime by releasing all resources.

        While this does not delete the class instance itself, it is generally recommended to only call this method once,
        during the general termination of the runtime that instantiated the class. Calling start() and stop() multiple
        times should be avoided, and instead it is better to re-initialize the class before cycling through start() and
        stop() calls. Every call to the start() method should be paired with the call to this method to properly release
        resources.

        Notes:
            The class will be kept alive until all frames buffered to the image_queue are saved. This is an intentional
            security feature that prevents information loss. If you want to override that behavior, you can initialize
            the class with a 'shutdown_timeout' argument to specify a delay after which all consumers will be forcibly
            terminated. Generally, it is highly advised not to tamper with this feature. The class uses the default
            timeout of 10 minutes (600 seconds), unless this is overridden at instantiation.
        """
        # Ensures that the stop procedure is only executed if the class is running
        if not self._running:
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
        for process in self._consumer_processes:
            process.join()

        # Ends listening for keypresses, does nothing if no keypresses were enabled.
        if self._interactive_mode:
            keyboard.unhook_all_hotkeys()

        # Disconnects from and destroys the terminator array buffer
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

        # Sets running tracker
        self._running = False

    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames.

        Does not interfere with grabbing and displaying the frames to user, this process is only stopped when the main
        stop() method is called.
        """
        if self._running:
            self._terminator_array.write_data(index=1, data=0)

    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames.

        The frames are grabbed and (optionally) displayed to user after the main start() method is called, but they
        are not initially written to non-volatile memory. The call to this method additionally enables saving the
        frames to non-volatile memory.
        """
        if self._running:
            self._terminator_array.write_data(index=1, data=1)

    def _on_press_q(self) -> None:
        """Specializes the 'on_press' method to 'q' key press."""
        self._on_press("q")

    def _on_press_s(self) -> None:
        """Specializes the 'on_press' method to 's' key press."""
        self._on_press("s")

    def _on_press_w(self) -> None:
        """Specializes the 'on_press' method to 'w' key press."""
        self._on_press("w")

    def _on_press(self, key: str) -> None:
        """Allows controlling certain class runtime behaviors based on specific keyboard key presses.

        This method is only used when the VideoSystem runs in the interactive mode. It allows toggling flow-control
        class methods (stop, start_frame_saving, stop_frame_saving) by pressing specific keyboard keys (q, w, s). By
        toggling these commands, the class alters certain values in the shared terminator_array, which alters the
        behavior of running consumer and producer processes.

        Notes:
            This method is designed to be used together with the keyboard library, which sets up the background
            listener loop and triggers appropriate _on_press() method specification depending on the key that was
            pressed.

        Args:
            key: The keyboard key that was pressed.
        """
        if key == "q":
            self.stop()
            console.echo(f"Initiated shutdown sequence for VideoSystem {self._name}")

        elif key == "s":
            self.stop_frame_saving()
            console.echo(f"Stopped saving frames acquired by VideoSystem {self._name}")

        elif key == "w":
            self.start_frame_saving()
            console.echo(f"Started saving frames acquired by VideoSystem {self._name}")
