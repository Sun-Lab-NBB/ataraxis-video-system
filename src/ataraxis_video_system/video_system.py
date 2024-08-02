"""This module contains the VideoSystem class and the helper Camera class...

Brief description the module's purpose and the main VideoSystem class.
"""

import os
from queue import Empty, Queue
from typing import Any, Generic, Literal, TypeVar, cast, get_args, Optional
from types import NoneType
from threading import Thread
import multiprocessing
from multiprocessing import (
    Queue as UntypedMPQueue,
    Process,
    ProcessError,
)
from multiprocessing.managers import SyncManager

import cv2
import numpy as np
import ffmpeg  # type: ignore
import keyboard
import tifffile as tff
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray

from .camera import HarvestersCamera, OpenCVCamera, MockCamera, CameraBackends
from pathlib import Path
from harvesters.core import Harvester

# Used in many functions to convert between units
Precision_Type = Literal["ns", "us", "ms", "s"]
unit_conversion: dict[Precision_Type, int] = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}
precision: Precision_Type = "ms"

T = TypeVar("T")


class MPQueue(Generic[T]):
    """A wrapper for a multiprocessing queue object that makes Queue typed. This is used to appease mypy type checker"""

    def __init__(self) -> None:
        self._queue = UntypedMPQueue()  # type: ignore

    def put(self, item: T) -> None:
        self._queue.put(item)

    def get(self) -> T:
        return cast(T, self._queue.get())

    def get_nowait(self) -> T:
        return cast(T, self._queue.get_nowait())

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()  # pragma: no cover

    def cancel_join_thread(self) -> None:
        return self._queue.cancel_join_thread()


class VideoSystem:
    """Provides a system for efficiently taking, processing, and saving images in real time.

    Args:
        save_directory: location where the system saves images.
        camera: camera for image collection.
        display_video: whether or not to display a video of the current frames being recorded.
        save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg' is
            a lossy format. Set to 'mp4' for video saving.
        tiff_compression_level: the amount of compression to apply for tiff image saving. Range is [0, 9] inclusive. 0 gives fastest saving but
            most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
            relevant when save_format is specified as 'tiff.'
        jpeg_quality: the amount of compression to apply for jpeg image saving. Range is [0, 100] inclusive. 0 gives
            highest level of compression but the most loss of image detail. 100 gives the lowest level of compression
            but no loss of image detail. Thiscompression value is only relevant when save_format is specified as 'jpg.'
        mp4_config: A dictionary of ffmpeg parameters in the form {parameter_name: parameter_value, ...}
        num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
        num_threads: The number of image-saving threads to run per process. Applies only to image saving.

    Attributes:
        save_directory: location where the system saves images.
        camera: camera for image collection.
        _display_video: whether or not to display a video of the current frames being recorded.
        _save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg' is
            a lossy format
        _mp4_config:
        _tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
            most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
            relevant when save_format is specified as 'tiff.'
        _jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
            the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
            compression value is only relevant when save_format is specified as 'jpg.'
        _running: whether or not the video system is running.
        _producer_process: multiprocessing process to control the image collection.
        _consumer_processes: list multiprocessing processes to control image saving.
        _terminator_array: multiprocessing array to keep track of process activity and facilitate safe process
            termination.
        _image_queue: multiprocessing queue to hold images before saving.
        _num_consumer_processes: number of processes to run the image consumer loop on. Applies only to image saving.
        _threads_per_process: The number of image-saving threads to run per process. Applies only to image saving.

    Raises:
        ProcessError: If the function is created not within the '__main__' scope
        ValueError: If the save format is specified to an invalid format.
        ValueError: If a specified tiff_compression_level is not within [0, 9] inclusive.
        ValueError: If a specified jpeg_quality is not within [0, 100] inclusive.
        ProcessError: If the computer does not have enough cpu cores.
    """

    img_name = "img"
    vid_name = "video"
    Codec_Type = Literal["h264", "h264_mf", "libx264", "hevc", "hevc_mf", "libx265"]
    Save_Format_Type = Literal["png", "tiff", "tif", "jpg", "jpeg", "mp4"]

    def __init__(
        self,
        save_directory: str,
        camera: HarvestersCamera | OpenCVCamera | MockCamera,
        display_video: bool = True,
        save_format: Save_Format_Type = "png",
        tiff_compression_level: int = 6,
        jpeg_quality: int = 95,
        mp4_config: dict[str, str | int] = {},
        num_processes: int = 3,
        num_threads: int = 4,
    ):
        # Check to see if class was run from within __name__ = "__main__" or equivalent scope
        in_unprotected_scope: bool = False
        try:
            p = Process(target=VideoSystem._empty_function)
            p.start()
            p.join()
        except RuntimeError:  # pragma: no cover
            in_unprotected_scope = True

        if in_unprotected_scope:
            console.error(
                message="Instantiation method outside of '__main__' scope", error=ProcessError
            )  # pragma: no cover

        if save_format not in get_args(VideoSystem.Save_Format_Type):
            console.error(
                message=f"'{save_format}' is an invalid save format. Expects one of {get_args(VideoSystem.Save_Format_Type)}.",
                error=ValueError,
            )

        if not 0 <= tiff_compression_level <= 9:
            console.error(
                message=f"{tiff_compression_level} is an invalid tiff_compression_level. tiff_compression_level should be in [0,9] inclusive.",
                error=ValueError,
            )

        if not 0 <= jpeg_quality <= 100:
            console.error(
                message=f"{jpeg_quality} is an invalid jpeg_quality. jpeg_quality should be in [0,100] inclusive.",
                error=ValueError,
            )

        num_cores = multiprocessing.cpu_count()
        if num_processes > num_cores:
            console.error(
                message=f"{num_processes} processes were specified but the computer only has {num_cores} cpu cores.",
                error=ProcessError,
            )

        self.save_directory: str = save_directory
        self.camera: OpenCVCamera | HarvestersCamera | MockCamera = camera
        self._display_video = display_video
        self._save_format = save_format
        self._jpeg_quality = jpeg_quality
        self._tiff_compression_level = tiff_compression_level

        self._mp4_config = {
            "codec": "h264",
            "preset": "slow",
            "profile": "main",
            "crf": 28,
            "quality": 23,
            "threads": 0,
        }
        for key in mp4_config.keys():
            self._mp4_config[key] = mp4_config[key]

        self._num_consumer_processes = num_processes
        self._threads_per_process = num_threads
        self._running: bool = False

        self._producer_process: Process | None = None
        self._consumer_processes: list[Process] = []
        self._terminator_array: SharedMemoryArray | None = None
        self._mpManager: SyncManager | None = None
        self._image_queue: multiprocessing.Queue[Any] | None = None
        self._interactive_mode: bool | None = None

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
        mock_color: Optional[bool] = None,
    ) -> OpenCVCamera | HarvestersCamera | MockCamera:
        """Creates and returns a Camera class instance that uses the specified camera backend.

        This method centralizes Camera class instantiation. It contains methods for verifying the input information
        and instantiating the specialized Camera class based on the requested camera backend. All Camera classes from
        this library have to be initialized using this method.

        Note:
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
            mock_color: Optional. A boolean indicating whether to simulate color in the mock camera. This is only used
                by the Mock backend and generally should not be used by anyone other than library developers.

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

            # Instantiates and returns the OpenCVCamera class object
            return OpenCVCamera(
                name=camera_name,
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
            mock_color = True if mock_color is not None else False

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
            if not isinstance(camera_backend, int):
                message = (
                    f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
                    f"camera_backend argument {camera_backend} of type {type(camera_backend).__name__}. "
                    f"camera_backend has to be one of the options available from the CameraBackends enumeration."
                )
                raise console.error(error=ValueError, message=message)

    @staticmethod
    def _empty_function() -> None:
        """A placeholder function used to verify the class is only instantiated inside the main scope of each runtime.

        The function itself does nothing.
        """
        pass

    def start(
        self,
        listen_for_keypress: bool = False,
        terminator_array_name: str = "terminator_array",
        display_video: bool | None = None,
        save_format: Save_Format_Type | None = None,
        tiff_compression_level: int | None = None,
        jpeg_quality: int | None = None,
        mp4_config: dict[str, str | int] = {},
        num_processes: int | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Starts the video system.

        Args:
            listen_for_keypress: If true, the video system will stop the image collection when the 'q' key is pressed
                and stop image saving when the 'w' key is pressed.
            terminator_array_name: The name of the shared_memory_array to be created. When running multiple
                video_systems concurrently, each terminator_array should have a unique name.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. Range is [0, 9] inclusive.
                0 gives fastest saving but most memory used. 9 gives slowest saving but least amount of memory used. This
                compression value is onlyrelevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. Range is [0, 100] inclusive. 0 gives
                highest level of compression but the most loss of image detail. 100 gives the lowest level of compression
                but no loss of image detail. Thiscompression value is only relevant when save_format is specified as 'jpg.'
            mp4_config: A dictionary of ffmpeg parameters in the form {parameter_name: parameter_value, ...}
            num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
            num_threads: The number of image-saving threads to run per process. Applies only to image saving.

        Raises:
            ProcessError: If the function is created not within the '__main__' scope.
            ValueError: If the save format is specified to an invalid format.
            ValueError: If a specified tiff_compression_level is not within [0, 9] inclusive.
            ValueError: If a specified jpeg_quality is not within [0, 100] inclusive.
            ProcessError: If the computer does not have enough cpu cores.
        """

        # Check to see if class was run from within __name__ = "__main__" or equivalent scope
        in_unprotected_scope: bool = False
        try:
            p = Process(target=VideoSystem._empty_function)
            p.start()
            p.join()
        except RuntimeError:  # pragma: no cover
            in_unprotected_scope = True

        if in_unprotected_scope:
            console.error(
                message="Instantiation method outside of '__main__' scope", error=ProcessError
            )  # pragma: no cover

        if display_video is not None:
            self._display_video = display_video

        if save_format is not None:
            if save_format in get_args(VideoSystem.Save_Format_Type):
                self._save_format = save_format
            else:
                console.error(
                    message=f"'{save_format}' is an invalid save format. Expects one of {get_args(VideoSystem.Save_Format_Type)}.",
                    error=ValueError,
                )

        if tiff_compression_level is not None:
            if 0 <= tiff_compression_level <= 9:
                self._tiff_compression_level = tiff_compression_level
            else:
                console.error(
                    message=f"{tiff_compression_level} is an invalid tiff_compression_level. tiff_compression_level should be in [0,9] inclusive.",
                    error=ValueError,
                )

        if jpeg_quality is not None:
            if 0 <= jpeg_quality <= 100:
                self._jpeg_quality = jpeg_quality
            else:
                console.error(
                    message=f"{jpeg_quality} is an invalid jpeg_quality. jpeg_quality should be in [0,100] inclusive.",
                    error=ValueError,
                )

        for key in mp4_config.keys():
            self._mp4_config[key] = mp4_config[key]

        if num_processes is not None:
            num_cores = multiprocessing.cpu_count()
            if num_processes > num_cores:
                console.error(
                    message=f"{num_processes} processes were specified but the computer only has {num_cores} cpu cores.",
                    error=ProcessError,
                )

            self._num_consumer_processes = num_processes

        if num_threads is not None:
            self._threads_per_process = num_threads

        self._interactive_mode = listen_for_keypress

        self.delete_images()

        self._mpManager = multiprocessing.Manager()
        self._image_queue = self._mpManager.Queue()  # type: ignore

        # First entry represents whether input stream is active, second entry represents whether output stream is active
        prototype = np.array([1, 1, 1], dtype=np.int32)

        self._terminator_array = SharedMemoryArray.create_array(
            name=terminator_array_name,
            prototype=prototype,
        )

        self._producer_process = Process(
            target=VideoSystem._produce_images_loop,
            args=(self.camera, self._image_queue, self._terminator_array, self._display_video),
            daemon=True,
        )

        if self._save_format in {"png", "tiff", "tif", "jpg", "jpeg"}:
            for _ in range(self._num_consumer_processes):
                self._consumer_processes.append(
                    Process(
                        target=VideoSystem._save_images_loop,
                        args=(
                            self._image_queue,
                            self._terminator_array,
                            self.save_directory,
                            self._save_format,
                            self._tiff_compression_level,
                            self._jpeg_quality,
                            self._threads_per_process,
                        ),
                        daemon=True,
                    )
                )
        else:  # self._save_format == "mp4"
            self._consumer_processes.append(
                Process(
                    target=VideoSystem._save_video_loop,
                    args=(
                        self._image_queue,
                        self._terminator_array,
                        self.save_directory,
                        self.camera._specs,
                        self._mp4_config,
                    ),
                    daemon=True,
                )
            )

        # Start save processes first to minimize queue buildup
        for process in self._consumer_processes:
            process.start()
        self._producer_process.start()

        if self._interactive_mode:
            keyboard.add_hotkey("q", self._on_press_q)
            keyboard.add_hotkey("w", self._on_press_w)

        self._running = True

    def stop_image_production(self) -> None:
        """Stops image collection."""
        if self._running:
            if self._terminator_array is not None:
                self._terminator_array.write_data(index=0, data=0)
            else:  # This error should never occur
                console.error(
                    message="Failure to start the stop image production process because _terminator_array is not initialized.",
                    error=TypeError,
                )

    # possibly delete this function
    def _stop_image_saving(self) -> None:
        """Stops image saving."""
        if self._running:
            if self._terminator_array is not None:
                self._terminator_array.write_data(index=1, data=0)
            else:  # This error should never occur
                console.error(
                    message="Failure to start the stop image saving process because _terminator_array is not initialized.",
                    error=TypeError,
                )

    def stop(self) -> None:
        """Stops image collection and saving. Ends all processes."""
        if self._running:
            if self._terminator_array is not None:
                self._terminator_array.write_data(index=(0, 3), data=[0, 0, 0])
            else:  # This error should never occur
                message = "Failure to start the stop video system  because _terminator_array is not initialized."
                console.error(message, error=TypeError)
                raise TypeError(message)  # pragma:no cover

            if self._mpManager is not None:
                self._mpManager.shutdown()
            else:  # This error should never occur
                console.error(
                    message="Failure to start the stop image production process because _mpManager is not initialized.",
                    error=TypeError,
                )

            if self._producer_process is not None:
                self._producer_process.join()
            else:  # This error should never occur
                console.error(
                    message="Failure to start the stop video system  because _producer_process is not initialized.",
                    error=TypeError,
                )

            for process in self._consumer_processes:
                process.join()

            # Empty joined processes from list to prepare for the system being started again
            self._consumer_processes = []

            # Ends listening for keypresses, does nothing if no keypresses were enabled.
            if self._interactive_mode:
                keyboard.unhook_all_hotkeys()
            self._interactive_mode = None

            self._terminator_array.disconnect()
            if self._terminator_array._buffer is not None:
                # This line needs to be changed to destroy TODO
                self._terminator_array._buffer.unlink()  # kill terminator array

            self._running = False

    # def __del__(self):
    #     """Ensures that the system is stopped upon garbage collection. """
    #     self.stop()

    @staticmethod
    def _delete_files_in_directory(path: str) -> None:
        """Generic method to delete all files in a specific folder.

        Args:
            path: Location of the folder.
        Raises:
            FileNotFoundError when the path does not exist.
        """
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)

    def delete_images(self) -> None:
        """Clears the save directory of all images.

        Raises:
            FileNotFoundError when self.save_directory does not exist.
        """
        VideoSystem._delete_files_in_directory(self.save_directory)

    @staticmethod
    def _produce_images_loop(
        camera: OpenCVCamera | HarvestersCamera | MockCamera,
        img_queue: MPQueue[Any],
        terminator_array: SharedMemoryArray,
        display_video: bool = False,
        fps: float | None = None,
    ) -> None:
        """Iteratively grabs images from the camera and adds to the img_queue.

        This function loops while the third element in terminator_array (index 2) is nonzero. It grabs frames as long as
        the first element in terminator_array (index 0) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a thread and will create an infinite loop if run on its
        own.

        Args:
            camera: a Camera object to take collect images.
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 0 is zero
                and completes when index 2 is zero.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
        # img_queue.cancel_join_thread()
        if display_video:
            window_name = "Camera Feed"
            cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
        camera.connect()
        terminator_array.connect()
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        n_images_produced = 0
        while terminator_array.read_data(index=2, convert_output=True):
            if terminator_array.read_data(index=0, convert_output=True):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    frame = camera.grab_frame()
                    img_queue.put((frame, n_images_produced))
                    if display_video:
                        cv2.imshow(window_name, frame)
                        cv2.waitKey(1)
                    n_images_produced += 1
                    if fps:
                        run_timer.reset()

        camera.disconnect()
        if display_video:
            cv2.destroyWindow(window_name)
        terminator_array.disconnect()

    @staticmethod
    def imwrite(filename: str, data: NDArray[Any], tiff_compression_level: int = 6, jpeg_quality: int = 95) -> None:
        """Saves an image to a specified file.

        Args:
            filename: path to image file to be created.
            data: pixel data of image.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. Range is [0, 9] inclusive. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: The amount of compression to apply for jpeg image saving. Range is [0, 100] inclusive. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
        """
        save_format = os.path.splitext(filename)[1][1:]
        if save_format in {"tiff", "tif"}:
            img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            tff.imwrite(
                filename, img_rgb, compression="zlib", compressionargs={"level": tiff_compression_level}
            )  # 0 to 9 default is 6
        elif save_format in {"jpg", "jpeg"}:
            cv2.imwrite(filename, data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])  # 0 to 100 default is 95
        else:  # save_format == "png"
            cv2.imwrite(filename, data)

    @staticmethod
    def _frame_saver(
        q: Queue[Any], save_directory: str, save_format: str, tiff_compression_level: int, jpeg_quality: int
    ) -> None:
        """A method that iteratively gets an image from a queue and saves it to save_directory. This method loops until
        it pulls an image off the queue whose id is 0. This loop is not meant to be called directly, rather it is meant
        to be the target of a separate thread.

        Args:
            q: A queue to hold images before saving.
            save_directory: relative path to location where image is to be saved.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
        """
        terminated = False
        while not terminated:
            frame, img_id = q.get()
            if img_id != -1:
                filename = os.path.join(save_directory, VideoSystem.img_name + str(img_id) + "." + save_format)
                VideoSystem.imwrite(
                    filename, frame, tiff_compression_level=tiff_compression_level, jpeg_quality=jpeg_quality
                )
            else:
                terminated = True
            q.task_done()

    @staticmethod
    def _save_images_loop(
        img_queue: MPQueue[Any],
        terminator_array: SharedMemoryArray,
        save_directory: str,
        save_format: str,
        tiff_compression_level: int,
        jpeg_quality: int,
        num_threads: int,
        fps: float | None = None,
    ) -> None:
        """Iteratively grabs images from the img_queue and saves them as png files.

        This function loops while the third element in terminator_array (index 2) is nonzero. It saves images as long as
        the second element in terminator_array (index 1) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a thread and will create an infinite loop if run on its
        own.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 1 is zero
                and completes when index 2 is zero.
            save_directory: relative path to location where images are to be saved.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
        q: Queue[Any] = Queue()
        workers = []
        for i in range(num_threads):
            workers.append(
                Thread(
                    target=VideoSystem._frame_saver,
                    args=(q, save_directory, save_format, tiff_compression_level, jpeg_quality),
                )
            )
        for worker in workers:
            worker.daemon = True
            worker.start()

        terminator_array.connect()
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        # img_queue.cancel_join_thread()
        while terminator_array.read_data(index=2, convert_output=True):
            if terminator_array.read_data(index=1, convert_output=True):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    try:
                        img = img_queue.get_nowait()
                        q.put(img)
                    except Empty:
                        pass
                    if fps:
                        run_timer.reset()
        terminator_array.disconnect()
        for _ in range(num_threads):
            q.put((None, -1))
        for worker in workers:
            worker.join()

    @staticmethod
    def _save_video_loop(
        img_queue: MPQueue[Any],
        terminator_array: SharedMemoryArray,
        save_directory: str,
        camera_specs: dict[str, Any],
        config: dict[str, str | int],
        fps: float | None = None,
    ) -> None:
        """Iteratively grabs images from the img_queue and adds them to an mp4 file.

        This creates runs the ffmpeg image saving process on a separate thread. It iteratively grabs images from the
        queue on the main thread.

        This function loops while the third element in terminator_array (index 2) is nonzero. It saves images as long as
        the second element in terminator_array (index 1) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a process 264rgbnd will create an infinite loop if run on its
        own.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 1 is zero
                and completes when index 2 is zero.
            save_directory: relative path to location where images are to be saved.
            camera_specs: a dictionary containing specifications of the camera. Specifically, the dictionary must
                contain the camera's frames per second, denoted 'fps', and the camera frame size denoted by
                'frame_width' and 'frame_height'.
            config:
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
        filename = os.path.join(save_directory, f"{VideoSystem.vid_name}.mp4")

        codec = "libx264"

        default_keys = ["codec", "preset", "profile", "crf", "quality", "threads"]
        additional_config = {}
        for key in config.keys():
            if key not in default_keys:
                additional_config[key] = config[key]

        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(camera_specs["fps"]),
                format="rawvideo",
                pix_fmt="bgr24",
                s="{}x{}".format(int(camera_specs["frame_width"]), int(camera_specs["frame_height"])),
            )
            .output(
                filename,
                vcodec=config["codec"],
                pix_fmt="nv21",
                preset=config["preset"],
                profile=config["profile"],
                crf=config["crf"],
                quality=config["quality"],
                threads=config["threads"],
                **additional_config,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        terminator_array.connect()
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        # img_queue.cancel_join_thread()
        while terminator_array.read_data(index=2, convert_output=True):
            if terminator_array.read_data(index=1, convert_output=True):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    if not img_queue.empty():
                        image, _ = img_queue.get()
                        ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())
                    if fps:
                        run_timer.reset()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        terminator_array.disconnect()

    # def save_imgs_as_vid(self) -> None:
    #     """Converts a set of id labeled images into an mp4 video file.
    #
    #     This is a wrapper class for the static method imgs_to_vid. It calls imgs_to_vid with arguments fitting a
    #     specific object instance.
    #
    #     Raises:
    #         Exception: If there are no images of the specified type in the specified directory.
    #     """
    #     VideoSystem.imgs_to_vid(
    #         fps=int(self.camera.fps), img_directory=self.save_directory, img_filetype=self._save_format
    #     )
    #
    # @staticmethod
    # def imgs_to_vid(
    #     fps: int, img_directory: str = "imgs", img_filetype: str = "png", vid_directory: str | None = None
    # ) -> None:
    #     """Converts a set of id labeled images into an mp4 video file.
    #
    #     Args:
    #         fps: The framerate of the video to be created.
    #         img_directory: The directory where the images are saved.
    #         img_filetype: The type of image to be read. Supported types are "tiff", "png", and "jpg"
    #         vid_directory: The location to save the video. Defaults to the directory that the images are saved in.
    #
    #     Raises:
    #         Exception: If there are no images of the specified type in the specified directory.
    #     """
    #
    #     if vid_directory is None:
    #         vid_directory = img_directory
    #
    #     vid_name = os.path.join(vid_directory, f"{VideoSystem.vid_name}.mp4")
    #
    #     sample_img = cv2.imread(os.path.join(img_directory, f"{VideoSystem.img_name}0.{img_filetype}"))
    #
    #     if sample_img is None:
    #         console.error(message=f"No {img_filetype} images found in {img_directory}.", error=Exception)
    #
    #     frame_height, frame_width, _ = sample_img.shape
    #
    #     ffmpeg_process = (
    #         ffmpeg.input(
    #             "pipe:",
    #             framerate="{}".format(fps),
    #             format="rawvideo",
    #             pix_fmt="bgr24",
    #             s="{}x{}".format(int(frame_width), int(frame_height)),
    #         )
    #         .output(vid_name, vcodec="h264", pix_fmt="nv21", **{"b:v": 2000000})
    #         .overwrite_output()
    #         .run_async(pipe_stdin=True)
    #     )
    #
    #     search_pattern = os.path.join(img_directory, f"{VideoSystem.img_name}*.{img_filetype}")
    #     files = glob.glob(search_pattern)
    #     num_files = len(files)
    #     for id in range(num_files):
    #         file = os.path.join(img_directory, f"{VideoSystem.img_name}{id}.{img_filetype}")
    #         dat = cv2.imread(file)
    #         ffmpeg_process.stdin.write(dat.astype(np.uint8).tobytes())
    #
    #     ffmpeg_process.stdin.close()
    #     ffmpeg_process.wait()

    def _on_press_q(self) -> None:
        self._on_press("q")

    def _on_press_w(self) -> None:
        self._on_press("w")

    def _on_press(self, key: str) -> None:
        """Stops certain aspects of the system (production, saving) based on specific key_presses ("q", "w")

        Stops video system if both terminator flags have been set to 0.

        Args:
            key: the key that was pressed.
            terminator_array: A multiprocessing array to hold terminate flags.
        """
        if key == "q":
            self.stop_image_production()
            console.echo("Stopped taking images")
        elif key == "w":
            self._stop_image_saving()
            console.echo("Stopped saving images")

        if self._running:
            if self._terminator_array is not None:
                if not self._terminator_array.read_data(
                    index=0, convert_output=True
                ) and not self._terminator_array.read_data(index=1, convert_output=True):
                    self.stop()

            else:  # This error should never occur
                console.error(
                    message="Failure to start the stop image production process because _terminator_array is not initialized.",
                    error=TypeError,
                )
