"""This module provides the main VideoSystem class that contains methods for setting up, running, and tearing down
interactions between Camera and Saver classes.

While Camera and Saver classes provide an interface for cameras and saver backends, VideoSystem connects cameras to
savers and manages the flow of frames between them. Each VideoSystem is a self-contained entity that provides a simple
API for recording data from a wide range of cameras as images or videos. The class is written in a way that maximizes
runtime performance.

All user-oriented functionality of this library is available through the public methods of the VideoSystem class.
"""

import sys
from queue import Queue
from typing import TYPE_CHECKING, Any
from pathlib import Path
import warnings
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)

import cv2
import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console
from ataraxis_data_structures import DataLogger, LogPackage, SharedMemoryArray
from ataraxis_time.time_helpers import convert_time, get_timestamp, TimestampFormats

from .saver import (
    VideoSaver,
    VideoEncoders,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from .camera import MockCamera, OpenCVCamera, CameraInterfaces, HarvestersCamera

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager

    from numpy.typing import NDArray
    from numpy.lib.npyio import NpzFile


# Determines the maximum qp value used when initializing VideoSystem instances.
_MAXIMUM_QUANTIZATION_VALUE = 51


class VideoSystem:
    """Acquires, displays, and saves camera frames to disk using the requested camera interface and video saver.

    This class controls the runtime of a camera interface and a video saver running in independent processes and
    efficiently moves the frames acquired by the camera to the saver process.

    Notes:
        This class reserves up to two logical cores to support the producer (camera interface) and consumer
        (video saver) processes. Additionally, it reserves a variable portion of the RAM to buffer the frames as they
        are moved from the producer to the consumer.

        Video saving relies on the third-party software 'FFMPEG' to encode the video frames as an .mp4 file.
        See https://www.ffmpeg.org/download.html for more information on installing the library.

    Args:
        system_id: The unique value to use for identifying the VideoSystem instance in all output streams (log files,
            terminal messages, video files).
        data_logger: An initialized DataLogger instance used to log the timestamps for all frames saved by this
            VideoSystem instance.
        output_directory: The path to the output directory where to store the acquired frames as the .mp4 video file.
            Setting this argument to None disabled video saving functionality.
        camera_interface: The interface to use for working with the camera hardware. Must be one of the CameraInterfaces
            enumeration members.
        camera_index: The index of the camera in the list of all cameras discoverable by the chosen interface, e.g.: 0
            for the first available camera, 1 for the second, etc. This specifies the camera hardware the instance
            should interface with at runtime.
        display_frame_rate: Determines the frame rate at which to display the acquired frames to the user. Setting this
            argument to None (default) disables frame display functionality. Note, frame displaying is not supported
            on some macOS versions.
        frame_rate: The desired rate, in frames per second, at which to capture the frames. Note; whether the requested
            rate is attainable depends on the hardware capabilities of the camera and the communication interface. If
            this argument is not explicitly provided, the instance uses the default frame rate of the managed camera.
        frame_width: The desired width of the acquired frames, in pixels. Note; the requested width must be compatible
            with the range of frame dimensions supported by the camera hardware. If this argument is not explicitly
            provided, the instance uses the default frame width of the managed camera.
        frame_height: Same as 'frame_width', but specifies the desired height of the acquired frames, in pixels. If this
            argument is not explicitly provided, the instance uses the default frame width of the managed camera.
        color: Specifies whether the camera acquires colored or monochrome images. This determines how to store the
            acquired frames. Colored frames are saved using the 'BGR' channel order, monochrome images are reduced to
            a single-channel format. This argument is only used by the OpenCV and Mock camera interfaces, the
            Harvesters interface infers this information directly from the camera's configuration.
        cti_path: The path to the CTI file that provides the GenTL Producer interface. It is recommended to use the
            file supplied by the camera vendor, but a general Producer, such as mvImpactAcquire, is also acceptable.
            See https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst for more details. This argument is
            only used by the Harvesters camera interface.
        gpu: The index of the GPU to use for video encoding. Setting this argument to a value of -1 (default) configures
            the instance to use the CPU for encoding. Valid GPU indices can be obtained from the 'nvidia-smi' terminal
            command.
        video_encoder: The encoder to use for generating the video file. Must be one of the valid VideoEncoders
            enumeration members.
        encoder_speed_preset: The encoding speed preset to use for generating the video file. Must be one of the valid
            EncoderSpeedPresets enumeration members.
        output_pixel_format: The pixel format to be used by the output video file. Must be one of the valid
            OutputPixelFormats enumeration members.
        quantization_parameter: The integer value to use for the 'quantization parameter' of the encoder. This
            determines how much information to discard from each encoded frame. Lower values produce better video
            quality at the expense of longer processing time and larger file size: 0 is best, 51 is worst. Setting this
            argument to a value of -1 uses the default preset for the chosen encoder.

    Attributes:
        _started: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores the SyncManager instance used to control the multiprocessing assets (Queue and Lock
            instances).
        _system_id: Stores the unique identifier code of the VideoSystem instance.
        _output_file: Stores the path to the output .mp4 video file to be generated at runtime or None, if the instance
            is not configured to save acquired camera frames.
        _camera: Stores the camera interface class instance used to interface with the camera hardware at runtime.
        _saver: Stores the video saver instance used to save the acquired camera frames or None, if ths instance is
            not configured to save acquired camera frames.
        _logger_queue: Stores the multiprocessing Queue instance used to buffer frame acquisition timestamp data to the
            logger process.
        _image_queue: Stores the multiprocessing Queue instance used to buffer and pipe acquired frames from the
            camera (producer) process to the video saver (consumer) process.
        _terminator_array: Stores the SharedMemoryArray instance used to manage the runtime behavior of the producer
            and consumer processes.
        _producer_process: A process that acquires camera frames using the managed camera interface.
        _consumer_process: A process that saves the acquired frames using managed video saver.
        _watchdog_thread: A thread used to monitor the runtime status of the remote consumer and producer processes.

    Raises:
        TypeError: If any of the provided arguments has an invalid type.
        ValueError: If any of the provided arguments has an invalid value.
        RuntimeError: If the host system does not have access to FFMPEG or Nvidia GPU (when the instance is configured
            to use hardware encoding).
    """

    def __init__(
        self,
        system_id: np.uint8,
        data_logger: DataLogger,
        output_directory: Path | None,
        camera_interface: CameraInterfaces | str = CameraInterfaces.OPENCV,
        camera_index: int = 0,
        display_frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame_rate: int | None = None,
        cti_path: Path | None = None,
        gpu: int = -1,
        video_encoder: VideoEncoders | str = VideoEncoders.H265,
        encoder_speed_preset: EncoderSpeedPresets | int = EncoderSpeedPresets.SLOW,
        output_pixel_format: OutputPixelFormats | str = OutputPixelFormats.YUV444,
        quantization_parameter: int = -1,
        *,
        color: bool | None = None,
    ) -> None:
        # Has to be set first to avoid stop method errors
        self._started: bool = False  # Tracks whether the system has active processes

        # The manager is created early in the __init__ phase to support del-based cleanup
        self._mp_manager: SyncManager = Manager()

        # Ensures system_id is a byte-convertible integer
        self._system_id: np.uint8 = np.uint8(system_id)

        # If cti_path is provided, checks if it is a valid file path.
        if cti_path is not None and (not cti_path.exists() or cti_path.suffix != ".cti"):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected the path to an existing "
                f"file with a '.cti' suffix or None as the 'cti_path' argument value, but encountered "
                f"{cti_path} of type {type(cti_path).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Ensures that the data_logger is an initialized DataLogger instance.
        if not isinstance(data_logger, DataLogger):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected an initialized "
                f"DataLogger instance as the 'data_logger' argument value, but encountered {data_logger} of type "
                f"{type(data_logger).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Ensures that the output_directory is either a Path instance or None:
        if output_directory is not None and not isinstance(output_directory, Path):
            message = (
                f"Unable to initialize the VideoSystem instance with id {system_id}. Expected a Path instance or None "
                f"as the 'output_directory' argument's value, but encountered {output_directory} of type "
                f"{type(output_directory).__name__}."
            )
            console.error(message=message, error=TypeError)

        # If the output directory is provided, resolves the path to the output .mp4 video file to be created during
        # runtime.
        self._output_file: Path | None = (
            None if output_directory is None else output_directory.joinpath(f"{system_id:03d}.mp4")
        )

        # Initializes the camera interface:

        # Validates camera-related inputs:
        if not isinstance(camera_index, int) or camera_index < 0:
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Expected a "
                f"zero or positive integer as the 'camera_id' argument value, but got {camera_index} of type "
                f"{type(camera_index).__name__}."
            )
            console.error(error=TypeError, message=message)
        if (frame_rate is not None and not isinstance(frame_rate, int)) or (
            isinstance(frame_rate, int) and frame_rate <= 0
        ):
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Expected a "
                f"positive integer or None as the 'frame_rate' argument value, but got "
                f"{frame_rate} of type {type(frame_rate).__name__}."
            )
            console.error(error=TypeError, message=message)
        if (frame_width is not None and not isinstance(frame_width, int)) or (
            isinstance(frame_width, int) and frame_width <= 0
        ):
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Expected a "
                f"positive integer or None as the 'frame_width' argument value, but got {frame_width} of type "
                f"{type(frame_width).__name__}."
            )
            console.error(error=TypeError, message=message)
        if (frame_height is not None and not isinstance(frame_height, int)) or (
            isinstance(frame_height, int) and frame_height <= 0
        ):
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Expected a "
                f"positive integer or None as the 'frame_height' argument value, but got {frame_height} of type "
                f"{type(frame_height).__name__}."
            )
            console.error(error=TypeError, message=message)

        # Presets the variable type
        self._camera: OpenCVCamera | HarvestersCamera | MockCamera

        # OpenCVCamera
        if camera_interface == CameraInterfaces.OPENCV:
            # Instantiates the OpenCVCamera object
            self._camera = OpenCVCamera(
                system_id=int(self._system_id),
                color=False if not isinstance(color, bool) else color,
                camera_index=camera_index,
                frame_height=frame_height,
                frame_width=frame_width,
                frame_rate=frame_rate,
            )

        # HarvestersCamera
        elif camera_interface == CameraInterfaces.HARVESTERS:
            # Ensures that the CTI file path is provided
            if cti_path is None:
                message = (
                    f"Unable to configure the HarvestersCamera interface for the VideoSystem with id "
                    f"{self._system_id}. Expected the VideoSystem's 'harvesters_cti_path' argument to be a Path object "
                    f"pointing to the '.cti' file, but got None instead."
                )
                console.error(error=ValueError, message=message)
                # Fallback to appease mypy, should not be reachable
                raise ValueError(message)  # pragma: no cover

            # Instantiates the HarvestersCamera object
            self._camera = HarvestersCamera(
                system_id=int(self._system_id),
                cti_path=cti_path,
                camera_index=camera_index,
                frame_height=frame_height,
                frame_width=frame_width,
                frame_rate=frame_rate,
            )

        # MockCamera
        elif camera_interface == CameraInterfaces.MOCK:
            # Instantiates the MockCamera object
            self._camera = MockCamera(
                system_id=int(self._system_id),
                frame_height=frame_height,
                frame_width=frame_width,
                frame_rate=frame_rate,
                color=False if not isinstance(color, bool) else color,
            )

        # If the requested camera interface does not match any of the supported interfaces, raises an error
        else:
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Encountered "
                f"an unsupported camera_interface argument value {camera_interface} of type "
                f"{type(camera_interface).__name__}. Use one of the supported CameraInterfaces enumeration members: "
                f"{', '.join(tuple(camera_interface))}."
            )
            console.error(error=ValueError, message=message)
            # Fallback to appease mypy, should not be reachable
            raise ValueError(message)  # pragma: no cover

        # Connects to the camera. This both verifies that the camera can be connected to and applies the camera
        # acquisition parameters.
        self._camera.connect()

        # Verifies that the frame acquisition works as expected.
        self._camera.grab_frame()

        # Disconnects from the camera. The camera is re-connected by the remote producer process once it is
        # instantiated.
        self._camera.disconnect()

        # Disables frame displaying on macOS until OpenCV backend issues are fixed
        if display_frame_rate is not None and "darwin" in sys.platform:
            warnings.warn(
                message=(
                    f"Displaying frames is currently not supported for Apple Silicon devices. See ReadMe for details. "
                    f"Disabling frame display for the VideoSystem with id {self._system_id}."
                ),
                stacklevel=2,
            )
            display_frame_rate = None

        # If the system is configured to display the acquired frames to the user, ensures that the display frame rate
        # is valid and works with the managed camera's frame acquisition rate.
        if (display_frame_rate is not None and not isinstance(display_frame_rate, int)) or (
            isinstance(display_frame_rate, int)
            and (display_frame_rate <= 0 or display_frame_rate > self._camera.frame_rate)
        ):
            message = (
                f"Unable to configure the camera interface for the VideoSystem with id {self._system_id}. Encountered "
                f"an unsupported 'display_frame_rate' argument value {display_frame_rate} of type "
                f"{type(display_frame_rate).__name__}. The display frame rate override has to be None or a positive "
                f"integer that does not exceed the camera acquisition frame rate ({self._camera.frame_rate})."
            )
            console.error(error=TypeError, message=message)

        # Ensures that the display frame rate is stored as an integer and saves it to an attribute.
        self._display_frame_rate: int = display_frame_rate if display_frame_rate is not None else 0

        # Only adds the video saver if the user intends to save the acquired frames (as indicated by providing a valid
        # output directory).
        self._saver: VideoSaver | None = None
        if output_directory is not None:
            # Validates the video saver configuration parameters:
            if not isinstance(gpu, int):
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. Expected an "
                    f"integer as the 'gpu' argument value, but got {gpu} of type {type(gpu).__name__}."
                )
                console.error(error=TypeError, message=message)
            if video_encoder not in VideoEncoders:
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. Encountered "
                    f"an unexpected 'video_encoder' argument value {video_encoder} of type "
                    f"{type(video_encoder).__name__}. Use one of the supported VideoEncoders enumeration members: "
                    f"{', '.join(tuple(VideoEncoders))}."
                )
                console.error(error=ValueError, message=message)
            if encoder_speed_preset not in EncoderSpeedPresets:
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. Encountered "
                    f"an unexpected 'encoder_speed_preset' argument value {encoder_speed_preset} of type "
                    f"{type(encoder_speed_preset).__name__}. Use one of the supported EncoderSpeedPresets enumeration "
                    f"members: {', '.join(tuple(EncoderSpeedPresets))}."
                )
                console.error(error=ValueError, message=message)
            if output_pixel_format not in OutputPixelFormats:
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. Encountered "
                    f"an unexpected 'output_pixel_format' argument value {output_pixel_format} of type "
                    f"{type(output_pixel_format).__name__}. Use one of the supported OutputPixelFormats enumeration "
                    f"members: {', '.join(tuple(OutputPixelFormats))}."
                )
                console.error(error=ValueError, message=message)
            if (
                not isinstance(quantization_parameter, int)
                or not -1 < quantization_parameter <= _MAXIMUM_QUANTIZATION_VALUE
            ):
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. Expected an "
                    f"integer between -1 and 51 as the 'quantization_parameter' argument value, but got "
                    f"{quantization_parameter} of type {type(quantization_parameter).__name__}."
                )
                console.error(error=TypeError, message=message)

            # VideoSaver relies on the FFMPEG library to be available on the system Path. Ensures that FFMPEG is
            # available for this runtime.
            if not check_ffmpeg_availability():
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. VideoSaver "
                    f"requires a third-party software, FFMPEG, to be available on the system's Path. Make sure FFMPEG "
                    f"is installed and callable from a Python shell. See https://www.ffmpeg.org/download.html for more "
                    f"information."
                )
                console.error(error=RuntimeError, message=message)

            # Since GPU encoding is currently only supported for NVIDIA GPUs, verifies that nvidia-smi is callable
            # for the host system. This is used as a proxy to determine whether the system has an Nvidia GPU:
            if gpu >= 0 and not check_gpu_availability():
                message = (
                    f"Unable to configure the video saver for the VideoSystem with id {self._system_id}. The saver is "
                    f"configured to use the GPU video encoder, which currently only supports NVIDIA GPUs. Calling "
                    f"'nvidia-smi' to verify the presence of NVIDIA GPUs did not run successfully, indicating "
                    f"that there are no available NVIDIA GPUs on the host system. Use a CPU encoder or make sure "
                    f"nvidia-smi is callable from a Python shell."
                )
                console.error(error=RuntimeError, message=message)

            # Instantiates the VideoSaver object
            self._saver = VideoSaver(
                system_id=int(system_id),
                output_file=self._output_file,
                frame_height=self._camera.frame_height,
                frame_width=self._camera.frame_width,
                frame_rate=self._camera.frame_rate,
                gpu=gpu,
                video_encoder=video_encoder,
                encoder_speed_preset=encoder_speed_preset,
                input_pixel_format=self._camera.pixel_color_format,
                output_pixel_format=output_pixel_format,
                quantization_parameter=quantization_parameter,
            )

        # Sets up the assets used to manage acquisition and saver processes. The assets are configured during the
        # start() method runtime, most of them are initialized to placeholder values here.
        self._logger_queue: MPQueue = data_logger.input_queue
        self._image_queue: MPQueue = self._mp_manager.Queue()
        self._terminator_array: SharedMemoryArray | None = None
        self._producer_process: Process | None = None
        self._consumer_process: Process | None = None
        self._watchdog_thread: Thread | None = None

    def __del__(self) -> None:
        """Releases all reserved resources before the instance is garbage-collected."""
        self.stop()
        self._mp_manager.shutdown()

    def __repr__(self) -> str:
        """Returns the string representation of the VideoSystem instance."""
        return (
            f"VideoSystem(system_id={self._system_id}, started={self._started}, "
            f"camera={str(type(self._camera).__name__)}, frame_saving={self._saver is not None})"
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
                f"Unable to start the VideoSystem with id {self._system_id}. The VideoSystem must be equipped with a Camera "
                f"before it can be started. Use add_camera() method to add a Camera class to the VideoSystem. If you "
                f"need to convert a directory of images to video, use the encode_video_from_images() method instead."
            )
            console.error(error=RuntimeError, message=message)

        # If the camera is configured to save frames, ensures it is matched with a Saver class instance.
        elif self._camera.save_frames and self._saver is None:
            message = (
                f"Unable to start the VideoSystem with id {self._system_id}. The managed Camera is configured to save frames "
                f"and has to be matched to a Saver instance, but no Saver was added. Use add_image_saver() or "
                f"add_video_saver() method to add a Saver instance to save camera frames."
            )
            console.error(error=RuntimeError, message=message)

        # Instantiates an array shared between all processes. This array is used to control all child processes.
        # Index 0 (element 1) is used to issue global process termination command
        # Index 1 (element 2) is used to flexibly enable or disable saving camera frames.
        # Index 2 (element 3) is used to track VideoSystem initialization status.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._system_id}_terminator_array",  # Uses class id with an additional specifier
            prototype=np.zeros(shape=3, dtype=np.uint8),
            exists_ok=True,  # Automatically recreates the buffer if it already exists
        )  # Instantiation automatically connects the main process to the array.

        # Only starts the consumer process if the managed camera is configured to save frames.
        if self._saver is not None:
            # Starts the consumer process first to minimize queue buildup once the producer process is initialized.
            # Technically, due to saving frames being initially disabled, queue buildup is no longer a major concern,
            # but this safety-oriented initialization order is still preserved.
            self._consumer_process = Process(
                target=self._frame_saving_loop,
                args=(
                    self._system_id,
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
                        f"Unable to start the VideoSystem with id {self._system_id}. The consumer process has "
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
                self._system_id,
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
                    f"Unable to start the VideoSystem with id {self._system_id}. The producer process has "
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
        # Ensures that the stop procedure is only executed if the class is running
        if not self._started or self._terminator_array is None:
            # Terminator array cannot be None if the process has started, so this check is to appease mypy.
            return

        # This timer is used to forcibly terminate the process that gets stuck in the shutdown sequence.
        shutdown_timer = PrecisionTimer(precision="s")

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

    @staticmethod
    def _frame_display_loop(
        display_queue: Queue,
        system_id: int,
    ) -> None:  # pragma: no cover
        """Continuously fetches frame images from the display_queue and displays them via the OpenCV's imshow()
        function.

        Notes:
            This method runs in a thread as part of the _produce_images_loop() runtime in the producer Process.

        Args:
            display_queue: The multithreading Queue that buffers the grabbed camera frames until they are displayed.
            system_id: The unique identifier of the VideoSystem that generated the visualized images.
        """
        # Initializes the display window using the 'normal' mode to support user-controlled resizing.
        window_name = f"VideoSystem {system_id} Frames."
        cv2.namedWindow(winname=window_name, flags=cv2.WINDOW_NORMAL)

        # Runs until manually terminated by the user through the GUI or programmatically through the thread kill
        # argument.
        while True:
            # It is safe to fetch the frames in the blocking mode, since the loop is terminated by passing 'None'
            # through the queue
            frame = display_queue.get()

            # Programmatic termination is done by passing a non-numpy-array input through the queue
            if not isinstance(frame, np.ndarray):
                display_queue.task_done()  # If the thread is terminated, ensures join() works as expected
                break

            # Displays the image using the window created above
            cv2.imshow(winname=window_name, mat=frame)

            # Manual termination is done through the window GUI
            if cv2.waitKey(1) & 0xFF == 27:
                display_queue.task_done()  # If the thread is terminated, ensures join() works as expected
                break

            # Ensures that each queue get() call is paired with a task_done() call once display cycle is over
            display_queue.task_done()

        # Cleans up after runtime by destroying the window. Specifically targets the window created by this thread to
        # avoid interfering with any other windows.
        cv2.destroyWindow(winname=window_name)

    @staticmethod
    def _frame_production_loop(
        system_id: np.uint8,
        camera: OpenCVCamera | HarvestersCamera | MockCamera,
        display_frame_rate: int,
        image_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
    ) -> None:  # pragma: no cover
        """Continuously grabs frames from the input camera and queues them up to be saved by the consumer process.

        Notes:
            This method should be executed by the producer Process. It is not intended to be executed by the main
            process where the VideoSystem is instantiated.

            This method displays the acquired frames in a separate display thread if the instance is configured to
            display acquired frame data.

        Args:
            system_id: The unique identifier code of the caller VideoSystem instance. This is used to identify the
                VideoSystem in data log entries.
            camera: The camera interface instance for the camera from which to acquire frames.
            display_frame_rate: The desired rate, in frames per second, at which to display (visualize) the acquired
                frame stream to the user. Setting this argument to 0 disables frame display functionality.
            image_queue: A multiprocessing Queue that buffers and pipes acquired frames to the consumer process.
            logger_queue: The multiprocessing Queue that buffers and pipes log entries to the DataLogger's logger
                process.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during the global shutdown.
        """
        # Connects to the terminator array.
        terminator_array.connect()

        # Creates a timer that time-stamps acquired frames.
        frame_timer: PrecisionTimer = PrecisionTimer("us")

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all future time
        # readouts.
        onset: NDArray[np.uint8] = get_timestamp(output_format=TimestampFormats.BYTES)
        frame_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # Sends the onset data to the logger queue. The acquisition_time of 0 is universally interpreted as the timer
        # onset.
        logger_queue.put(LogPackage(source_id=system_id, acquisition_time=np.uint64(0), serialized_data=onset))

        # If the camera is configured to display frames, creates a worker thread and a queue object that handles
        # displaying the frames.
        show_time: float | None = None
        show_timer: PrecisionTimer | None = None
        display_queue: Queue | None = None
        display_thread: Thread | None = None
        if display_frame_rate > 0:
            # Creates the queue and thread for displaying camera frames
            display_queue = Queue()
            display_thread = Thread(target=VideoSystem._frame_display_loop, args=(display_queue, system_id))
            display_thread.start()

            # Converts the frame display rate from frames per second to microseconds per frame. This gives the delay
            # between displaying any two consecutive frames, which is used to limit how frequently the displayed image
            # updates.
            show_time = convert_time(
                time=1 / display_frame_rate,
                from_units="s",
                to_units="us",
                as_float=True
            )
            show_timer = PrecisionTimer("us")

        camera.connect()  # Connects to the hardware of the camera.

        # Indicates that the camera interface has started successfully.
        terminator_array[2] = 1

        try:
            # The loop runs until the VideoSystem is terminated by setting the first element (index 0) of the array to 1
            while not terminator_array[0]:

                # Grabs the first available frame as a numpy array. For Harvesters and Mock interfaces, this method
                # blocks until the frame is available if it is called too early. For OpenCV interface, this method
                # returns the same frame as grabbed during the previous call.
                frame = camera.grab_frame()
                frame_stamp = frame_timer.elapsed  # Generates the time-stamp for the acquired frame

                # If the camera is configured to display acquired frames, queues each frame to be displayed. The rate
                # at which the frames are displayed does not have to match the rate at which they are acquired.
                if display_queue is not None and show_timer.elapsed >= show_time:
                    show_timer.reset()  # Resets the display timer
                    display_queue.put(frame)

                # If frame saving is enabled, sends the acquired frame data and the acquisition timestamp to the
                # consumer (video saver) process.
                if terminator_array[1] == 1:
                    image_queue.put((frame, frame_stamp))

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise

        # Ensures that local assets are always properly terminated
        finally:

            # Releases camera and shared memory assets.
            terminator_array.disconnect()
            camera.disconnect()

            # Terminates the display thread
            if display_queue is not None:
                display_queue.put(None)

            # Waits for the thread to close
            if display_thread is not None:
                display_thread.join()

    @staticmethod
    def _frame_saving_loop(
        video_system_id: np.uint8,
        saver: VideoSaver,
        image_queue: MPQueue,
        logger_queue: MPQueue,
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
            image_queue: A multiprocessing queue that buffers and pipes acquired frames to the consumer process.
            logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
                This queue is used to buffer and pipe data to be logged to the logger cores.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during global shutdown.
        """
        # Connects to the terminator array used to manage the loop runtime.
        terminator_array.connect()

        # Initializes the FFMPEG encoder process.
        saver.start()

        # Indicates that the video saver has started successfully.
        terminator_array[2] = 2

        # Precreates the placeholder array used to log frame acquisition timestamps.
        data_placeholder = np.array([], dtype=np.uint8)

        try:
            # This loop runs until the global shutdown command is issued (via the variable under index 0) and until the
            # image_queue is empty.
            while not terminator_array[0] or not image_queue.empty():
                # Grabs the image bundled with its acquisition time from the queue and passes it to the video saver
                # instance.
                try:
                    frame: NDArray[np.integer[Any]]
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
                    acquisition_time=np.uint64(frame_time),
                    serialized_data=np.array(object=data_placeholder, dtype=np.uint8),
                )
                logger_queue.put(package)

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise

        # Ensures proper resource cleanup even during error shutdowns
        finally:
            # Disconnects from the shared memory array
            terminator_array.disconnect()

            # Carries out the necessary shut-down procedures:
            saver.stop()

    def _watchdog(self) -> None:  # pragma: no cover
        """This function should be used by the watchdog thread to ensure the producer and consumer processes are alive
        during runtime.

        This function will raise a RuntimeError if it detects that a monitored process has prematurely shut down. It
        will verify process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):
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
                self._terminator_array.write_data(index=0, data=np.uint8(1))
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
                        f"The producer process for the VideoSystem with id {self._system_id} has been prematurely "
                        f"shut down. This likely indicates that the process has encountered a runtime error that "
                        f"terminated the process."
                    )
                    console.error(message=message, error=RuntimeError)

                else:
                    message = (
                        f"The consumer process for the VideoSystem with id {self._system_id} has been prematurely "
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

    @property
    def video_file(self) -> Path | None:
        """Returns the path to the output video file if the instance is configured to save acquired camera frames and
        None otherwise.
        """
        return self._output_file if self._saver is not None else None

    @property
    def started(self) -> bool:
        """Returns True if the system has been started and has active producer and (optionally) consumer processes."""
        return self._started

    @property
    def system_id(self) -> np.uint8:
        """Returns the unique identifier code assigned to the VideoSystem instance."""
        return self._system_id


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
