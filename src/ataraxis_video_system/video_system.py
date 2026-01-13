"""Provides the main VideoSystem class that contains methods for setting up, running, and tearing down interactions
between camera interfaces and video saver instances.

All user-oriented functionality of this library is available through the public methods of the VideoSystem class.
"""

import sys
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any
from pathlib import Path
import warnings
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
    cpu_count,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm
import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, convert_time, get_timestamp
from ataraxis_base_utilities import console, chunk_iterable
from ataraxis_data_structures import DataLogger, LogPackage, SharedMemoryArray

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


# Determines the maximum qp value used when initializing VideoSystem instances.
_MAXIMUM_QUANTIZATION_VALUE = 51

# The maximum number of seconds to wait for the consumer and producer processes to initialize.
_PROCESS_INITIALIZATION_TIME = 20

# The maximum number of seconds to wait for the consumer and producer processes to shut down.
_PROCESS_SHUTDOWN_TIME = 600

# Specifies the size of the frame timestamp log message, in bytes. This is used during log message parsing to extract
# the acquisition timestamps for all saved frames.
_FRAME_TIMESTAMP_LOG_MESSAGE_SIZE = 9

# The minimum number of messages that must be contained in the camera frame timestamp .npz archive for it to be
# processed in parallel.
_MINIMUM_LOG_SIZE_FOR_PARALLELIZATION = 2000


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
            argument is not explicitly provided, the instance uses the default frame height of the managed camera.
        color: Specifies whether the camera acquires colored or monochrome images. This determines how to store the
            acquired frames. Colored frames are saved using the 'BGR' channel order, monochrome images are reduced to
            a single-channel format. This argument is only used by the OpenCV and Mock camera interfaces, the
            Harvesters interface infers this information directly from the camera's configuration.
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
            quality at the expense of longer processing time and larger file size: 0 is best, 51 is worst. Note, the
            default value is calibrated for the H265 encoder and is likely too low for the H264 encoder.

    Attributes:
        _started: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores the SyncManager instance used to control the multiprocessing assets (Queue and Lock
            instances).
        _system_id: Stores the unique identifier code of the VideoSystem instance.
        _output_file: Stores the path to the output .mp4 video file to be generated at runtime or None, if the instance
            is not configured to save acquired camera frames.
        _camera: Stores the camera interface class instance used to interface with the camera hardware at runtime.
        _saver: Stores the video saver instance used to save the acquired camera frames or None, if the instance is
            not configured to save acquired camera frames.
        _logger_queue: Stores the multiprocessing Queue instance used to buffer frame acquisition timestamp data to the
            logger process.
        _saver_queue: Stores the multiprocessing Queue instance used to buffer and pipe acquired frames from the
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
        gpu: int = -1,
        video_encoder: VideoEncoders | str = VideoEncoders.H265,
        encoder_speed_preset: EncoderSpeedPresets | int = EncoderSpeedPresets.SLOW,
        output_pixel_format: OutputPixelFormats | str = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        *,
        color: bool | None = None,
    ) -> None:
        # Has to be set first to avoid stop method errors
        self._started: bool = False  # Tracks whether the system has active processes

        # The manager is created early in the __init__ phase to support del-based cleanup
        self._mp_manager: SyncManager = Manager()

        # Ensures system_id is a byte-convertible integer
        self._system_id: np.uint8 = np.uint8(system_id)

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
            # Instantiates the HarvestersCamera object
            self._camera = HarvestersCamera(
                system_id=int(self._system_id),
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

        # Disables frame displaying on macOS as this OS does not support displaying frames outside the main thread.
        if display_frame_rate is not None and "darwin" in sys.platform:
            warnings.warn(
                message=(
                    f"Displaying frames is currently not supported for Apple Silicon devices. See ReadMe for details. "
                    f"Disabling frame display for the VideoSystem with id {self._system_id}."
                ),
                stacklevel=2,
            )
            display_frame_rate = None

        # Ensures that the display frame rate is stored as an integer and saves it to an attribute.
        self._display_frame_rate: int = display_frame_rate if display_frame_rate is not None else 0

        # Only adds the video saver if the user intends to save the acquired frames (as indicated by providing a valid
        # output directory).
        self._saver: VideoSaver | None = None
        if self._output_file is not None:
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
                    f"members: {', '.join([str(preset) for preset in tuple(EncoderSpeedPresets)])}."
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
                or not -1 <= quantization_parameter <= _MAXIMUM_QUANTIZATION_VALUE
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
        self._logger_queue: MPQueue = data_logger.input_queue  # type: ignore[type-arg]
        self._saver_queue: MPQueue = self._mp_manager.Queue()  # type: ignore[type-arg, assignment]
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
            f"camera={type(self._camera).__name__!s}, frame_saving={self._saver is not None})"
        )

    def start(self) -> None:
        """Starts the instance's producer (camera interface) and consumer (video saver) processes and begins acquiring
        camera frames.

        Notes:
            Calling this method does not enable saving camera frames to non-volatile memory. To enable saving camera
            frames, call the start_frame_saving() method.

        Raises:
            RuntimeError: If starting the consumer or producer processes stalls or fails.
        """
        # Prevents restarting an already started VideoSystem instance.
        if self._started:
            return

        # This timer is used to forcibly terminate processes that stall at initialization.
        initialization_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

        # Instantiates a SharedMemoryArray used to control the runtime of the child processes.
        # Index 0 (element 1) is used to issue the global process termination command.
        # Index 1 (element 2) is used to flexibly enable or disable saving camera frames.
        # Index 2 (element 3) is used to track the producer process initialization status.
        # Index 3 (element 4) is used to track the consumer process initialization status.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._system_id}_terminator_array",  # Uses class id with an additional specifier
            prototype=np.zeros(shape=4, dtype=np.uint8),
            exists_ok=True,  # Automatically recreates the buffer if it already exists
        )

        # Only starts the consumer process if the managed camera is configured to save frames.
        if self._saver is not None:
            self._consumer_process = Process(
                target=self._frame_saving_loop,
                args=(
                    self._system_id,
                    self._saver,
                    self._saver_queue,
                    self._logger_queue,
                    self._terminator_array,
                ),
                daemon=True,
            )
            self._consumer_process.start()

        # Starts the producer process
        self._producer_process = Process(
            target=self._frame_production_loop,
            args=(
                self._system_id,
                self._camera,
                self._display_frame_rate,
                self._saver_queue,
                self._logger_queue,
                self._terminator_array,
            ),
            daemon=True,
        )
        self._producer_process.start()

        # Connects to the shared memory array to receive control signals. It is important for this to be done after
        # both processes have been started.
        self._terminator_array.connect()
        # Ensures the buffer is destroyed if the instance is garbage-collected to prevent memory leaks.
        self._terminator_array.enable_buffer_destruction()

        # Waits for the processes to report that they have been successfully initialized.
        initialization_timer.reset()
        while self._terminator_array[2] != 1 and (
            self._consumer_process is not None and self._terminator_array[3] != 1
        ):  # pragma: no cover
            # If the processes take too long to initialize or die, raises an error.
            error = False
            message: str = ""  # Pre-initialization to appease mypy
            if (
                initialization_timer.elapsed > _PROCESS_INITIALIZATION_TIME and self._terminator_array[2] != 1
            ) or not self._producer_process.is_alive():
                message = (
                    f"Unable to start the VideoSystem with id {self._system_id}. The producer process has "
                    f"unexpectedly shut down or stalled for more than {_PROCESS_INITIALIZATION_TIME} seconds "
                    f"during initialization. This likely indicates a problem with the camera interface instance "
                    f"managed by the process."
                )
                error = True
            elif self._consumer_process is not None and (
                (initialization_timer.elapsed > _PROCESS_INITIALIZATION_TIME and self._terminator_array[3] != 1)
                or not self._consumer_process.is_alive()
            ):
                message = (
                    f"Unable to start the VideoSystem with id {self._system_id}. The consumer process has "
                    f"unexpectedly shut down or stalled for more than {_PROCESS_INITIALIZATION_TIME} seconds "
                    f"during initialization. This likely indicates a problem with the VideoSaver instance managed "
                    f"by the process."
                )
                error = True

            # Reclaims all committed resources before terminating with an error.
            if error:
                # Emits the process termination command
                self._terminator_array[0] = 1

                # Waits for any active processes to finish their execution
                if self._consumer_process is not None:
                    self._consumer_process.join()
                self._producer_process.join()

                # Disconnects from and destroys the shared memory array buffer
                self._terminator_array.disconnect()
                self._terminator_array.destroy()

                console.error(error=RuntimeError, message=message)

        # Creates and starts the watchdog thread
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        # Sets the _started flag, which also activates the watchdog monitoring.
        self._started = True

    def stop(self) -> None:
        """Stops the instance's producer (camera interface) and consumer (video saver) processes and releases all
        reserved resources.

        Notes:
            The consumer process is kept alive until all frames buffered to the saver_queue are saved. However, if the
            saver_queue does not become empty within 10 minutes from calling this method, it forcibly terminates the
            consumer process and discards any unprocessed data.
        """
        # Prevents stopping an already stopped VideoSystem instance.
        if not self._started or self._terminator_array is None:
            return

        # This timer is used to forcibly terminate the process that gets stuck in the shutdown sequence.
        shutdown_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

        # This inactivates the watchdog thread monitoring, ensuring it does not err when the processes are terminated.
        self._started = False

        # Emits the process shutdown signal.
        self._terminator_array[0] = 1

        # Delays for 2 seconds to allow the consumer process to terminate its runtime
        shutdown_timer.delay(delay=2, allow_sleep=True, block=False)

        # Waits until the saver_queue is empty. This is aborted if the shutdown stalls at this step for 10+ minutes.
        while not self._saver_queue.empty():
            if shutdown_timer.elapsed > _PROCESS_SHUTDOWN_TIME:
                break

        # Joins the producer and consumer processes
        if self._producer_process is not None:
            self._producer_process.join(timeout=20)
        if self._consumer_process is not None:
            self._consumer_process.join(timeout=20)

        # Joins the watchdog thread
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=20)

        # Disconnects from and destroys the terminator array buffer.
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

    @staticmethod
    def _frame_display_loop(
        display_queue: Queue,  # type: ignore[type-arg]
        system_id: np.uint8,
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
            escape_key = 27  # The code for the ESC key in ASCII
            if cv2.waitKey(1) & 0xFF == escape_key:
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
        saver_queue: MPQueue,  # type: ignore[type-arg]
        logger_queue: MPQueue,  # type: ignore[type-arg]
        terminator_array: SharedMemoryArray,
    ) -> None:  # pragma: no cover
        """Continuously grabs frames from the managed camera and queues them up to be saved by the consumer process.

        If the VideoSystem instance is configured to display acquired frame data, this method also uses a separate
        thread to render the acquired frames into and display them to the user in real time.

        Notes:
            This method should be executed by the producer Process. It is not intended to be executed by the main
            process where the VideoSystem is instantiated.

        Args:
            system_id: The unique identifier code of the caller VideoSystem instance. This is used to identify the
                VideoSystem in data log entries.
            camera: The camera interface instance for the camera from which to acquire frames.
            display_frame_rate: The desired rate, in frames per second, at which to display (visualize) the acquired
                frame stream to the user. Setting this argument to 0 disables frame display functionality.
            saver_queue: The multiprocessing Queue that buffers and pipes acquired frames to the consumer process.
            logger_queue: The multiprocessing Queue that buffers and pipes log entries to the DataLogger's logger
                process.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during the global shutdown.
        """
        # Connects to the terminator array.
        terminator_array.connect()

        # Creates a timer that time-stamps acquired frames.
        frame_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MICROSECOND)

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all future time
        # readouts.
        onset: NDArray[np.uint8] = get_timestamp(output_format=TimestampFormats.BYTES)  # type: ignore[assignment]
        frame_timer.reset()  # Immediately resets the stamp timer to make it as close as possible to the onset time

        # Sends the onset data to the logger queue. The acquisition_time of 0 is universally interpreted as the timer
        # onset.
        logger_queue.put(LogPackage(source_id=system_id, acquisition_time=np.uint64(0), serialized_data=onset))

        # If the camera is configured to display frames, creates a worker thread and a queue object that handles
        # displaying the frames.
        show_time: float | None = None
        show_timer: PrecisionTimer | None = None
        display_queue: Queue | None = None  # type: ignore[type-arg]
        display_thread: Thread | None = None
        if display_frame_rate > 0:
            # Creates the queue and thread for displaying camera frames
            display_queue = Queue()
            display_thread = Thread(target=VideoSystem._frame_display_loop, args=(display_queue, system_id))
            display_thread.start()

            # Converts the frame display rate from frames per second to microseconds per frame. This gives the delay
            # between displaying any two consecutive frames, which is used to limit how frequently the displayed image
            # updates.
            show_time = convert_time(time=1 / display_frame_rate, from_units="s", to_units="us", as_float=True)
            show_timer = PrecisionTimer(precision=TimerPrecisions.MICROSECOND)

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
                if display_queue is not None and show_timer.elapsed >= show_time:  # type: ignore[union-attr, operator]
                    # Resets the display timer
                    show_timer.reset()  # type: ignore[union-attr]
                    display_queue.put(frame)

                # If frame saving is enabled, sends the acquired frame data and the acquisition timestamp to the
                # consumer (video saver) process.
                if terminator_array[1] == 1:
                    saver_queue.put((frame, frame_stamp))

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
        system_id: np.uint8,
        saver: VideoSaver,
        saver_queue: MPQueue,  # type: ignore[type-arg]
        logger_queue: MPQueue,  # type: ignore[type-arg]
        terminator_array: SharedMemoryArray,
    ) -> None:  # pragma: no cover
        """Continuously grabs the frames from the image_queue and saves them as an .mp4 video file.

        This method also logs the acquisition time for each saved frame via the logger_queue instance.

        Notes:
            This method should be executed by the consumer Process. It is not intended to be executed by the main
            process where the VideoSystem is instantiated.

            This method's main loop is kept alive until the image_queue is empty. This is an intentional security
            feature that ensures all buffered images are processed before the saver is terminated. To override this
            behavior, you will need to use the process kill command, but it is strongly advised not to tamper
            with this feature.

        Args:
            system_id: The unique identifier code of the caller VideoSystem instance. This is used to identify the
                VideoSystem in data log entries.
            saver: The VideoSaver instance to use for saving the input frames as an .mp4 video file.
            saver_queue: The multiprocessing Queue that buffers and pipes acquired frames to the consumer process.
            logger_queue: The multiprocessing Queue that buffers and pipes log entries to the DataLogger's logger
                process.
            terminator_array: A SharedMemoryArray instance used to control the runtime behavior of the process
                and terminate it during the global shutdown.
        """
        # Connects to the terminator array used to manage the loop runtime.
        terminator_array.connect()

        # Initializes the FFMPEG encoder process.
        saver.start()

        # Indicates that the video saver has started successfully.
        terminator_array[3] = 1

        # Pre-creates the placeholder array used to log frame acquisition timestamps.
        data_placeholder = np.array([], dtype=np.uint8)

        try:
            # This loop runs until the global shutdown command is issued (via the variable under index 0) and until the
            # image_queue is empty.
            while not terminator_array[0] or not saver_queue.empty():
                # Grabs the frame data and its acquisition timestamp from the queue
                try:
                    frame: NDArray[np.integer[Any]]
                    frame_time: int
                    frame, frame_time = saver_queue.get_nowait()
                except Empty:
                    # Cycles the loop if the queue is empty
                    continue

                # Sends the frame to be saved by the saver
                saver.save_frame(frame)

                # Logs the saved frame's acquisition timestamp
                logger_queue.put(
                    LogPackage(
                        system_id,
                        acquisition_time=np.uint64(frame_time),
                        serialized_data=np.array(object=data_placeholder, dtype=np.uint8),
                    )
                )

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise

        # Ensures that local assets are always properly terminated
        finally:
            # Disconnects from the shared memory array
            terminator_array.disconnect()

            # Stops the encoder process
            saver.stop()

    def _watchdog(self) -> None:  # pragma: no cover
        """Monitors the producer and consumer processes to ensure they remain alive during runtime.

        Raises RuntimeErrors if any of the processes has prematurely shut down. Verifies the process state in
        20-millisecond cycles and releases the GIL between state verifications.

        Notes:
            If the method detects that the consumer or producer process has terminated prematurely, it carries out the
            necessary resource cleanup steps before raising the error and terminating the overall runtime.
        """
        # Initializes the timer used to space out the process state checks.
        timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # The watchdog function runs until the global shutdown signal is emitted.
        while self._terminator_array is not None and not self._terminator_array[0]:
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay(delay=20, allow_sleep=True, block=False)

            # The watchdog functionality only kicks-in after the VideoSystem has been started
            if not self._started:
                continue

            # Checks if the producer is alive
            error = False
            producer = False
            if self._producer_process is not None and not self._producer_process.is_alive():
                error = True
                producer = True

            # Checks if the consumer is alive
            if self._consumer_process is not None and not self._consumer_process.is_alive():
                error = True

            # If either consumer or producer is dead, ensures proper resource reclamation before terminating with an
            # error
            if error:
                # Reclaims all committed resources before terminating with an error.
                self._terminator_array[0] = 1

                # If the consumer process is alive, gives it time to finish processing any remaining frames.
                while (
                    self._consumer_process is not None
                    and not self._saver_queue.empty()
                    and self._consumer_process.is_alive()
                ):
                    if timer.elapsed > _PROCESS_SHUTDOWN_TIME * 1000:
                        break

                # Joins all processes
                if self._consumer_process is not None:
                    self._consumer_process.join()
                if self._producer_process is not None:
                    self._producer_process.join()

                # Disconnects from the shared memory array and destroys the shared memory buffer.
                if self._terminator_array is not None:
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()

                # The code above is equivalent to stopping the instance
                self._started = False

                # Raises the error.
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

    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames to disk as an .mp4 video file."""
        if self._started and self._terminator_array is not None:
            self._terminator_array[1] = 1

    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames to disk as an .mp4 video file.

        Notes:
            Calling this method does not stop the frame acquisition process. It only prevents the acquired frames from
            being sent to the consumer process, which prevents them from being saved to disk.
        """
        if self._started and self._terminator_array is not None:
            self._terminator_array[1] = 0

    @property
    def video_file_path(self) -> Path | None:
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


def _process_frame_message_batch(log_path: Path, file_names: list[str], onset_us: np.uint64) -> list[np.uint64]:
    """Processes the target batch of VideoSystem-generated messages stored in the .npz log file.

    This worker function is used by the _extract_camera_timestamps() function to process multiple message batches in
    parallel to speed up the overall camera timestamp data processing.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the frame data acquisition, in microseconds elapsed since UTC epoch onset.

    Returns:
        The list of frame acquisition timestamps for all frames whose messages have been processed as part of the
        batch, stored as microseconds since UTC epoch onset.
    """
    # Opens the processed log archive using memory mapping. If frame processing is performed in parallel, all processes
    # interact with the archive concurrently.
    with np.load(log_path, allow_pickle=False, fix_imports=False, mmap_mode="r") as archive:
        frame_timestamps = []

        # Loops over the batch of frame messages and extracts frame acquisition timestamps.
        for item in file_names:
            message = archive[item]

            # Frame timestamp messages do not have a payload, they only contain the source ID and the acquisition
            # timestamp. This gives them the length of 9 bytes.
            if len(message) == _FRAME_TIMESTAMP_LOG_MESSAGE_SIZE:
                # Extracts the number of microseconds elapsed since acquisition onset and uses it to calculate the
                # global timestamp for the message, in microseconds since UTC epoch onset.
                elapsed_microseconds = message[1:9].view(np.uint64).item()
                frame_timestamps.append(onset_us + elapsed_microseconds)

    return frame_timestamps


def extract_logged_camera_timestamps(
    log_path: Path,
    n_workers: int = -1,
) -> tuple[np.uint64, ...]:
    """Extracts the video camera frame acquisition timestamps from the target .npz log file generated by a VideoSystem
    instance during runtime.

    This function reads the '.npz' archive generated by the DataLogger's assemble_log_archives() method for a
    VideoSystem instance and, if the system saved any frames acquired by the managed camera, extracts the tuple of
    frame timestamps. The order of timestamps in the tuple is sequential and matches the order in which the frames were
    appended to the .mp4 video file.

    Notes:
        The timestamps are given as microseconds elapsed since the UTC epoch onset.

        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

    Args:
        log_path: The path to the .npz log file that stores the logged data generated by the VideoSystem
            instance during runtime.
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.

    Returns:
        A tuple that stores the frame acquisition timestamps. Each timestamp is stored as the number of microseconds
        elapsed since the UTC epoch onset.

    Raises:
        ValueError: If the target .npz archive does not exist.
    """
    # Ensures that the target .npz log archive exists.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        error_message = (
            f"Unable to extract camera frame timestamp data from the log file {log_path}, as it does not exist or does "
            f"not point to a valid .npz archive."
        )
        console.error(message=error_message, error=ValueError)

    # Memory-maps the processed archive to conserve RAM. The first processing pass is designed to find the onset
    # timestamp value.
    with np.load(log_path, allow_pickle=False, fix_imports=False, mmap_mode="r") as archive:
        # Locates the logging onset timestamp. The onset is used to convert the relative timestamps for logged frame
        # data into absolute UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in
        # microseconds relative to the onset timestamp.
        onset_us = np.uint64(0)
        timestamp_offset = 0
        message_list = list(archive.files)
        for number, item in enumerate(message_list):
            message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

            # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
            # message starting at index 1. If the timestamp value is 0, the message contains the onset timestamp value
            # stored as an 8-byte payload. Index 0 stores the source ID (uint8 value).
            timestamp_value = message[1:9].view(np.uint64).item()
            if timestamp_value == 0:
                # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
                onset_us = np.uint64(message[9:].view(np.int64).item())

                # Breaks the loop once the onset is found. Generally, the onset is expected to be found very early into
                # the loop.
                timestamp_offset = number  # Records the item number at which the onset value was found.
                break

    # Builds the list of files to process after discovering the timestamp (the list of remaining messages)
    messages_to_process = message_list[timestamp_offset + 1 :]

    # If there are no leftover messages to process, return an empty tuple
    if not messages_to_process:
        return ()

    # Small archives are processed sequentially to avoid the unnecessary overhead of setting up the multiprocessing
    # runtime. This is also done for large files if the user explicitly requests to use a single worker process.
    if n_workers == 1 or len(messages_to_process) < _MINIMUM_LOG_SIZE_FOR_PARALLELIZATION:
        return tuple(_process_frame_message_batch(log_path, messages_to_process, onset_us))

    # If the user enabled using all available cores, configures the runtime to use all available CPUs
    if n_workers < 0:
        n_workers = cpu_count()

    # Creates batches of messages to process during runtime. Uses a fairly high batch multiplier to create many smaller
    # batches, which leads to a measurable increase in the processing speed, especially for large archives. The optimal
    # multiplier value (4) was determined experimentally.
    batches = []
    batch_indices = []  # Keeps track of batch order
    for i, batch in enumerate(chunk_iterable(messages_to_process, n_workers * 4)):
        if batch:
            batches.append((log_path, list(batch), onset_us))
            batch_indices.append(i)

    # Processes batches using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submits all tasks
        future_to_index = {
            executor.submit(_process_frame_message_batch, *batch_args): idx
            for idx, batch_args in zip(batch_indices, batches, strict=False)
        }

        # Collects results while maintaining frame order. This also propagates processing errors to the caller process.
        results: list[list[np.uint64] | None] = [None] * len(batches)

        # Creates a progress bar for batch processing
        with tqdm(total=len(batches), desc="Extracting camera frame timestamps", unit="batch") as pbar:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
                pbar.update(1)  # Updates the progress bar after each batch completes

    # Combines processing results in order
    all_timestamps: list[np.uint64] = []
    for batch_timestamps in results:
        # noinspection PyUnreachableCode
        if batch_timestamps is not None:  # Skips None results
            all_timestamps.extend(batch_timestamps)

    return tuple(all_timestamps)
