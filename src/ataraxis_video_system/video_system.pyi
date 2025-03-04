from queue import Queue
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .saver import (
    ImageSaver as ImageSaver,
    VideoSaver as VideoSaver,
    VideoCodecs as VideoCodecs,
    ImageFormats as ImageFormats,
    VideoFormats as VideoFormats,
    CPUEncoderPresets as CPUEncoderPresets,
    GPUEncoderPresets as GPUEncoderPresets,
    InputPixelFormats as InputPixelFormats,
    OutputPixelFormats as OutputPixelFormats,
)
from .camera import (
    MockCamera as MockCamera,
    OpenCVCamera as OpenCVCamera,
    CameraBackends as CameraBackends,
    HarvestersCamera as HarvestersCamera,
)

@dataclass(frozen=True)
class _CameraSystem:
    """Stores a Camera class instance managed by the VideoSystem class, alongside additional runtime parameters.

    This class is used as a container that aggregates all objects and parameters required by the VideoSystem to
    interface with a camera during runtime.
    """

    camera: OpenCVCamera | HarvestersCamera | MockCamera
    save_frames: bool
    fps_override: int | float
    output_frames: bool
    output_frame_rate: int | float
    display_frames: bool
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

    _started: bool
    _mp_manager: Incomplete
    _id: Incomplete
    _logger_queue: Incomplete
    _log_directory: Incomplete
    _output_directory: Incomplete
    _cti_path: Incomplete
    _camera: Incomplete
    _saver: Incomplete
    _image_queue: Incomplete
    _output_queue: Incomplete
    _terminator_array: Incomplete
    _producer_process: Incomplete
    _consumer_process: Incomplete
    _watchdog_thread: Incomplete
    def __init__(
        self,
        system_id: np.uint8,
        data_logger: DataLogger,
        output_directory: Path | None = None,
        harvesters_cti_path: Path | None = None,
    ) -> None: ...
    def __del__(self) -> None:
        """Ensures that all resources are released when the instance is garbage-collected."""
    def __repr__(self) -> str:
        """Returns a string representation of the VideoSystem class instance."""
    def add_camera(
        self,
        save_frames: bool,
        camera_index: int = 0,
        camera_backend: CameraBackends = ...,
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
    def add_image_saver(
        self,
        image_format: ImageFormats = ...,
        tiff_compression_strategy: int = ...,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = ...,
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
    def add_video_saver(
        self,
        hardware_encoding: bool = False,
        video_format: VideoFormats = ...,
        video_codec: VideoCodecs = ...,
        preset: GPUEncoderPresets | CPUEncoderPresets = ...,
        input_pixel_format: InputPixelFormats = ...,
        output_pixel_format: OutputPixelFormats = ...,
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
    def stop(self) -> None:
        """Stops the producer and consumer processes and releases all resources.

        The instance will be kept alive until all frames buffered to the image_queue are saved. This is an intentional
        security feature that prevents information loss.

        Notes:
            This method waits for at most 10 minutes for the output_queue and the image_queue to become empty. If the
            queues are not empty by that time, the method forcibly terminates the instance and discards any unprocessed
            data.
        """
    @property
    def started(self) -> bool:
        """Returns true if the system has been started and has active daemon processes connected to cameras and
        saver.
        """
    @property
    def system_id(self) -> np.uint8:
        """Returns the unique identifier code assigned to the VideoSystem class instance."""
    @property
    def output_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue object used by the system's producer process to send frames to other
        concurrently active processes.
        """
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
    @staticmethod
    def _frame_display_loop(display_queue: Queue, camera_id: int) -> None:
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
    @staticmethod
    def _frame_production_loop(
        video_system_id: np.uint8,
        camera_system: _CameraSystem,
        image_queue: MPQueue,
        output_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
    ) -> None:
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
    @staticmethod
    def _frame_saving_loop(
        video_system_id: np.uint8,
        saver: ImageSaver | VideoSaver,
        camera_system: _CameraSystem,
        image_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
    ) -> None:
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
    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the producer and consumer processes are alive
        during runtime.

        This function will raise a RuntimeError if it detects that a monitored process has prematurely shut down. It
        will verify process states every ~20 ms and will release the GIL between checking the states.
        """
    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames.

        Does not interfere with grabbing and displaying the frames to the user, this process is only stopped when the
        main stop() method is called.
        """
    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames.

        The frames are grabbed and (optionally) displayed to the user after the main start() method is called, but they
        are not initially saved to disk. The call to this method additionally enables saving the frames to disk
        """
    def stop_frame_output(self) -> None:
        """Disables outputting frame data via the instance output_queue."""
    def start_frame_output(self) -> None:
        """Enables outputting frame data via the instance output_queue.

        Some cameras can be configured to additionally share acquired frame data with other concurrently active
        processes. When the VideoSystem starts, this functionality is not enabled by default and has to be enabled
        separately via this method.
        """
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
    @property
    def output_directory(self) -> Path | None:
        """Returns the path to the directory where the Saver managed by the VideoSystem outputs acquired frames as
        images or video file.

        If the VideoSystem does not have a Saver, it returns None to indicate that there is no valid output directory.
        """
    @property
    def log_path(self) -> Path:
        """Returns the path to the compressed .npz log archive that would be generated for the VideoSystem by the
        DataLogger instance given to the class at initialization.

        Primarily, this path should be used as an argument to the instance-independent
        'extract_logged_video_system_data' data extraction function.
        """

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
