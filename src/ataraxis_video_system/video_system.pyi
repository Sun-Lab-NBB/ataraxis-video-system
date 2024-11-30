from queue import Queue
from pathlib import Path
from multiprocessing import Queue as MPQueue

from _typeshed import Incomplete
from ataraxis_data_structures import SharedMemoryArray

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

class VideoSystem:
    """Efficiently combines Camera and Saver classes to acquire and save camera frames in real time.

    This class exposes methods for instantiating Camera and Saver classes, which, in turn, can be used to instantiate
    a VideoSystem class. The class achieves two main objectives: It efficiently moves frames acquired by the Camera
    class to be saved by the Saver class and manages runtime behavior of the bound classes. To do so, the class
    initializes and controls multiple subprocesses to ensure frame producer and consumer classes have sufficient
    computational resources. The class abstracts away all necessary steps to set up, execute, and tear down the
    processes through an easy-to-use API.

    Notes:
        Due to using multiprocessing to improve efficiency, this class needs a minimum of 2 logical cores per class
        instance to operate efficiently. Additionally, due to a time-lag of moving frames from a producer process to a
        consumer process, the class will reserve a variable portion of the RAM to buffer the frame images. The reserved
        memory depends on many factors and can only be known during runtime.

        This class requires Camera and Saver class instances during instantiation. To initialize
        these classes, use create_camera(), create_image_saver(), and create_video_saver() static methods available
        from the VideoSystem class.

        This class is written using a 'one-shot' design: start() and stop() methods can only be called one time. The
        class ahs to be re-initialized to be able to cycle these methods again. This is a conscious design decision that
        ensures proper resource release and cleanup, which was deemed important for our use pattern of this class.

    Args:
        camera: An initialized Camera class instance that interfaces with one of the supported cameras and allows
            acquiring camera frames. Use create_camera() method to instantiate this class.
        saver: An initialized Saver class instance that interfaces with OpenCV or FFMPEG and allows saving camera
            frames as images or videos. Use create_image_saver() or create_video_saver() methods to instantiate this
            class.
        system_name: A unique identifier for the VideoSystem instance. This is used to identify the system in messages,
            logs, and generated video files.
        image_saver_process_count: The number of ImageSaver processes to use. This parameter is only used when the
            saver class is an instance of ImageSaver. Since each saved image is independent of all other images, the
            performance of ImageSvers can be improved by using multiple processes with multiple threads to increase the
            saving throughput. For most use cases, a single saver process will be enough.
        fps_override: The number of frames to grab from the camera per second. This argument allows optionally
            overriding the frames per second (fps) parameter of the Camera class. When provided, the frame acquisition
            process will only trigger the frame grab procedure at the specified interval. Generally, this override
            should only be used for cameras with a fixed framerate or cameras that do not have framerate control
            capability at all. For cameras that support onboard framerate control, setting the fps through the Camera
            class will almost always be better. The override will not be able to exceed the acquisition speed enforced
            by the camera hardware.
        shutdown_timeout: The number of seconds after which non-terminated processes will be forcibly terminated during
            class shutdown. When class shutdown is triggered by calling stop() method, it first attempts to shut the
            processes gracefully, which may require some time. Primarily, this is because the consumer process tries
            to save all buffered images before it is allowed to shut down. If this process exceeds the specified
            timeout, the class will discard all unsaved data by forcing the processes to terminate.
        display_frames: Determines whether to display acquired frames to the user. This allows visually monitoring
            the camera feed in real time, which is frequently desirable in scientific experiments.

    Attributes:
        _camera: Stores the Camera class instance that provides camera interface.
        _saver: Stores the Saver class instance that provides saving backend interface.
        _shutdown_timeout: Stores the time in seconds after which to forcefully terminate processes
            during shutdown.
        _running: Tracks whether the system is currently running (has active subprocesses).
        _mp_manager: Stores a SyncManager class instance from which the image_queue and the log file Lock are derived.
        _image_queue: A cross-process Queue class instance used to buffer and pipe acquired frames from producer to
            consumer processes.
        _terminator_array: A SharedMemoryArray instance that provides a cross-process communication interface used to
            manage runtime behavior of spawned processes.
        _producer_process: A process that acquires camera frames using the bound Camera class.
        _consumer_processes: A tuple of processes that save camera frames using the bound Saver class. ImageSaver
            instance can use multiple processes, VideoSaver will use a single process.

    Raises:
        TypeError: If any of the provided arguments has an invalid type.
        ValueError: If any of the provided arguments has an invalid value.
    """

    _camera: Incomplete
    _saver: Incomplete
    _name: Incomplete
    _shutdown_timeout: Incomplete
    _running: bool
    _mp_manager: Incomplete
    _image_queue: Incomplete
    _expired: bool
    _terminator_array: Incomplete
    _producer_process: Incomplete
    _consumer_processes: Incomplete
    def __init__(
        self,
        camera: HarvestersCamera | OpenCVCamera | MockCamera,
        saver: VideoSaver | ImageSaver,
        system_name: str,
        image_saver_process_count: int = 1,
        fps_override: int | float | None = None,
        shutdown_timeout: int | None = 600,
        *,
        display_frames: bool = True,
    ) -> None: ...
    def __del__(self) -> None:
        """Ensures that all resources are released upon garbage collection."""
    def __repr__(self) -> str:
        """Returns a string representation of the VideoSystem class instance."""
    @property
    def is_running(self) -> bool:
        """Returns true oif the class has active subprocesses (is running) and false otherwise."""
    @property
    def name(self) -> str:
        """Returns the name of the VideoSystem class instance."""
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
            advised to use test each of the IDs with interactive-run CLI command to manually map IDs to cameras based
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
    @staticmethod
    def create_camera(
        camera_name: str,
        camera_backend: CameraBackends = ...,
        camera_id: int = 0,
        frame_width: int | None = None,
        frame_height: int | None = None,
        frames_per_second: int | float | None = None,
        opencv_backend: int | None = None,
        cti_path: Path | None = None,
        color: bool | None = None,
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
    @staticmethod
    def create_image_saver(
        output_directory: Path,
        image_format: ImageFormats = ...,
        tiff_compression: int = ...,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = ...,
        png_compression: int = 1,
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
            thread_count: The number of writer threads to be used by the saver class. Since
                ImageSaver uses the c-backed OpenCV library, it can safely process multiple frames at the same time
                via multithreading. This controls the number of simultaneously saved images the class will support.

        Raises:
            TypeError: If the input arguments are not of the correct type.
        """
    @staticmethod
    def create_video_saver(
        output_directory: Path,
        hardware_encoding: bool = False,
        video_format: VideoFormats = ...,
        video_codec: VideoCodecs = ...,
        preset: GPUEncoderPresets | CPUEncoderPresets = ...,
        input_pixel_format: InputPixelFormats = ...,
        output_pixel_format: OutputPixelFormats = ...,
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
    @staticmethod
    def _frame_production_loop(
        camera: OpenCVCamera | HarvestersCamera | MockCamera,
        image_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        log_path: Path,
        display_video: bool = False,
        fps: float | None = None,
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
    @staticmethod
    def _frame_saving_loop(
        saver: VideoSaver | ImageSaver,
        image_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        log_path: Path,
        frame_width: int | None = None,
        frame_height: int | None = None,
        video_frames_per_second: float | None = None,
        video_id: str | None = None,
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
    @staticmethod
    def _empty_function() -> None:
        """A placeholder function used to verify the class is only instantiated inside the main scope of each runtime.

        The function itself does nothing. It is used to enforce that the start() method of the class only triggers
        inside the main scope, to avoid uncontrolled spawning of daemon processes.
        """
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
    def stop_frame_saving(self) -> None:
        """Disables saving acquired camera frames.

        Does not interfere with grabbing and displaying the frames to user, this process is only stopped when the main
        stop() method is called.
        """
    def start_frame_saving(self) -> None:
        """Enables saving acquired camera frames.

        The frames are grabbed and (optionally) displayed to user after the main start() method is called, but they
        are not initially written to non-volatile memory. The call to this method additionally enables saving the
        frames to non-volatile memory.
        """
