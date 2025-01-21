from enum import Enum
from typing import Any
from pathlib import Path
from subprocess import Popen

from _typeshed import Incomplete
from numpy.typing import NDArray

class SaverBackends(Enum):
    """Maps valid literal values used to specify Saver class backend to programmatically callable variables.

    The backend primarily determines the output of the Saver class. Generally, it is advised to use the 'video'
    backend where possible to optimize the storage space used by each file. However, the 'image' backend is also
    available if it is desirable to save frames as images.
    """

    VIDEO: str
    IMAGE: str

class ImageFormats(Enum):
    """Maps valid literal values for supported image file formats to programmatically callable variables.

    The image format is an instantiation parameter that is unique to the ImageSaver class. It determines the output
    format the class uses to save incoming camera frames as images.
    """

    TIFF: str
    JPG: str
    PNG: str

class VideoFormats(Enum):
    """Maps valid literal values for supported video file formats to programmatically callable variables.

    The video format is an instantiation parameter that is unique to VideoSaver classes (GPU and CPU). It determines
    the output format the class uses to save incoming camera frames as videos.
    """

    MP4: str
    MKV: str
    AVI: str

class VideoCodecs(Enum):
    """Maps valid literal values for supported video codecs (encoders) to programmatically callable variables.

    The video codec is an instantiation parameter that is unique to VideoSaver classes (GPU and CPU). It determines the
    specific encoder used to compress and encode frames as a video file. All codecs we support come as Software (CPU)
    and Hardware (GPU) versions. The specific version of the codec (GPU or CPU) depends on the saver backend used!
    """

    H264: str
    H265: str

class GPUEncoderPresets(Enum):
    """Maps valid literal values for supported GPU codec presets to programmatically callable variables.

    Presets balance out encoding speed and resultant video quality. This acts on top of the 'constant quality'
    setting and determines how much time the codec spends optimizing individual frames. The more time the codec is
    allowed to spend on each frame, the better the resultant quality. Note, this enumeration is specifically designed
    for GPU encoders and will not work for CPU encoders.
    """

    FASTEST: str
    FASTER: str
    FAST: str
    MEDIUM: str
    SLOW: str
    SLOWER: str
    SLOWEST: str
    LOSSLESS: str

class CPUEncoderPresets(Enum):
    """Maps valid literal values for supported CPU codec presets to programmatically callable variables.

    Presets balance out encoding speed and resultant video quality. This acts on top of the 'constant rate factor'
    setting and determines how much time the codec spends optimizing individual frames. The more time the codec is
    allowed to spend on each frame, the better the resultant quality. Note, this enumeration is specifically designed
    for CPU encoders and will not work for GPU encoders.
    """

    ULTRAFAST: str
    SUPERFAST: str
    VERYFAST: str
    FASTER: str
    FAST: str
    MEDIUM: str
    SLOW: str
    SLOWER: str
    VERYSLOW: str

class InputPixelFormats(Enum):
    """Maps valid literal values for supported input pixel formats to programmatically callable variables.

    Setting the input pixel format is necessary to properly transcode the input data to video files. All our videos use
    the 'yuv' color space format, but many scientific and general cameras acquire data as images in the grayscale or
    BGR/A format. Therefore, it is necessary for the encoder to know the 'original' color space of images to properly
    convert them into the output 'yuv' color space format. This enumeration is only used by the CPU and GPU video
    Savers.
    """

    MONOCHROME: str
    BGR: str
    BGRA: str

class OutputPixelFormats(Enum):
    """Maps valid literal values for supported output pixel formats to programmatically callable variables.

    The output pixel format primarily determines how the algorithm compresses the chromatic (color) information in the
    video. This can be a good way of increasing encoding speed and decreasing video file size at the cost of reducing
    the chromatic range of the video.
    """

    YUV420: str
    YUV444: str

class ImageSaver:
    """Saves input video frames as images.

    This Saver class is designed to use a memory-inefficient approach of saving video frames as individual images.
    Compared to video-savers, this preserves more of the color-space and visual-data of each frame and can
    achieve very high saving speeds. However, this method has the least storage-efficiency and can easily produce
    data archives in the range of TBs.

    Notes:
        An additional benefit of this method is its robustness. Due to encoding each frame as a discrete image, the data
        is constantly moved into non-volatile memory. In case of an unexpected shutdown, only a handful of frames are
        lost. For scientific applications with sufficient NVME or SSD storage space, recording data as images and then
        transcoding it as videos is likely to be the most robust and flexible approach to saving video data.

        To improve runtime efficiency, the class uses a multithreaded saving approach, where multiple images are saved
        at the same time due to GIL-releasing C-code. Generally, it is safe to use 5-10 saving threads, but that number
        depends on the specific system configuration and output image format.

    Args:
        output_directory: The path to the output directory where the images will be stored. To optimize data flow during
            runtime, the class pre-creates the saving directory ahead of time and only expects integer IDs to accompany
            the input frame data. The frames are then saved as 'id.extension' files to the pre-created directory.
        image_format: The format to use for the output image. Use ImageFormats enumeration to specify the desired image
            format. Currently, only 'TIFF', 'JPG', and 'PNG' are supported.
        tiff_compression: The integer-code that specifies the compression strategy used for Tiff image files. Has to be
            one of the OpenCV 'IMWRITE_TIFF_COMPRESSION_*' constants. It is recommended to use code 1 (None) for
            lossless and fastest file saving or code 5 (LZW) for a good speed-to-compression balance.
        jpeg_quality: An integer value between 0 and 100 that controls the 'loss' of the JPEG compression. A higher
            value means better quality, less information loss, bigger file size, and slower processing time.
        jpeg_sampling_factor: An integer-code that specifies how JPEG encoder samples image color-space. Has to be one
            of the OpenCV 'IMWRITE_JPEG_SAMPLING_FACTOR_*' constants. It is recommended to use code 444 to preserve the
            full color space of the image for scientific applications.
        png_compression: An integer value between 0 and 9 that specifies the compression of the PNG file. Unlike JPEG,
            PNG files are always lossless. This value controls the trade-off between the compression ratio and the
            processing time.
        thread_count: The number of writer threads to be used by the class. Since this class uses the c-backed OpenCV
            library, it can safely process multiple frames at the same time though multithreading. This controls the
            number of simultaneously saved images the class will support.

    Attributes:
        _tiff_parameters: A tuple that contains OpenCV configuration parameters for writing .tiff files.
        _jpeg_parameters: A tuple that contains OpenCV configuration parameters for writing .jpg files.
        _png_parameters: A tuple that contains OpenCV configuration parameters for writing .png files.
        _output_directory: Stores the path to the output directory.
        _thread_count: The number of writer threads to be used by the class.
        _queue: Local queue that buffer input data until it can be submitted to saver threads. The primary job of this
            queue is to function as a local buffer given that the class is intended to be used in a multiprocessing
            context.
        _executor: A ThreadPoolExecutor for managing the image writer threads.
        _running: A flag indicating whether the worker thread is running.
        _worker_thread: A thread that continuously fetches data from the queue and passes it to worker threads.
        _frame_counter: A monotonic counter used to iteratively generate names for frame images.
    """

    _tiff_parameters: Incomplete
    _jpeg_parameters: Incomplete
    _png_parameters: Incomplete
    _thread_count: Incomplete
    _output_directory: Incomplete
    _image_format: Incomplete
    _queue: Incomplete
    _running: bool
    _executor: Incomplete
    _worker_thread: Incomplete
    _frame_counter: int
    def __init__(
        self,
        output_directory: Path,
        image_format: ImageFormats = ...,
        tiff_compression: int = ...,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = ...,
        png_compression: int = 1,
        thread_count: int = 5,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the ImageSaver object."""
    def __del__(self) -> None:
        """Ensures the class releases all resources before being garbage-collected."""
    def create_live_image_saver(self) -> None:
        """Initializes the saver by starting the saver threads.

        This method works similar to the create_live_video_encoder() method from the VideoSaver class and is responsible
        for starting the saving infrastructure. Primarily, it is used to make ImageSaver compatible with the
        VideoSystem class, as some assets used by the saver are not picklable. It has to be called to enable saving
        input frames as images via the save_frame() method.

        Notes:
            Each call to this method has to be paired with a call to the terminate_image_saver() method.
        """
    def _worker(self) -> None:
        """Fetches frames to save from the queue and sends them to available writer thread(s).

        This thread manages the Queue object and ensures only one thread at a time can fetch the buffered data.
        It allows decoupling the saving process, which can have any number of worker threads, from the data-flow-control
        process.
        """
    def _save_image(self, frame_id: str, frame: NDArray[Any]) -> None:
        """Saves the input frame data as an image using the specified ID and class-stored output parameters.

        This method is passed to the ThreadPoolExecutor for concurrent execution, allowing for efficient saving of
        multiple images at the same time. The method is written to be as minimal as possible to optimize execution
        speed.

        Args:
            frame_id: The zero-padded ID of the image to save, e.g.: '0001'. The IDs have to be unique, as images are
                saved to the same directory and are only distinguished by the ID. For other library methods to work as
                expected, the ID must be a digit-convertible string.
            frame: The data of the frame to save in the form of a Numpy array. Can be monochrome or colored.
        """
    def save_frame(self, frame: NDArray[Any]) -> None:
        """Queues the input frame to be saved by one of the writer threads.

        This method functions as the class API entry-point. For a well-configured class to save a frame as an image,
        only frame data passed to this method is necessary. The class automatically handles everything else, including
        assigning the appropriate zero-padded frame ID as the name of the output image.

        Args:
            frame: The data of the frame to save. The frame can be monochrome or colored.

        Raises:
            RuntimeError: If this method is called before starting the live image saver.
        """
    def terminate_image_saver(self) -> None:
        """Stops the live image saver and waits for all pending tasks to complete.

        This method has to be called to properly release class resources during shutdown.
        """
    @property
    def is_live(self) -> bool:
        """Returns True if the image saver has been started (is ready to save frames)."""

class VideoSaver:
    """Saves input video frames as a video file.

    This Saver class is designed to use a memory-efficient approach of saving acquired video frames as
    a video file. To do so, it uses FFMPEG library and either Nvidia hardware encoding or CPU software encoding.
    Generally, this is the most storage-space-efficient approach available through this library. The only downside of
    this approach is that if the process is interrupted unexpectedly, all acquired data may be lost.

    Notes:
        Since hardware acceleration relies on Nvidia GPU hardware, it will only work on systems with an Nvidia GPU that
        supports hardware encoding. Most modern Nvidia GPUs come with one or more dedicated software encoder, which
        frees up CPU and 'non-encoding' GPU hardware to carry out other computations. This makes it optimal in the
        context of scientific experiments, where CPU and GPU may be involved in running the experiment, in addition to
        data saving.

        This class supports both GPU and CPU encoding. If you do not have a compatible Nvidia GPU, be sure to set the
        'hardware_encoding' parameter to False.

        The class is statically configured to operate in the constant_quantization mode. That is, every frame will be
        encoded using the same quantization, discarding the same amount of information for each frame. The lower the
        quantization parameter, the less information is discarded and the larger the file size. It is very likely that
        the default parameters of this class will need to be adjusted for your specific use-case.

    Args:
        output_directory: The path to the output directory where the video file will be stored. To optimize data flow
            during runtime, the class pre-creates the saving directory ahead of time and only expects integer ID(s) to
            be passed as argument to video-writing commands. The videos are then saved as 'id.extension' files to the
            output directory.
        hardware_encoding: Determines whether to use GPU (hardware) encoding or CPU (software) encoding. It is almost
            always recommended to use the GPU encoding for considerably faster encoding with almost no quality loss.
            GPU encoding is only supported by modern Nvidia GPUs, however.
        video_format: The container format to use for the output video. Use VideoFormats enumeration to specify the
            desired container format. Currently, only 'MP4', 'MKV', and 'AVI' are supported.
        video_codec: The codec (encoder) to use for generating the video file. Use VideoCodecs enumeration to specify
            the desired codec. Currently, only 'H264' and 'H265' are supported.
        preset: The encoding preset to use for generating the video file. Use GPUEncoderPresets or CPUEncoderPresets
            enumerations to specify the preset. Note, you have to select the correct preset enumeration based on whether
            hardware encoding is enabled!
        input_pixel_format: The pixel format used by input data. This only applies when encoding simultaneously
            acquired frames. When encoding pre-acquire images, FFMPEG will resolve color formats automatically.
            Use InputPixelFormats enumeration to specify the desired pixel format. Currently, only 'MONOCHROME' and
            'BGR' and 'BGRA' options are supported. The option to choose depends on the configuration of the Camera
            class that was used for frame acquisition.
        output_pixel_format: The pixel format to be used by the output video. Use OutputPixelFormats enumeration to
            specify the desired pixel format. Currently, only 'YUV420' and 'YUV444' options are supported.
        quantization_parameter: The integer value to use for the 'quantization parameter' of the encoder.
            The encoder uses 'constant quantization' to discard the same amount of information from each macro-block of
            the frame, instead of varying the discarded information amount with the complexity of macro-blocks. This
            allows precisely controlling output video size and distortions introduced by the encoding process, as the
            changes are uniform across the whole video. Lower values mean better quality (0 is best, 51 is worst).
            Note, the default assumes H265 encoder and is likely too low for H264 encoder. H264 encoder should default
            to ~25.
        gpu: The index of the GPU to use for encoding. Valid GPU indices can be obtained from the 'nvidia-smi' command.
            This is only used when hardware_encoding is True.

    Attributes:
        _output_directory: Stores the path to the output directory.
        _video_format: Stores the desired video container format.
        _input_pixel_format: Stores the pixel format used by the input frames when 'live' saving is used. This is
            necessary to properly 'colorize' binarized image data.
        _ffmpeg_command: The 'base' ffmpeg command. Since most encoding parameters are known during class instantiation,
            the class generates the main command body with all parameters set to the desired values at instantiation.
            Subsequently, when video-creation methods are called, they pre-pend the necessary input stream information
            and append the output file information before running the command.
        _repr_body: Stores the 'base' of the class representation string. This is used to save static class parameters
            as a string that is then used by the _repr_() method to construct an accurate representation of the class
            instance.
        _supported_image_formats: Statically stores the supported image file-extensions. This is used when creating
            videos from pre-acquired images to automatically extract source images from the input directory.
        _ffmpeg_process: Stores the Popen object that controls the FFMPEG process. This is used for 'live' frame
            acquisition to instantiate the encoding process once and then 'feed' the images into the stdin pipe to be
            encoded.
    """

    _supported_image_formats: set[str]
    _output_directory: Incomplete
    _video_format: Incomplete
    _input_pixel_format: Incomplete
    _ffmpeg_command: Incomplete
    _repr_body: Incomplete
    _ffmpeg_process: Incomplete
    def __init__(
        self,
        output_directory: Path,
        hardware_encoding: bool = False,
        video_format: VideoFormats = ...,
        video_codec: VideoCodecs = ...,
        preset: GPUEncoderPresets | CPUEncoderPresets = ...,
        input_pixel_format: InputPixelFormats = ...,
        output_pixel_format: OutputPixelFormats = ...,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the VideoEncoder object."""
    def __del__(self) -> None:
        """Ensures live encoder is terminated when the VideoEncoder object is deleted."""
    @property
    def is_live(self) -> bool:
        """Returns True if the class is running an active 'live' encoder and False otherwise."""
    @staticmethod
    def _report_encoding_progress(process: Popen[bytes], video_id: str) -> None:
        """Reports FFMPEG's video encoding progress to the user via ataraxis console.

        This reads stderr output from the process used to call FFMPEG and transfers encoding progress information
        to the file log or terminal window via console class.

        Notes:
            This is only used when encoding pre-acquired images, as that process can run for a long time with no
            sign that encoding is running. Encoding 'live' frames does not need a reported process as that
            functionality is taken care of by the VideoSystem class.

        Args:
            process: The Popen object representing the ffmpeg process.
            video_id: The identifier for the video being encoded.
        """
    def create_video_from_image_folder(
        self, video_frames_per_second: float, image_directory: Path, video_id: str, *, cleanup: bool = False
    ) -> None:
        """Converts a set of existing id-labeled images stored in a folder into a video file.

        This method can be used to convert individual images stored inside the input directory into a video file. It
        uses encoding parameters specified during class initialization and supports encoding tiff, png and jpg images.
        This method expects the frame-images to use integer-convertible IDs (e.g.: "00001.png"), as the method sorts
        images based on the ID, which determines the order they are encoded into the video.

        Notes:
            FFMPEG automatically resolves image color-space. This method does not make use of the class
            \'input_pixel_format\' attribute.

            The video is written to the output directory of the class and uses the provided video_id as a name.

            The dimensions of the video are determined from the first image passed to the encoder.

        Args:
            video_frames_per_second: The framerate of the video to be created.
            image_directory: The directory where the images are saved. The method scans the directory for image files
                to be used for video creation.
            video_id: The ID or name of the generated video file. The videos will be saved as \'id.extension\' format.
            cleanup: Determines whether to clean up (delete) source images after the video creation. The cleanup is
                only carried out after the FFMPEG process terminates with a success code. Make sure to test your
                pipeline before enabling this option, as this method does not verify the encoded video for corruption.

        Raises:
            Exception: If there are no images with supported file-extensions in the specified directory.
        """
    def create_live_video_encoder(
        self, frame_width: int, frame_height: int, video_id: str, video_frames_per_second: float
    ) -> None:
        """Creates a 'live' FFMPEG encoder process, making it possible to use the save_frame() class method.

        Until the 'live' encoder is created, other class methods related to live encoding will not function. Every
        saver class can have a single 'live' encoder at a time. This number does not include any encoders initialized
        through the create_video_from_image_folder () method, but the encoders from all methods will compete for
        resources.

        This method should be called once for each 'live' recording session and paired with a call to
        terminate_live_encoder() method to properly release FFMPEG resources. If you need to encode a set of acquired
        images as a video, use the create_video_from_image_folder() method instead.

        Args:
            frame_width: The width of the video to be encoded, in pixels.
            frame_height: The height of the video to be encoded, in pixels.
            video_id: The ID or name of the generated video file. The videos will be saved as 'id.extension' format.
            video_frames_per_second: The framerate of the video to be created.

        Raises:
            RuntimeError: If a 'live' FFMPEG encoder process already exists.
        """
    def save_frame(self, frame: NDArray[Any]) -> None:
        """Sends the input frame to be encoded by the 'live' FFMPEG encoder process.

        This method is used to submit frames to be saved to a precreated FFMPEG process. It expects that the
        process has been created by the create_live_video_encoder () method. The frames must have the dimensions and
        color format specified during saver class instantiation and create_live_video_encoder() method runtime.

        Notes:
            This method should only be used to save frames that are continuously grabbed from a live camera. When
            encoding a set of pre-acquired images, it is more efficient to use the create_video_from_image_folder()
            method.

        Args:
            frame: The data of the frame to be encoded into the video by the active live encoder.

        Raises:
            RuntimeError: If 'live' encoder does not exist. Also, if the method encounters an error when submitting the
                frame to the FFMPEG process.
        """
    def terminate_live_encoder(self, timeout: float | None = None) -> None:
        """Terminates the 'live' FFMPEG encoder process if it exists.

        This method has to be called to properly release FFMPEG resources once the process is no longer necessary. Only
        call this method if you have created an encoder through the create_live_video_encoder() method.

        Args:
            timeout: The number of seconds to wait for the process to terminate or None to disable timeout. The timeout
                is used to prevent deadlocks while still allowing the process to finish encoding buffered frames before
                termination.
        """
