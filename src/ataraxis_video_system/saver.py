"""This module contains classes that expose methods for saving frames obtained from one of the supported Camera classes
as images or video files.

The classes from this module function as a unified API that allows any other module to save camera frames. The primary
intention behind this module is to abstract away the configuration and flow control steps typically involved in saving
video frames. The class leverages efficient libraries, such as FFMPEG, to maximize encoding performance and efficiency.

The classes from this module are not meant to be instantiated or used directly. Instead, they should be created using
the 'create_saver()' method from the VideoSystem class.
"""

import os
import re
from enum import Enum
from queue import Empty, Queue
from typing import Any
from pathlib import Path
import threading
from threading import Thread
import subprocess
from subprocess import Popen, TimeoutExpired
from concurrent.futures import ThreadPoolExecutor

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists


class SaverBackends(Enum):
    """Maps valid literal values used to specify Saver class backend to programmatically callable variables.

    The backend primarily determines the output of the Saver class. Generally, it is advised to use the 'video'
    backend where possible to optimize the storage space used by each file. However, the 'image' backend is also
    available if it is desirable to save frames as images.
    """

    VIDEO = "video"
    """
    This backend is used to instantiate a Saver class that outputs a video-file. All video savers use FFMPEG to write
    video frames or pre-acquired images as a video-file and require that FFMPEG is installed, and available on the 
    local system. Saving frames as videos is the most memory-efficient storage mechanism, but it is susceptible to 
    corruption if the process is interrupted unexpectedly. It is recommended to configure a 'video' saver to use the 
    GPU hardware acceleration (currently only Nvidia GPUs are supported) to optimize encoding speed.
    """
    IMAGE = "image"
    """
    This is an alternative backend to the generally preferred 'video' backend. Saver classes using this backend save 
    video frames as individual images. This method is less memory-efficient than the 'video' backend, but it is 
    generally more robust to corruption and is typically less lossy compared to the 'video' backend. Additionally, this
    backend can be configured to do the least 'processing' of frames, making it possible to achieve very high saving 
    speeds.
    """


class ImageFormats(Enum):
    """Maps valid literal values for supported image file formats to programmatically callable variables.

    The image format is an instantiation parameter that is unique to the ImageSaver class. It determines the output
    format the class uses to save incoming camera frames as images.
    """

    TIFF = "tiff"
    """
    Generally, this is the recommended image format for most scientific uses. Tiff is a lossless format (like png) that 
    is typically more efficient to encode and work with for the purpose of visual data analysis compared to png format.
    """
    JPG = "jpg"
    """
    This is a lossy format that relies on DCT (Discrete Cosine Transform) compression to encode images. This method of
    compression is fast and can result in small file sizes, but this comes at the expense of losing image quality. 
    Depending on your use case and Saver class configuration, this format may be sufficient, but it is generally not 
    recommended, especially if you plan to re-code the images as a video file.
    """
    PNG = "png"
    """
    A lossless format (like tiff) that is frequently the default in many cases. Compared to tiff, png has less features
    and may be slower to encode and decode. That said, this format is widely supported and is perfect for testing and 
    quick pipeline validation purposes.
    """


class VideoFormats(Enum):
    """Maps valid literal values for supported video file formats to programmatically callable variables.

    The video format is an instantiation parameter that is unique to VideoSaver classes (GPU and CPU). It determines
    the output format the class uses to save incoming camera frames as videos.
    """

    MP4 = "mp4"
    """
    This is the most widely supported video container format and it is the recommended format to use. All common video
    players and video data analysis tools support this format. This container supports all video codecs currently 
    available through this library.
    """
    MKV = "mkv"
    """
    A free and open-source format that is less well supported compared to mp4, but is the most flexible of
    all offered formats. This format is recommended for users with nuanced needs that may need to modify the code of 
    this library to implement desired features.
    """
    AVI = "avi"
    """
    An older format that may produce larger file sizes and does not support all available codecs. Generally, it is 
    advised not to use this format unless saved video data will be used together with a legacy system.
    """


class VideoCodecs(Enum):
    """Maps valid literal values for supported video codecs (encoders) to programmatically callable variables.

    The video codec is an instantiation parameter that is unique to VideoSaver classes (GPU and CPU). It determines the
    specific encoder used to compress and encode frames as a video file. All codecs we support come as Software (CPU)
    and Hardware (GPU) versions. The specific version of the codec (GPU or CPU) depends on the saver backend used!
    """

    H264 = "H264"
    """
    For CPU savers this will use libx264, for GPU savers this will use h264_nvenc. H264 is a widely used video codec 
    format that is optimized for smaller file sizes. This is an older standard and it will struggle with encoding
    very high-resolution and high-quality data. Therefore, it is generally recommended to use H265 over H264 for most
    scientific applications, if your acquisition hardware can handle the additional computation cost.
    """
    H265 = "H265"
    """
    For CPU savers this will use libx265, fro GPU savers this will use hevc_nvenc. H265 is the most modern video codec 
    format, which is slightly less supported compared to H264. This codec has improved compression efficiency without 
    compromising quality and is better equipped to handle high-volume and high-resolution video recordings. 
    This comes at the expense of higher computational costs compared to H264 and, therefore, this codec may not work 
    on older / less powerful systems.
    """


class GPUEncoderPresets(Enum):
    """Maps valid literal values for supported GPU codec presets to programmatically callable variables.

    Presets balance out encoding speed and resultant video quality. This acts on top of the 'constant quality'
    setting and determines how much time the codec spends optimizing individual frames. The more time the codec is
    allowed to spend on each frame, the better the resultant quality. Note, this enumeration is specifically designed
    for GPU encoders and will not work for CPU encoders.
    """

    FASTEST = "p1"
    """
    The best encoding speed with the lowest resultant quality of video. Generally, not recommended.
    """
    FASTER = "p2"
    """
    Lower encoding speed compared to FASTEST, but slightly better video quality.
    """
    FAST = "p3"
    """
    Fast encoding speed and low video quality.
    """
    MEDIUM = "p4"
    """
    Intermediate encoding speed and moderate video quality. This is the default preset.
    """
    SLOW = "p5"
    """
    Good video quality but slower encoding speed.
    """
    SLOWER = "p6"
    """
    Better video quality, but slower encoding speed compared to SLOW. This preset is recommended for all science 
    applications if sufficient computational power is available.
    """
    SLOWEST = "p7"
    """
    Best video quality, but even slower encoding speed than SLOWEST.
    """
    LOSSLESS = "lossless"
    """
    This is not part of the 'standardized' preset range. This preset is specifically optimized for acquiring lossless
    videos (not recommended!). Using this preset will result in very large file sizes and very slow encoding speeds, 
    but will produce maximum video quality with no data loss. This should not be needed outside of clinical research
    use cases.
    """


class CPUEncoderPresets(Enum):
    """Maps valid literal values for supported CPU codec presets to programmatically callable variables.

    Presets balance out encoding speed and resultant video quality. This acts on top of the 'constant rate factor'
    setting and determines how much time the codec spends optimizing individual frames. The more time the codec is
    allowed to spend on each frame, the better the resultant quality. Note, this enumeration is specifically designed
    for CPU encoders and will not work for GPU encoders.
    """

    ULTRAFAST = "ultrafast"
    """
    The best encoding speed with the lowest resultant quality of video. Generally, not recommended. Roughly maps to 
    GPU 'fastest' preset.
    """
    SUPERFAST = "superfast"
    """
    Lower encoding speed compared to ULTRAFAST, but slightly better video quality.
    """
    VERYFAST = "veryfast"
    """
    Fast encoding speed and fairly low video quality.
    """
    FASTER = "faster"
    """
    This is an additional level roughly between GPU 'medium' and 'fast' presets. The video quality is still low, but is 
    getting better.
    """
    FAST = "fast"
    """
    This is the same as the 'medium' GPU preset in terms of quality, but the encoding speed is slightly lower.
    """
    MEDIUM = "medium"
    """
    Intermediate encoding speed and moderate video quality. This is the default preset.
    """
    SLOW = "slow"
    """
    Better video quality, but slower encoding speed compared to MEDIUM. This preset is recommended for all science 
    applications if sufficient computational power is available. Roughly maps to GPU 'slower' preset.
    """
    SLOWER = "slower"
    """
    Best video quality, but even slower encoding speed than SLOWER. This preset is qualitatively between GPU 'slower' 
    and 'slowest' presets. 
    """
    VERYSLOW = "veryslow"
    """
    While not exactly lossless, this preset results in minimal video quality loss, very large file size and very slow 
    encoding speed. This is the slowest 'sane' preset that may be useful in some cases, but is generally advised 
    against.
    """


class InputPixelFormats(Enum):
    """Maps valid literal values for supported input pixel formats to programmatically callable variables.

    Setting the input pixel format is necessary to properly transcode the input data to video files. All our videos use
    the 'yuv' color space format, but many scientific and general cameras acquire data as images in the grayscale or
    BGR/A format. Therefore, it is necessary for the encoder to know the 'original' color space of images to properly
    convert them into the output 'yuv' color space format. This enumeration is only used by the CPU and GPU video
    Savers.
    """

    MONOCHROME = "gray"
    """
    The preset for grayscale (monochrome) inputs. This is the typical output for IR cameras and many color cameras can 
    be configured to image in grayscale to conserve bandwidth.
    """
    BGR = "bgr24"
    """
    The preset for color inputs that do not use the alpha-channel. To be consistent with our Camera classes, we only 
    support BGR channel order for colored inputs.
    """
    BGRA = "bgra"
    """
    This preset is similar to the BGR preset, but also includes the alpha channel. This is the only 'alternative' 
    color preset we support at this time and it is fairly uncommon to use BGRA in scientific imaging. 
    """


class OutputPixelFormats(Enum):
    """Maps valid literal values for supported output pixel formats to programmatically callable variables.

    The output pixel format primarily determines how the algorithm compresses the chromatic (color) information in the
    video. This can be a good way of increasing encoding speed and decreasing video file size at the cost of reducing
    the chromatic range of the video.
    """

    YUV420 = "yuv420p"
    """
    The 'standard' video color space format that uses half-bandwidth chrominance (U/V) and full width luminance (Y).
    Generally, the resultant reduction in chromatic precision is not apparent to the viewer. However, this may be 
    undesirable for some applications and, in this case, the full-width 'yuv444' format should be used.
    """
    YUV444 = "yuv444p"
    """
    While still doing some chroma value reduction, this profile uses most of the chrominance channel-width. This relies 
    in very little chromatic data loss and may be necessary for some scientific applications. This format is more 
    computationally expensive compared to the yuv420 format.
    """


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

    def __init__(
        self,
        output_directory: Path,
        image_format: ImageFormats = ImageFormats.TIFF,
        tiff_compression: int = cv2.IMWRITE_TIFF_COMPRESSION_LZW,
        jpeg_quality: int = 95,
        jpeg_sampling_factor: int = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        png_compression: int = 1,
        thread_count: int = 5,
    ):
        # Does not contain input-checking. Expects the initializer method of the VideoSystem class to verify all
        # input parameters before instantiating the class.

        # Saves arguments to class attributes. Builds OpenCV 'parameter sequences' to optimize lower-level processing
        # and uses tuple for efficiency.
        self._tiff_parameters: tuple[int, ...] = (int(cv2.IMWRITE_TIFF_COMPRESSION), tiff_compression)
        self._jpeg_parameters: tuple[int, ...] = (
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality,
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
            jpeg_sampling_factor,
        )
        self._png_parameters: tuple[int, ...] = (int(cv2.IMWRITE_PNG_COMPRESSION), png_compression)
        self._thread_count: int = thread_count

        # Ensures that the input directory exists.
        ensure_directory_exists(output_directory)

        # Saves output directory and image format to class attributes
        self._output_directory: Path = output_directory
        self._image_format: ImageFormats = image_format

        # Local queue to distribute frames to writer threads
        self._queue: None | Queue = None  # type: ignore
        self._running: bool = False  # Tracks whether the threads are running

        # Defines thread management attributes but does not start them.
        # Executor to manage write operations
        self._executor: None | ThreadPoolExecutor = None
        self._worker_thread: None | Thread = None

        # Initializes the input frame counter, which is used to generate frame (image) IDs.
        self._frame_counter: int = 0

    def __repr__(self) -> str:
        """Returns a string representation of the ImageSaver object."""
        representation_string = (
            f"ImageSaver(output_directory={self._output_directory}, image_format={self._image_format.value},"
            f"tiff_compression_strategy={self._tiff_parameters[1]}, jpeg_quality={self._jpeg_parameters[1]},"
            f"jpeg_sampling_factor={self._jpeg_parameters[3]}, png_compression_level={self._png_parameters[1]}, "
            f"thread_count={self._thread_count})"
        )
        return representation_string

    def __del__(self) -> None:
        """Ensures the class releases all resources before being garbage-collected."""
        self.terminate_image_saver()

    def create_live_image_saver(self) -> None:
        """Initializes the saver by starting the saver threads.

        This method works similar to the create_live_video_encoder() method from the VideoSaver class and is responsible
        for starting the saving infrastructure. Primarily, it is used to make ImageSaver compatible with the
        VideoSystem class, as some assets used by the saver are not picklable. It has to be called to enable saving
        input frames as images via the save_frame() method.

        Notes:
            Each call to this method has to be paired with a call to the terminate_image_saver() method.
        """
        if not self._running:
            self._running = True
            self._queue = Queue()
            self._executor = ThreadPoolExecutor(max_workers=self._thread_count)
            self._worker_thread = Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _worker(self) -> None:
        """Fetches frames to save from the queue and sends them to available writer thread(s).

        This thread manages the Queue object and ensures only one thread at a time can fetch the buffered data.
        It allows decoupling the saving process, which can have any number of worker threads, from the data-flow-control
        process.
        """
        while self._running:
            # Continuously pops the data from the queue if data is available and sends it to saver threads.
            try:
                # Uses a low-delay polling delay strategy to both release the GIL and maximize fetching speed.
                frame_id, frame_data = self._queue.get(timeout=0.1)  # type: ignore
                self._executor.submit(self._save_image, frame_id, frame_data)  # type: ignore
            except Empty:
                continue

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
        # Uses output directory, image ID, and image format to construct the image output path
        output_path = Path(self._output_directory, f"{frame_id}.{self._image_format.value}")

        # Tiff format
        if self._image_format.value == "tiff":
            cv2.imwrite(filename=str(output_path), img=frame, params=self._tiff_parameters)

        # JPEG format
        elif self._image_format.value == "jpg":
            cv2.imwrite(filename=str(output_path), img=frame, params=self._jpeg_parameters)

        # PNG format
        else:
            cv2.imwrite(filename=str(output_path), img=frame, params=self._png_parameters)

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
        if not self._running:
            message = (
                "Unable to submit the frame to the 'live' image saver as teh process does not exist. Call "
                "create_live_image_saver() method to create a 'live' saver before calling save_frame() method."
            )
            console.error(message=message, error=RuntimeError)

        self._frame_counter += 1  # Increments the frame counter for the next image ID
        frame_id = str(self._frame_counter).zfill(20)  # Generates a zero-padded frame ID based on the processed count
        # Queues the data to be saved locally
        self._queue.put((frame_id, frame))  # type: ignore

    def terminate_image_saver(self) -> None:
        """Stops the live image saver and waits for all pending tasks to complete.

        This method has to be called to properly release class resources during shutdown.
        """
        if self._running:
            self._running = False
            if self._worker_thread is not None:
                self._worker_thread.join()

            if self._executor is not None:
                self._executor.shutdown(wait=True)

    @property
    def is_live(self) -> bool:
        """Returns True if the image saver has been started (is ready to save frames)."""
        return self._running


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

    # Lists supported image input extensions. This is used for transcoding folders of images as videos to filter out
    # possible inputs.
    _supported_image_formats: set[str] = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}

    def __init__(
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
    ):
        # Ensures that the output directory exists
        ensure_directory_exists(output_directory)

        self._output_directory: Path = output_directory
        self._video_format: str = str(video_format.value)
        self._input_pixel_format: str = str(input_pixel_format.value)

        # Depending on the requested codec type and hardware_acceleration preference, selects the specific codec to
        # use for video encoding.
        video_encoder: str
        if video_codec == VideoCodecs.H264 and hardware_encoding:
            video_encoder = "h264_nvenc"
        elif video_codec == VideoCodecs.H265 and hardware_encoding:
            video_encoder = "hevc_nvenc"
        elif video_codec == VideoCodecs.H264 and not hardware_encoding:
            video_encoder = "libx264"
        else:
            video_encoder = "libx265"

        # Depending on the desired output pixel format and the selected video codec, resolves the appropriate profile
        # to support chromatic coding.
        encoder_profile: str
        if video_encoder == "h264_nvenc":
            if output_pixel_format.value == "yuv444p":
                encoder_profile = "high444p"  # The only profile capable of 444p encoding.
            else:
                encoder_profile = "main"  # 420p falls here
        elif video_encoder == "hevc_nvenc":
            if output_pixel_format.value == "yuv444p":
                encoder_profile = "rext"  # The only profile capable of 444p encoding.
            else:
                encoder_profile = "main"  # Same as above, 420p works with the main profile
        elif video_encoder == "libx265":
            if output_pixel_format.value == "yuv444p":
                encoder_profile = "main444-8"  # 444p requires this profile
            else:
                encoder_profile = "main"  # 420p requires at least this profile
        elif output_pixel_format.value == "yuv444p":
            encoder_profile = "high444"  # 444p requires this profile
        else:
            encoder_profile = "high420"  # 420p requires at least this profile

        # This is unique to CPU codecs. Resolves the 'parameter' specifier based on the codec name. This is used to
        # force CPU encoders to use the QP control mode.
        parameter_specifier: str
        if video_encoder == "libx264":
            parameter_specifier = "-x264-params"
        else:
            parameter_specifier = "-x265-params"

        # Constructs the main body of the ffmpeg command that will be used to generate video file(s). This block
        # lacks the input header and the output file path, which is added by other methods of this class when they
        # are called.
        self._ffmpeg_command: str
        if hardware_encoding:
            self._ffmpeg_command = (
                f"-vcodec {video_encoder} -qp {quantization_parameter} -preset {preset.value} "
                f"-profile:v {encoder_profile} -pixel_format {output_pixel_format.value} -gpu {gpu} -rc constqp"
            )
        else:
            # Note, the qp has to be preceded by the '-parameter' specifier for the desired h265 / h265 codec
            self._ffmpeg_command = (
                f"-vcodec {video_encoder} {parameter_specifier} qp={quantization_parameter} -preset {preset.value} "
                f"-profile {encoder_profile} -pixel_format {output_pixel_format.value}"
            )

        # Also generates the body for the representation string to be used by the repr method. This is done here to
        # reduce the number of class attributes.
        self._repr_body: str = (
            f"output_directory={self._output_directory}, hardware_encoding{hardware_encoding}, "
            f"video_format={self._video_format}, input_pixel_format={self._input_pixel_format}, "
            f"video_codec={video_encoder}, encoder_preset={preset.value}, "
            f"quantization_parameter={quantization_parameter}, gpu_index={gpu}"
        )

        # Stores the FFMPEG process for 'live' frame saving. Initialized to a None placeholder value
        self._ffmpeg_process: Popen[bytes] | None = None

    def __repr__(self) -> str:
        """Returns a string representation of the VideoEncoder object."""
        if self._ffmpeg_process is None:
            live_encoder = False
        else:
            live_encoder = True

        return f"VideoSaver({self._repr_body}, live_encoder={live_encoder})"

    def __del__(self) -> None:
        """Ensures live encoder is terminated when the VideoEncoder object is deleted."""
        if self._ffmpeg_process is not None:
            self.terminate_live_encoder(timeout=600)

    @property
    def is_live(self) -> bool:
        """Returns True if the class is running an active 'live' encoder and False otherwise."""
        if self._ffmpeg_process is None:
            return False
        return True

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
        # Initial message to notify the user that encoding is in progress
        console.echo(message=f"Started encoding video: {video_id}", level=LogLevel.INFO)

        # Specifies the regular expression pattern used by FFMPEG to report encoding progress. This is used to
        # extract this information from the stderr output.
        pattern = re.compile(r"time=(\d{2}:\d{2}:\d{2}.\d{2})")

        # Loops until the FFMPEG is done encoding the requested video:
        while process.poll() is None:
            # If new lines are available from stderr, reads each line and determines whether it contains progress
            # information
            if process.stderr:
                stderr_line = process.stderr.readline().decode("utf-8").strip()
                match = pattern.search(stderr_line)

                # If progress information is found, passes it to the console for handling
                if match:
                    progress_time = match.group(1)
                    console.echo(f"Video {video_id} encoding progress: {progress_time}", level=LogLevel.INFO)

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
            'input_pixel_format' attribute.

            The video is written to the output directory of the class and uses the provided video_id as a name.

            The dimensions of the video are determined from the first image passed to the encoder.

        Args:
            video_frames_per_second: The framerate of the video to be created.
            image_directory: The directory where the images are saved. The method scans the directory for image files
                to be used for video creation.
            video_id: The ID or name of the generated video file. The videos will be saved as 'id.extension' format.
            cleanup: Determines whether to clean up (delete) source images after the video creation. The cleanup is
                only carried out after the FFMPEG process terminates with a success code. Make sure to test your
                pipeline before enabling this option, as this method does not verify the encoded video for corruption.

        Raises:
            Exception: If there are no images with supported file-extensions in the specified directory.
        """
        # First, crawls the image directory and extracts all image files (based on the file extension). Also, only keeps
        # images whose names are convertible to integers (the format used by VideoSystem class). This process also
        # sorts the images based on their integer IDs (this is why they have to be integers).
        images = sorted(
            [
                img
                for img in image_directory.iterdir()
                if img.is_file() and img.suffix.lower() in self._supported_image_formats and img.stem.isdigit()
            ],
            key=lambda x: int(x.stem),
        )

        # If the process above did not discover any images, raises an error:
        if len(images) == 0:
            message = (
                f"Unable to create video {video_id} from images. No valid image candidates discovered when crawling "
                f"the image directory ({image_directory}). Valid candidates are images using one of the supported "
                f"file-extensions ({sorted(self._supported_image_formats)}) with "
                f"digit-convertible names (e.g: 0001.jpg)."
            )
            console.error(error=RuntimeError, message=message)

        # Reads the first image using OpenCV to get image dimensions. Assumes image dimensions are consistent across
        # all images.
        frame_height, frame_width, _ = cv2.imread(filename=str(images[0])).shape

        # Generates a temporary file to serve as the image roster fed into ffmpeg. The list is saved to the image
        # source folder.
        file_list_path: Path = image_directory.joinpath("source_images.txt")
        with open(file_list_path, "w") as file_list:
            for input_frame in images:
                # NOTE!!! It is MANDATORY to include 'file:' when the file_list.txt itself is located inside the root
                # source folder and each image path is given as an absolute path. Otherwise, ffmpeg appends the root
                # path to the text file in addition to each image path, resulting in an incompatible path.
                # Also, quotation (single) marks are necessary to ensure ffmpeg correctly processes special
                # characters and spaces.
                file_list.write(f"file 'file:{input_frame}'\n")

        # Uses class attributes and input video ID to construct the output video path
        output_path = Path(self._output_directory, f"{video_id}.{self._video_format}")

        # Constructs the ffmpeg command, using the 'base' command created during instantiation for video parameters
        ffmpeg_command = (
            f"ffmpeg -y -f concat -safe 0 -r {video_frames_per_second} -i {file_list_path} {self._ffmpeg_command} "
            f"{output_path}"
        )

        # Starts the ffmpeg process
        ffmpeg_process: Popen[bytes] = subprocess.Popen(
            ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Instantiates and starts a thread that monitors stderr pipe of the FFMPEG process and reports progress
        # information to the user
        progress_thread = threading.Thread(target=self._report_encoding_progress, args=(ffmpeg_process, video_id))
        progress_thread.start()

        # Waits for the encoding process to complete
        stdout, stderr = ffmpeg_process.communicate()

        # Waits for the progress reporting thread to terminate
        progress_thread.join()

        # Removes the temporary image source file after encoding is complete
        file_list_path.unlink(missing_ok=True)

        # Checks for encoding errors. If there were no errors, reports successful encoding to the user
        if ffmpeg_process.returncode != 0:  # pragma: no cover
            error_output = stderr.decode("utf-8")
            message = f"FFmpeg process failed to encode video {video_id} with error: {error_output}"
            console.error(error=RuntimeError, message=message)
        else:
            console.echo(f"Successfully encoded video {video_id}.", level=LogLevel.SUCCESS)

            # If cleanup is enabled, deletes all source images used to encode the video
            if cleanup:
                for image in images:
                    image.unlink(missing_ok=True)

                console.echo(f"Removed source images used to encode video {video_id}.", level=LogLevel.SUCCESS)

    def create_live_video_encoder(
        self,
        frame_width: int,
        frame_height: int,
        video_id: str,
        video_frames_per_second: float,
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
        # If the FFMPEG process does not already exist, creates a new process before encoding the input frame
        if self._ffmpeg_process is None:
            # Uses class attributes and input video ID to construct the output video path
            output_path = Path(self._output_directory, f"{video_id}.{self._video_format}")

            # Constructs the ffmpeg command, using the 'base' command created during instantiation for video parameters
            ffmpeg_command = (
                f"ffmpeg -y -f rawvideo -pix_fmt {self._input_pixel_format} -s {frame_width}x{frame_height} "
                f"-r {video_frames_per_second} -i pipe: {self._ffmpeg_command} {output_path}"
            )

            # Starts the ffmpeg process and saves it to class attribute
            self._ffmpeg_process = subprocess.Popen(
                ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )

        # Only allows one 'live' encoder at a time
        else:
            message = (
                f"Unable to create live video encoder for video {video_id}. FFMPEG process already exists and a "
                f"video saver class can have at most one 'live' encoder at a time. Call the terminate_live_encoder() "
                f"method to terminate the existing encoder before creating a new one."
            )
            console.error(message=message, error=RuntimeError)

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
        # Raises an error if the 'live' encoder does not exist
        if self._ffmpeg_process is None:
            message = (
                "Unable to submit the frame to a 'live' FFMPEG encoder process as the process does not exist. Call "
                "create_live_video_encoder() method to create a 'live' encoder before calling save_frame() method."
            )
            console.error(message=message, error=RuntimeError)

        # Writes the input frame to the ffmpeg process's standard input pipe.
        try:
            self._ffmpeg_process.stdin.write(frame.tobytes())  # type: ignore
        except Exception as e:  # pragma: no cover
            message = f"FFMPEG process failed to process the input frame with error: {e}"
            console.error(message=message, error=RuntimeError)

    def terminate_live_encoder(self, timeout: float | None = None) -> None:
        """Terminates the 'live' FFMPEG encoder process if it exists.

        This method has to be called to properly release FFMPEG resources once the process is no longer necessary. Only
        call this method if you have created an encoder through the create_live_video_encoder() method.

        Args:
            timeout: The number of seconds to wait for the process to terminate or None to disable timeout. The timeout
                is used to prevent deadlocks while still allowing the process to finish encoding buffered frames before
                termination.
        """
        # If the process does not exist, returns immediately
        if self._ffmpeg_process is None:
            return

        # Specified termination timeout. If the process does not terminate 'gracefully,' it is terminated
        # forcefully to prevent deadlocks.
        try:
            _ = self._ffmpeg_process.communicate(timeout=timeout)
        except TimeoutExpired:  # pragma: no cover
            self._ffmpeg_process.kill()

        # Sets the process variable to None placeholder. This causes the underlying Popen object to be garbage
        # collected.
        self._ffmpeg_process = None
