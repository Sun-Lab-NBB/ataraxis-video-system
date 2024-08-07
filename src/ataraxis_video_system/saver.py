"""This module contains classes that expose methods for saving frames obtained from one of the supported Camera classes
as images or video files.

The classes from this module function as a unified API that allows any other module to save camera frames. The primary
intention behind this module is to abstract away the configuration and flow control steps typically involved in saving
video frames. The class leverages efficient libraries, such as FFMPEG, to maximize encoding performance and efficiency.

The classes from this module are not meant to be instantiated or used directly. Instead, they should be created using
the 'create_saver()' method from the VideoSystem class.
"""

from pathlib import Path
import cv2
from enum import Enum
from ataraxis_base_utilities import console
from ataraxis_base_utilities.console.console_class import Console
from typing import Any
import numpy as np
import subprocess
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from ataraxis_data_structures import SharedMemoryArray
from numpy.typing import NDArray


class SaverBackends(Enum):
    """Maps valid literal values used to specify Saver class backend to programmatically callable variables.

    The backend primarily determines the output of the Saver class. Generally, it is advised to use the 'video'
    backend where possible to optimize the storage space used by each file. Additionally, the 'gpu' backend is preferred
    due to optimal (highest) encoding speed with marginal impact on quality. However, the 'image' backend is also
    available if it is desirable to save frames as images.
    """

    VIDEO_GPU: str = "video_gpu"
    """
    This backend is used to instantiate a Saver class that outputs a video-file. All video savers use FFMPEG to write
    video frames or pre-acquired images as a video-file and require that FFMPEG is installed, and available on the 
    local system. Saving frames as videos is the most memory-efficient storage mechanism, but it is susceptible to 
    corruption if the process is interrupted unexpectedly. This specific backend writes video files using GPU 
    hardware-accelerated (nvenc) encoders. Select 'video_cpu' if you do not have an nvenc-compatible Nvidia GPU.
    """
    VIDEO_CPU: str = "video_cpu"
    """
    This backend is largely similar to the 'video_gpu' backend, but it writes video frames using a Software (CPU) 
    encoder. Typically, the backend enabled OpenCL hardware-acceleration to speed up encoding where possible, but it 
    will always be slower than the 'video_gpu' backend. For most scientific applications, it is preferable to use the 
    'video_gpu' backend over 'video_cpu' if possible.
    """
    IMAGE: str = "image"
    """
    This is an alternative backend to the generally preferred 'video' backends. Saver classes using this backend save 
    video frames as individual images. This method is less memory-efficient than the 'video' backend, but it is 
    generally more robust to corruption and is typically less lossy compared to the 'video' backend. Additionally, this
    backend can be configured to do the least 'processing' of frames, making it possible to achieve very high saving 
    speeds.
    """


class ImageFormats(Enum):
    """Maps valid literal values for supported image file formats to programmatically callable variables.

    The image format is an instantiation parameter that is unique to ImageSaver class. It determines the output format
    the class uses to save incoming camera frames as images.
    """
    TIFF: str = "tiff"
    """
    Generally, this is the recommended image format for most scientific uses. Tiff is a lossless format (like png) that 
    is typically more efficient to encode and work with for the purpose of visual data analysis compared to png format.
    """
    JPG: str = "jpg"
    """
    This is a lossy format that relies on DCT (Discrete Cosine Transform) compression to encode images. This method of
    compression is fast and can result in small file sizes, but this comes at the expense of losing image quality. 
    Depending on your use case and Saver class configuration, this format may be sufficient, but it is generally not 
    recommended, especially if you plan to re-code the images as a video file.
    """
    PNG: str = "png"
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
    MP4: str = "mp4"
    """
    This is the most widely supported video container format and it is the recommended format to use. All common video
    players and video data analysis tools support this format. This container supports all video codecs currently 
    available through this library.
    """
    MKV: str = "mkv"
    """
    A free and open-source format that is less well supported compared to mp4, but is the most flexible of
    all offered formats. This format is recommended for users with nuanced needs that may need to modify the code of 
    this library to implement desired features.
    """
    AVI: str = "avi"
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
    H264: str = "H264"
    """
    For CPU savers this will use libx264, for GPU savers this will use h264_nvenc. H264 is a widely used video codec 
    format that is optimized for smaller file sizes. This is an older standard and it will struggle with encoding
    very high-resolution and high-quality data. Therefore, it is generally recommended to use H265 over H264 for most
    scientific applications, if your acquisition hardware can handle the additional computation cost.
    """
    H265: str = "H265"
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
    FASTEST: str = "p1"
    """
    The best encoding speed with the lowest resultant quality of video. Generally, not recommended.
    """
    FASTER: str = "p2"
    """
    Lower encoding speed compared to FASTEST, but slightly better video quality.
    """
    FAST: str = "p3"
    """
    Fast encoding speed and low video quality.
    """
    MEDIUM: str = "p4"
    """
    Intermediate encoding speed and moderate video quality. This is the default preset.
    """
    SLOW: str = "p5"
    """
    Good video quality but slower encoding speed.
    """
    SLOWER: str = "p6"
    """
    Better video quality, but slower encoding speed compared to SLOW. This preset is recommended for all science 
    applications if sufficient computational power is available.
    """
    SLOWEST: str = "p7"
    """
    Best video quality, but even slower encoding speed than SLOWEST.
    """
    LOSSLESS: str = "lossless"
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
    ULTRAFAST: str = "ultrafast"
    """
    The best encoding speed with the lowest resultant quality of video. Generally, not recommended. Roughly maps to 
    GPU 'fastest' preset.
    """
    SUPERFAST: str = "superfast"
    """
    Lower encoding speed compared to ULTRAFAST, but slightly better video quality.
    """
    VERYFAST: str = "veryfast"
    """
    Fast encoding speed and fairly low video quality.
    """
    FASTER: str = "faster"
    """
    This is an additional level roughly between GPU 'medium' and 'fast' presets. The video quality is still low, but is 
    getting better.
    """
    FAST: str = "fast"
    """
    This is the same as the 'medium' GPU preset in terms of quality, but the encoding speed is slightly lower.
    """
    MEDIUM: str = "medium"
    """
    Intermediate encoding speed and moderate video quality. This is the default preset.
    """
    SLOW: str = "slow"
    """
    Better video quality, but slower encoding speed compared to MEDIUM. This preset is recommended for all science 
    applications if sufficient computational power is available. Roughly maps to GPU 'slower' preset.
    """
    SLOWER: str = "slower"
    """
    Best video quality, but even slower encoding speed than SLOWER. This preset is qualitatively between GPU 'slower' 
    and 'slowest' presets. 
    """
    VERYSLOW: str = "veryslow"
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
    MONOCHROME: str = "gray"
    """
    The preset for grayscale (monochrome) inputs. This is the typical output for IR cameras and many color cameras can 
    be configured to image in grayscale to conserve bandwidth.
    """
    BGR: str = "bgr24"
    """
    The preset for color inputs that do not use the alpha-channel. To be consistent with our Camera classes, we only 
    support BGR channel order for colored inputs.
    """
    BGRA: str = "bgra"
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
    YUV420: str = "yuv420"
    """
    The 'standard' video color space format that uses half-bandwidth chrominance (U/V) and full width luminance (Y).
    Generally, the resultant reduction in chromatic precision is not apparent to the viewer. However, this may be 
    undesirable for some applications and, in this case, the full-width 'yuv444' format should be used.
    """
    YUV444: str = "yuv444"
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
            value means better quality, less data loss, bigger file size, and slower processing time.
        jpeg_sampling_factor: An integer-code that specifies how JPEG encoder samples image color-space. Has to be one
            of the OpenCV 'IMWRITE_JPEG_SAMPLING_FACTOR_*' constants. It is recommended to use code 444 to preserve the
            full color-space of the image for scientific applications.
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
            queue is to function as a local buffer given that the class is intended to be sued in a multiprocessing
            context.
        _executor: A ThreadPoolExecutor for managing the image writer threads.
        _running: A flag indicating whether the worker thread is running.
        _worker_thread: A thread that continuously fetches data from the queue and passes it to worker threads.
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

        # Saves arguments to class attributes. Builds OpenCV 'parameter sequences' to optimize lower level processing
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
        # noinspection PyProtectedMember
        Console._ensure_directory_exists(output_directory)

        # Saves output directory and image format to class attributes
        self._output_directory: Path = output_directory
        self._image_format: ImageFormats = image_format

        # Initializes class multithreading control structure
        self._queue: Queue = Queue()  # Local queue to distribute frames to writer threads
        # Executor to manage write operations
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=thread_count)
        self._running: bool = True  # Tracks whether the threads are running

        # Launches the thread that manages the queue. The only job of this thread is to de-buffer the images and
        # balance them across multiple writer threads.
        self._worker_thread: Thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def __repr__(self):
        """Returns a string representation of the ImageSaver object."""
        representation_string = (
            f"ImageSaver(output_directory={self._output_directory}, image_format={self._image_format.value},"
            f"tiff_compression_strategy={self._tiff_parameters[1]}, jpeg_quality={self._jpeg_parameters[1]},"
            f"jpeg_sampling_factor={self._jpeg_parameters[3]}, png_compression_level={self._png_parameters[1]}, "
            f"thread_count={self._thread_count})"
        )

    def __del__(self) -> None:
        """Ensures the class releases all resources before being garbage-collected."""
        self.shutdown()

    def _worker(self) -> None:
        """Fetches frames to save from the queue and sends them to available writer thread(s).

        This thread manages the Queue object and ensures only one thread at a time can fetch the buffered data.
        It allows decoupling the saving process, which can have any number of worker threads, from the data-flow-control
        process.
        """
        while self._running:
            # Continuously pops the data from the queue if data is available, and sends it to saver threads.
            try:
                # Uses a low-delay polling delay strategy to both release the GIL and maximize fetching speed.
                output_path, data = self._queue.get(timeout=0.1)
                self._executor.submit(self._save_image, output_path, data)
            except Empty:
                continue

    def _save_image(self, image_id: str, data: NDArray[Any]) -> None:
        """Saves the input frame data as an image using the specified ID and class-stored output parameters.

        This method is passed to the ThreadPoolExecutor for concurrent execution, allowing for efficient saving of
        multiple images at the same time. The method is written to be as minimal as possible to optimize execution
        speed.

        Args:
            image_id: The zero-padded ID of the image to save, e.g.: '0001'. The IDs have to be unique, as images are
                saved to the same directory and are only distinguished by the ID. For other library methods to work as
                expected, the ID must be a digit-convertible string.
            data: The data of the frame to save in the form of a Numpy array. Can be monochrome or colored.
        """

        # Uses output directory, image ID and image format to construct the image output path
        output_path = Path(self._output_directory, f"{image_id}.{self._image_format.value}")

        # Tiff format
        if self._image_format.value == "tiff":
            cv2.imwrite(filename=str(output_path), img=data, params=self._tiff_parameters)

        # JPEG format
        elif self._image_format.value == "jpg":
            cv2.imwrite(filename=str(output_path), img=data, params=self._jpeg_parameters)

        # PNG format
        else:
            cv2.imwrite(filename=str(output_path), img=data, params=self._png_parameters)

    def save_image(self, image_id: str, data: NDArray[Any]) -> None:
        """Queues an image to be saved by one of the writer threads.

        This method functions as the class API entry-point. For a well-configured class to save an image, only image
        data and ID passed to this method are necessary. The class automatically handles everything else.

        Args:
            image_id: The zero-padded ID of the image to save, e.g.: '0001'. The IDs have to be unique, as images are
                saved to the same directory and are only distinguished by the ID. For other library methods to work as
                expected, the ID must be a digit-convertible string.
            data: The data of the frame to save in the form of a Numpy array. Can be monochrome or colored.

        Raises:
            ValueError: If input image_id does not conform to the expected format.
        """

        # Ensures that input IDs conform to the expected format.
        if not image_id.isdigit():
            message = (
                f"Unable to save the image with the ID {image_id} as the ID is not valid. The ID must be a "
                f"digit-convertible string, such as 0001."
            )
            console.error(error=ValueError, message=message)

        # Queues the data to be saved locally
        self._queue.put((image_id, data))

    def shutdown(self):
        """Stops the worker thread and waits for all pending tasks to complete.

        This method has to be called to properly release class resources during shutdown.
        """
        self._running = False
        self._worker_thread.join()
        self._executor.shutdown(wait=True)


class GPUVideoSaver:
    """Saves input video frames as a video file.

    This Saver class is designed to use a memory-efficient approach of saving video frames acquired with the camera as
    a video file. To do so, it uses FFMPEG library and, in the case of this specific class, Nvidia GPU hardware codec.
    Generally, this is the most storage-space and encoding-time efficient approach available through this library. The
    only downside of this approach is that if the process is interrupted unexpectedly, all acquired data may be lost.

    Notes:
        Since this method relies on Nvidia GPU hardware, it will only work on systems with an Nvidia GPU that supports
        hardware encoding. Since most modern Nvidia GPUs come with a dedicated software encoder, using this method has
        little effect on CPU performance. This makes it optimal for the context of scientific experiments, where CPU and
        GPU may be involved in running the experiment, in addition to data saving.

        The class is statically configured to operate in the constant_quantization mode. That is, every frame will be
        encoded using the same quantization, discarding the same amount of information for each frame. The lower the
        quantization parameter, the less information is discarded and the larger the file size. It is very likely that
        the default parameters of this class will need to be adjusted for your specific use-case.

    Args:
        output_directory: The path to the output directory where the video will be stored. To optimize data flow during
            runtime, the class pre-creates the saving directory ahead of time and only expects integer IDs to be passed
            as argument to video-writing commands. The videos are then saved as 'id.extension' files to the output
            directory.
        video_format: The container format to use for the output video. Use VideoFormats enumeration to specify the
            desired container format. Currently, only 'MP4', 'MKV', and 'AVI' are supported.
        video_codec: The codec (encoder) to use for generating the video file. Use VideoCodecs enumeration to specify
            the desired codec. Currently, only 'H264' and 'H265' are supported.
        preset: The encoding preset to use for generating the video file. Use GPUEncoderPresets enumeration to
            specify the preset. Note, there are two EncoderPreset enumerations, one for GPU and one for CPU. You have to
            use the GPU enumeration here!
        input_pixel_format: The pixel format used by input data. This applies to both frames and standalone images.
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
        gpu: The index of the GPU to use for encoding. Valid GPU indices can be obtained from 'nvidia-smi' command.

    Attributes:
        _output_directory: Stores the output directory.
        _video_format: Stores the desired video container format.
        _video_encoder: Stores the video codec specification.
        _encoding_preset: Stores the encoding preset of the codec.
        _output_pixel_format: Stores the pixel-format to be used by the generated video file.
        _input_pixel_format: Stores the pixel-format used by input frames / images.
        _quantization_parameter: Stores the quantization value used to control how much information is discarded from
            each macro-block of each frame.
        _gpu: Stores the index of the GPU to use for encoding.
        _encoder_profile: Stores the codec-specific profile, which is also dependent on the output pixel format.
   """

    # Lists supported image input extensions. This is used for transcoding folders of images as videos, to filter out
    # possible inputs.
    _supported_image_formats: set[str] = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}

    def __init__(
        self,
        output_directory: Path,
        video_format: VideoFormats = VideoFormats.MP4,
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: GPUEncoderPresets = GPUEncoderPresets.SLOWEST,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ):
        # Ensures that the output directory exists and saves it to class attributes
        # noinspection PyProtectedMember
        Console._ensure_directory_exists(output_directory)
        self._output_directory: Path = output_directory

        # Video container format
        self._video_format: str = video_format.value

        # Depending on the codec name, resolves the specific hardware codec.
        if video_codec == VideoCodecs.H264:
            self._video_encoder: str = "h264_nvenc"
        elif video_codec == VideoCodecs.H265:
            self._video_encoder: str = "hevc_nvenc"

        # Codec presets are identical NVENC codecs.
        self._encoding_preset: str = preset.value

        # Depending on the generic output pixel format, resolves the specific format name supported by NVENC.
        if output_pixel_format == OutputPixelFormats.YUV420:
            self._output_pixel_format: str = "yuv420p"
        else:
            self._output_pixel_format: str = "yuv444p"

        # Input pixel format
        self._input_pixel_format: str = input_pixel_format.value

        # Constant quality setting. Statically, the codec is instructed to use VBR mode to support CQ setting.
        self._quantization_parameter: int = quantization_parameter

        # The index of the GPU to use for encoding.
        self._gpu: int = gpu

        # Depending on the desired output pixel format and the selected video codec, resolves the appropriate profile
        # to support chromatic coding.
        self._encoder_profile: str
        if video_codec == "h264_nvenc" and self._output_pixel_format == "yuv444p":
            self._encoder_profile = "high444p"  # The only profile capable of 444p encoding.
        elif video_codec == "hevc_nvenc" and self._output_pixel_format == "yuv444p":
            self._encoder_profile = "rext"  # The only profile capable of 444p encoding.
        else:
            self._encoder_profile = "main"  # Since 420p is the 'default', the main profile works good here.

    def create_video_from_images(
        self, video_frames_per_second: int | float, image_directory: Path, video_id: str,
    ) -> None:
        """Converts a set of id-labeled images into a video file.

        This method can be used to convert individual images stored inside the input directory into a video file. It
        uses encoding parameters specified during class initialization and supports all image formats supported by the
        ImageSaver class.

        Notes:
            The class assumes that all images use the pixel format specified during class initialization. If this
            assumption is not met, chromatic aberrations may occur in the encoded video.

            The video is written to the output directory of the class and uses the provided video_id as name.

            The dimensions of the video are determined from the first image passed to the encoder.

        Args:
            video_frames_per_second: The framerate of the video to be created.
            image_directory: The directory where the images are saved.
            video_id: The location to save the video. Defaults to the directory that the images are saved in.

        Raises:
            Exception: If there are no images with supported file-types in the specified directory.
        """

        # First, crawls the image directory and extracts all image files (based on the file extension). Also, only keeps
        # images whose names are convertible to integers (the format used by VideoSystem). This process also sorts the
        # images based on their integer IDs.
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
                f"Unable to crate video from images. No valid image candidates discovered when crawling the image "
                f"directory ({image_directory}). Valid candidates should be images using one of the supported formats "
                f"({sorted(self._supported_image_formats)}) with digit-convertible names, e.g: 0001.jpg)"
            )
            console.error(error=RuntimeError, message=message)

        # Reads the first image using OpenCV to get image dimensions. Assumes image dimensions are consistent across
        # all images.
        frame_height, frame_width, _ = cv2.imread(filename=str(images[0])).shape

        # Generates a temporary file to serve as the image roster fed into ffmpeg
        file_list_path = f"{image_directory}/file_list.txt"  # The file is saved to the image source folder
        with open(file_list_path, "w") as file_list:
            for input_frame in images:
                # NOTE!!! It is MANDATORY to include 'file:' when the file_list.txt itself is located inside root
                # source folder and each image path is given as an absolute path. Otherwise, ffmpeg appends the root
                # path to the text file in addition to each image path, resulting in an incompatible path.
                # Also, quotation (single) marks are necessary to ensure ffmpeg correctly processes special
                # characters and spaces.
                file_list.write(f"file 'file:{input_frame}'\n")

        # Uses class attributes and input video ID to construct the output video path
        output_path = Path(self._output_directory, f"{video_id}.{self._video_format}")

        # Constructs the ffmpeg command
        ffmpeg_command = (
            f"ffmpeg -y -f concat -safe 0 -r {video_frames_per_second} -i {file_list_path} "
            f"-vcodec {self._video_encoder} -qp {self._quantization_parameter} -preset {self._encoding_preset} "
            f"-profile {self._encoder_profile} -pixel_format {self._output_pixel_format} "
            f"-gpu {self._gpu} -rc constqp {output_path}"
        )

        # Executes the ffmpeg command
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Waits for the process to complete
        stdout, stderr = ffmpeg_process.communicate()

        # Checks for any errors
        if ffmpeg_process.returncode != 0:
            error_output = stderr.decode("utf-8")
            raise RuntimeError(f"FFmpeg process failed with error: {error_output}")
        else:
            print("Video creation completed successfully.")

    # def create_video_from_queue(
    #     self,
    #     image_queue: Queue,
    #     terminator_array: SharedMemoryArray,
    #     video_frames_per_second: int | float,
    #     frame_height: int,
    #     frame_width: int,
    #     video_path: Path,
    # ) -> None:
    #     # Constructs the ffmpeg command
    #     ffmpeg_command = [
    #         "ffmpeg",
    #         "-f",
    #         "rawvideo",  # Format: video without an audio
    #         "-pixel_format",
    #         "bgr24",  # Input data pixel format, bgr24 due to how OpenCV reads images
    #         "-video_size",
    #         f"{int(frame_width)}x{int(frame_height)}",  # Video frame size
    #         "-framerate",
    #         str(video_frames_per_second),  # Video fps
    #         "-i",
    #         "pipe:",  # Input mode: Pipe
    #         "-c:v",
    #         f"{self._encoder}",  # Specifies the used encoder
    #         "-preset",
    #         f"{self._preset}",  # Preset balances encoding speed and resultant video quality
    #         "-cq",
    #         f"{self._cq}",  # Constant quality factor, determines the overall output quality
    #         "-profile:v",
    #         f"{self._profile}",  # For h264_nvenc; use "main" for hevc_nvenc
    #         "-pixel_format",
    #         f"{self._pixel_format}",  # Make sure this is compatible with your chosen codec
    #         "-rc",
    #         "vbr_hq",  # Variable bitrate, high-quality preset
    #         "-y",  # Overwrites the output file without asking
    #         video_path,
    #     ]
    #
    #     # Starts the ffmpeg process
    #     ffmpeg_process = subprocess.Popen(
    #         ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    #     )
    #
    #     try:
    #         terminator_array.connect()
    #         while terminator_array.read_data(index=2, convert_output=True):
    #             if terminator_array.read_data(index=1, convert_output=True) and not image_queue.empty():
    #                 image, _ = image_queue.get()
    #                 ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())
    #     finally:
    #         # Ensure we always close the stdin and wait for the process to finish
    #         ffmpeg_process.stdin.close()
    #         ffmpeg_process.wait()
    #         terminator_array.disconnect()
    #
    #     # Checks for any errors
    #     if ffmpeg_process.returncode != 0:
    #         error_output = ffmpeg_process.stderr.read().decode("utf-8")
    #         raise RuntimeError(f"FFmpeg process failed with error: {error_output}")


class CPUVideoSaver:
    # Lists supported codecs, and input, and output formats. Uses set for efficiency at the expense of fixed order.
    _supported_image_formats: set[str] = {"png", "tiff", "tif", "jpg", "jpeg"}
    _supported_video_formats: set[str] = {"mp4"}
    _supported_video_codecs: set[str] = {"h264_nvenc", "libx264", "hevc_nvenc", "libx265"}
    _supported_pixel_formats: set[str] = {"yuv444", "gray"}

    """Saves input video frames as a video file."""

    def __init__(self, video_codec="h264", pixel_format: str = "yuv444", crf: int = 13, cq: int = 23):
        self._codec = video_codec

    @property
    def supported_video_codecs(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported video codec options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_video_codecs))

    @property
    def supported_video_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported video format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_video_formats))

    @property
    def supported_image_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported image format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_image_formats))


def is_monochrome(image: NDArray[Any]) -> bool:
    """Verifies whether the input image is grayscale (monochrome) using HSV conversion.

    Specifically, converts input images to HSV colorspace and ensures that Saturation is zero across the entire image.
    This method is used across encoders to properly determine input image colorspace.

    Args:
        image: Image to verify in the form of a NumPy array. Note, expects the input images to use the BGR or BGRA
            color format.

    Returns:
        True if the image is grayscale, False otherwise.
    """

    if len(image.shape) < 3 or image.shape[2] == 1:
        return True
    elif 2 < image.ndim < 5:
        if image.shape[2] == 3:  # BGR image
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif image.shape[2] == 4:  # BGRA image
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Unsupported number of channels")
    else:
        raise ValueError("Unsupported number of dimensions")

    # Checks if all pixels have 0 saturation value
    return np.all(hsv_image[:, :, 1] == 0)
