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
from typing import Any, Literal
import numpy as np
import subprocess
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from ataraxis_data_structures import SharedMemoryArray
from numpy.typing import NDArray


class SaverBackends(Enum):
    """Maps valid literal values used to specify Saver class backend when requesting it from create_saver() VideoSystem
    class method to programmatically callable variables.

    Use this enumeration instead of 'hardcoding' Saver backends where possible to automatically adjust to future API
    changes to this library.

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


class ImageSaver:
    """Saves input video frames as individual images.

    This Saver class is designed to use a memory-inefficient approach of saving video frames as individual images.
    Compared to video-savers, this preserves more of the color-space and visual-data of each frame and can
    achieve very high saving speeds. That said, this method ahs the least storage-efficiency and can easily produce
    data archives in the range of TBs.

    Notes:
        An additional benefit of this method is its robustness. Due to encoding each frame as a discrete image, the data
        is constantly moved into non-volatile memory and in case of an unexpected shutdown, only a handful of frames are
        lost. For scientific applications with sufficient NVME or SSD storage space, recording data as images and then
        transcoding it as videos is likely to be the most robust and flexible approach to saving video data.

        To improve runtime efficiency, the class uses a multithreaded saving approach, where multiple images are saved
        at the same time due to GIL-releasing C-code. Generally, it is safe to use 5-10 saving threads, but that number
        depends on the specific system configuration.

    Args:
        output_directory: The path to the output directory where the images will be stored. To optimize data flow during
            runtime, the class pre-creates the saving directory ahead of time and only expects integer IDs to accompany
            the input frame data. The frames are then saved as 'id.extension' files to the pre-created directory.
        image_format: The format to use for the output image. Currently, supported formats are 'jpg', 'png', and 'tiff'.
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

    # Stores supported output image formats as a set for efficiency.
    _supported_image_formats: set[str] = {"png", "tiff", "jpg"}

    def __init__(
        self,
        output_directory: Path,
        image_format: Literal["png", "tiff", "jpg"] = "tiff",
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
        self._thread_count = thread_count

        # Ensures that the input directory exists.
        # noinspection PyProtectedMember
        Console._ensure_directory_exists(output_directory)

        # Saves output directory and image format to class attributes
        self._output_directory = output_directory
        self._image_format = image_format

        # Initializes class multithreading control structure
        self._queue = Queue()  # Local queue to distribute frames to writer threads
        self._executor = ThreadPoolExecutor(max_workers=thread_count)  # Executor to manage write operations
        self._running = True  # Tracks whether the threads are running

        # Launches the thread that manages the queue. The only job of this thread is to de-buffer the images and
        # balance them across multiple writer threads.
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def __repr__(self):
        """Returns a string representation of the ImageSaver object."""
        representation_string = (
            f"ImageSaver(output_directory={self._output_directory}, image_format={self._image_format},"
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
        output_path = Path(self._output_directory, f"{image_id}.{self._image_format}")

        # Tiff format
        if self._image_format == "tiff":
            cv2.imwrite(filename=str(output_path), img=data, params=self._tiff_parameters)

        # JPEG format
        elif self._image_format == "jpg":
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

        # Quees the data to be saved locally
        self._queue.put((image_id, data))

    @property
    def supported_image_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported output image formats."""
        return tuple(sorted(self._supported_image_formats))

    def shutdown(self):
        """Stops the worker thread and waits for all pending tasks to complete.

        This method has to be called to properly release class resources during shutdown.
        """
        self._running = False
        self._worker_thread.join()
        self._executor.shutdown(wait=True)


class GPUVideoSaver:
    # Lists supported codecs, and input, and output formats. Uses set for efficiency at the expense of fixed order.
    _supported_image_formats: set[str] = {"png", "tiff", "tif", "jpg", "jpeg"}
    _supported_video_codecs: set[str] = {"h264_nvenc", "hevc_nvenc"}
    _supported_pixel_formats: set[str] = {"yuv420p", "yuv444", "bgr0", "bgra"}
    _supported_presets: set[int] = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}
    _supported_input_pixel_formats: set[str] = {"gray", "bgr24", "bgra"}

    """Saves input video frames as a video file."""

    def __init__(
        self,
        output_directory: Path,
        video_format: Literal["mp4", "mkv", "avi"] = "mp4",
        video_encoder="h264_nvenc",
        preset="p4",
        video_pixel_format: str = "yuv420p",
        constant_quality: int = 23,
        gpu: int = 0,
    ):
        # Ensures that the output directory exists and saves it to class attributes
        # noinspection PyProtectedMember
        Console._ensure_directory_exists(output_directory)
        self._output_directory = output_directory

        self._video_format = video_format
        self._video_encoder = video_encoder
        self._encoding_preset = preset
        self._video_pixel_format = video_pixel_format
        self._constant_quality = constant_quality
        self._gpu = gpu

        if video_encoder == "h264_nvenc":
            self._encoder_profile = "high444p"
        else:
            self._encoder_profile = "rext"

    def create_video_from_images(
        self, video_frames_per_second: int | float, image_directory: Path, video_id: str,
    ) -> None:
        """Converts a set of id labeled images into an mp4 video file.

        Args:
            video_frames_per_second: The framerate of the video to be created.
            image_directory: The directory where the images are saved.
            video_id: The location to save the video. Defaults to the directory that the images are saved in.

        Raises:
            Exception: If there are no images of the specified type in the specified directory.
        """

        # First, crawls the image directory and extracts all image files (based on the file extension). Also, only keeps
        # images whose names are convertible to integers (the format used by VideoSystem).
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
                f"({self.supported_image_formats}) with digit-convertible names, e.g: 0001.jpg)"
            )
            console.error(error=RuntimeError, message=message)

        # Sorts selected image files based on their names, n ignoring suffixes. This expects that all images are named
        # using leading-zero numbers e.g.: '001', '002', etc.
        images.sort(key=lambda x: int(x.stem))

        # Reads the first image using OpenCV to get image dimensions. Assumes image dimensions are consistent across
        # all images.
        frame_height, frame_width, _ = cv2.imread(filename=str(images[0])).shape

        # Generates a temporary file to serve as the image roster fed into ffmpeg
        file_list_path = f"{image_directory}/file_list.txt"  # The file is saved locally, to the folder being encoded
        with open(file_list_path, "w") as fl:
            for input_frame in images:
                # NOTE!!! It is MANDATORY to include 'file:' when the file_list.txt itself is located inside root
                # source folder and each image path is given as an absolute path. Otherwise, ffmpeg appends the root
                # path to the text file in addition to each image path, resulting in an incompatible path.
                # Also, quotation (single) marks are necessary to ensure ffmpeg correctly processes special
                # characters and spaces.
                fl.write(f"file 'file:{input_frame}'\n")

        output_path = Path(self._output_directory, f"{video_id}.{self._video_format}")

        # Constructs the ffmpeg command
        ffmpeg_command = (
            f"ffmpeg -f concat -safe 0 -i {file_list_path} -y -pixel_format {'bgr24'} "
            f"-framerate {video_frames_per_second} -vcodec {self._video_encoder} -preset {self._encoding_preset} "
            f"-cq {self._constant_quality} -profile {self._encoder_profile} -pix_fmt {self._video_pixel_format} "
            f"-gpu {self._gpu} -rc vbr_hq -rgb_mode yuv444 -tune hq {output_path}"
        )

        # Starts the ffmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            for image in images:
                image_data = cv2.imread(str(image))
                ffmpeg_process.stdin.write(image_data.astype(np.uint8).tobytes())
        finally:
            # Ensure we always close the stdin and wait for the process to finish
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

        # Checks for any errors
        if ffmpeg_process.returncode != 0:
            error_output = ffmpeg_process.stderr.read().decode("utf-8")
            raise RuntimeError(f"FFmpeg process failed with error: {error_output}")

    def create_video_from_queue(
        self,
        image_queue: Queue,
        terminator_array: SharedMemoryArray,
        video_frames_per_second: int | float,
        frame_height: int,
        frame_width: int,
        video_path: Path,
    ) -> None:
        # Constructs the ffmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-f",
            "rawvideo",  # Format: video without an audio
            "-pixel_format",
            "bgr24",  # Input data pixel format, bgr24 due to how OpenCV reads images
            "-video_size",
            f"{int(frame_width)}x{int(frame_height)}",  # Video frame size
            "-framerate",
            str(video_frames_per_second),  # Video fps
            "-i",
            "pipe:",  # Input mode: Pipe
            "-c:v",
            f"{self._encoder}",  # Specifies the used encoder
            "-preset",
            f"{self._preset}",  # Preset balances encoding speed and resultant video quality
            "-cq",
            f"{self._cq}",  # Constant quality factor, determines the overall output quality
            "-profile:v",
            f"{self._profile}",  # For h264_nvenc; use "main" for hevc_nvenc
            "-pixel_format",
            f"{self._pixel_format}",  # Make sure this is compatible with your chosen codec
            "-rc",
            "vbr_hq",  # Variable bitrate, high-quality preset
            "-y",  # Overwrites the output file without asking
            video_path,
        ]

        # Starts the ffmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            terminator_array.connect()
            while terminator_array.read_data(index=2, convert_output=True):
                if terminator_array.read_data(index=1, convert_output=True) and not image_queue.empty():
                    image, _ = image_queue.get()
                    ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())
        finally:
            # Ensure we always close the stdin and wait for the process to finish
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            terminator_array.disconnect()

        # Checks for any errors
        if ffmpeg_process.returncode != 0:
            error_output = ffmpeg_process.stderr.read().decode("utf-8")
            raise RuntimeError(f"FFmpeg process failed with error: {error_output}")

    @property
    def supported_image_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported image format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_image_formats))

    @property
    def supported_video_codecs(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported video codec options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_video_codecs))

    @property
    def supported_pixel_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported pixel format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_pixel_formats))

    @property
    def supported_presets(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported encoding preset options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_pixel_formats))


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
