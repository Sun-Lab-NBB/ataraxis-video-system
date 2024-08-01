"""This module contains classes for saving frames obtained from one of the supported Camera classes as images or
video files.

The classes from this module function as a unified API that allows any other module to save camera frames.

The classes from this module are not meant to be instantiated or used directly. Instead, they should be created using
the 'create_saver()' method from the VideoSystem class.
"""

import tempfile
from pathlib import Path
import cv2
from enum import Enum
from ataraxis_base_utilities import console
import numpy as np
import subprocess
from queue import Queue
from ataraxis_data_structures import SharedMemoryArray


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
    'video_gpu' backend over 'video_cpu' if possible.'
    """
    IMAGE: str = "image"
    """
    This is an alternative backend to the generally preferred 'video' backend. Saver classes using this backend save 
    video frames as individual images. This method is less memory-efficient than the 'video' backend, but it is 
    generally more robust to corruption and is typically less lossy compared to the 'video' backend.
    """


class ImageSaver:
    _supported_image_formats: set[str] = {"png", "tiff", "tif", "jpg", "jpeg"}

    """Saves input video frames as images."""

    def __init__(self):
        pass

    @staticmethod
    def write_image(output_path: Path, data: NDArray[Any], tiff_compression_level: int = 6, jpeg_quality: int = 95) -> None:
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
        save_format = output_path.suffix
        if save_format in {"tiff", "tif"}:
            if data.ndim :  # If input data is RGB
            img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            tff.imwrite(
                filename, img_rgb, compression="zlib", compressionargs={"level": tiff_compression_level}
            )  # 0 to 9 default is 6
        elif save_format in {"jpg", "jpeg"}:
            cv2.imwrite(filename, data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])  # 0 to 100 default is 95
        else:  # save_format == "png"
            cv2.imwrite(filename, data)


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


class GPUVideoSaver:
    # Lists supported codecs, and input, and output formats. Uses set for efficiency at the expense of fixed order.
    _supported_image_formats: set[str] = {"png", "tiff", "tif", "jpg", "jpeg"}
    _supported_video_formats: set[str] = {"mp4"}
    _supported_video_codecs: set[str] = {"h264_nvenc", "hevc_nvenc"}
    _supported_pixel_formats: set[str] = {"yuv420p", "yuv444", "yuv444p16le", "bgr0", "bgra"}
    _supported_presets: set[int] = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}

    """Saves input video frames as a video file."""

    def __init__(
        self,
        video_encoder="h264_nvenc",
        preset="p4",
        pixel_format: str = "yuv444",
        constant_quality: int = 23,
        gpu: int = 0,
    ):
        self._encoder = video_encoder
        if self._encoder == "h264_nvenc":
            self._profile = "high444p"  # More or less required for everything other than 'yuv420p'.
        else:
            self._profile = "rext"
        self._preset = preset
        self._pixel_format = pixel_format
        self._cq = constant_quality
        self._gpu = gpu

    @property
    def supported_image_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported image format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_image_formats))

    @property
    def supported_video_formats(self) -> tuple[str, ...]:
        """Returns a tuple that stores supported video format options."""

        # Sorts to address the issue of 'set' not having a reproducible order.
        return tuple(sorted(self._supported_video_formats))

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

    def create_video_from_images(
        self, video_frames_per_second: int | float, image_directory: Path, video_path: Path
    ) -> None:
        """Converts a set of id labeled images into an mp4 video file.

        Args:
            video_frames_per_second: The framerate of the video to be created.
            image_directory: The directory where the images are saved.
            video_path: The location to save the video. Defaults to the directory that the images are saved in.

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
        images.sort(key=lambda image: int(image.stem))

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

        # Constructs the ffmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-f",
            "concat",  # Format: video without an audio
            "-safe",
            "0"  # Forces to accept any filename as input
            "-pixel_format",
            "bgr24",  # Input data pixel format, bgr24 due to how OpenCV reads images
            "-video_size",
            f"{int(frame_width)}x{int(frame_height)}",  # Video frame size
            "-framerate",
            str(video_frames_per_second),  # Video fps
            "-i",
            f"{image_directory}/%d.{images[0].suffix[1:]}",  # Input mode: file pattern
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
