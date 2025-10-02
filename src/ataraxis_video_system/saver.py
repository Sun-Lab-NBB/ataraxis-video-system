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
from enum import IntEnum, StrEnum
from typing import Any
from pathlib import Path
import threading
import subprocess
from subprocess import Popen, TimeoutExpired

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Improves OpenCV's performance on Windows.
import cv2
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists


class VideoCodecs(StrEnum):
    """Stores the supported video codecs used when saving camera frames as videos via VideoSaver instances."""

    H264 = "H264"
    """
    For CPU savers this uses the libx264 codec and for GPU savers this uses the h264_nvenc codec. H264 is a widely used 
    video codec format that is optimized for encoding lower resolution videos at a fast speed. This is an older 
    standard that may have performance issues when working with high-resolution and high-quality data. Generally, this 
    codec is best for the cases where the overall data quality does not need to be very high, the encoding speed is a 
    critical factor, or the encoding hardware is not very powerful.
    """
    H265 = "H265"
    """
    For CPU savers this uses the libx265 and for GPU savers this uses the hevc_nvenc codec. H265 is a more modern video 
    codec format that is slightly less supported compared to H264. This codec achieves improved compression efficiency 
    without compromising quality and is better equipped to handle high-volume and high-resolution video recordings. 
    This comes at the expense of higher computational costs and slower encoding speed compared to H264 and, therefore, 
    this codec may not work on older / less powerful systems.
    """


class EncoderSpeedPreset(IntEnum):
    """Stores the supported video encoding speed presets used when saving camera frames as videos via VideoSaver
    instances.

    Generally, the faster the encoding speed, the lower is the resultant video quality.

    Notes:
        It is impossible to perfectly match the encoding presets for the CPU and GPU encoders. The scale defined
        in this enumeration represents the best-effort to align the preset scale for the two encoders.
    """

    FASTEST = 1
    """
    For CPU encoders, this matches the 'veryfast' level. For GPU encoders, this matches the 'p1' level.
    """
    FASTER = 2
    """
    For CPU encoders, this matches the 'faster' level. For GPU encoders, this matches the 'p2' level.
    """
    FAST = 3
    """
    For CPU encoders, this matches the 'fast' level. For GPU encoders, this matches the 'p3' level.
    """
    MEDIUM = 4
    """
    For CPU encoders, this matches the 'medium' level. For GPU encoders, this matches the 'p4' level.
    """
    SLOW = 5
    """
    For CPU encoders, this matches the 'slow' level. For GPU encoders, this matches the 'p5' level.
    """
    SLOWER = 6
    """
    For CPU encoders, this matches the 'slower' level. For GPU encoders, this matches the 'p6' level.
    """
    SLOWEST = 7
    """
    For CPU encoders, this matches the 'veryslow' level. For GPU encoders, this matches the 'p7' level.
    """


class InputPixelFormats(StrEnum):
    """Stores the supported camera frame data (color) formats used when saving camera frames as videos via VideoSaver
    instances.
    """

    MONOCHROME = "gray"
    """
    The preset for grayscale (monochrome) images.
    """
    BGR = "bgr24"
    """
    The preset for color images that do not use the alpha-channel.
    """
    BGRA = "bgra"
    """
    This preset is similar to the BGR preset, but also includes the alpha channel. 
    """


class OutputPixelFormats(StrEnum):
    """Stores the supported video color formats used when saving camera frames as videos via VideoSaver instances."""

    YUV420 = "yuv420p"
    """
    The 'standard' video color space format that uses half-bandwidth chrominance (U/V) and full-bandwidth luminance (Y).
    Generally, the resultant reduction in chromatic precision is not apparent to the viewer.
    """
    YUV444 = "yuv444p"
    """
    While still minorly reducing the chromatic precision, this profile uses most of the chrominance channel-width. 
    This results in minimal chromatic data loss compared to the more common 'yuv420p' format, but increases the 
    encoding processing time.
    """


# Defines the maps used to convert EncoderPresets members to the appropriate preset string for the GPU and CPU encoders.
# This allows standardizing the preset scale for both types of encoders at the library API level, improving the user
# experience.
_gpu_encoder_map: dict[int, str] = {
    1: "p1",
    2: "p2",
    3: "p3",
    4: "p4",
    5: "p5",
    6: "p6",
    7: "p7",
}

_cpu_encoder_map: dict[int, str] = {
    1: "veryfast",
    2: "faster",
    3: "fast",
    4: "medium",
    5: "slow",
    6: "slower",
    7: "veryslow",
}


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
        video_codec: VideoCodecs = VideoCodecs.H265,
        preset: EncoderSpeedPreset = EncoderSpeedPreset.SLOW,
        input_pixel_format: InputPixelFormats = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats = OutputPixelFormats.YUV444,
        quantization_parameter: int = 15,
        gpu: int = 0,
    ):
        # Ensures that the output directory exists
        ensure_directory_exists(output_directory)

        self._output_directory: Path = output_directory
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
            f"video_format=mp4, input_pixel_format={self._input_pixel_format}, "
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
        """Ensures that the live encoder is terminated when the VideoEncoder object is deleted."""
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
        uses encoding parameters specified during class initialization and supports encoding tiff, png, and jpg images.
        This method expects the frame-images to use integer-convertible IDs (e.g.: "00001.png"), as the method sorts
        images based on the ID, which determines the order they are encoded into the video.

        Notes:
            FFMPEG automatically resolves image color-space. This method does not make use of the class
            'input_pixel_format' attribute.

            The video is written to the output directory of the class and uses the provided video_id as a name.

            The dimensions of the video are determined from the first image passed to the encoder.

        Args:
            video_frames_per_second: The frame rate of the video to be created.
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
        output_path = Path(self._output_directory, f"{video_id}.mp4")

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
            video_frames_per_second: The frame rate of the video to be created.

        Raises:
            RuntimeError: If a 'live' FFMPEG encoder process already exists.
        """
        # If the FFMPEG process does not already exist, creates a new process before encoding the input frame
        if self._ffmpeg_process is None:
            # Uses class attributes and input video ID to construct the output video path
            output_path = Path(self._output_directory, f"{video_id}.mp4")

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

        This method is used to submit frames to be saved to a pre-created FFMPEG process. It expects that the
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
