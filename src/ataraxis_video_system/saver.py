"""This module contains classes that expose methods for saving frames obtained from one of the supported Camera classes
as images or video files.

The classes from this module function as a unified API that allows any other module to save camera frames. The primary
intention behind this module is to abstract away the configuration and flow control steps typically involved in saving
video frames. The class leverages efficient libraries, such as FFMPEG, to maximize encoding performance and efficiency.

The classes from this module are not meant to be instantiated or used directly. Instead, they should be created using
the 'create_saver()' method from the VideoSystem class.
"""

from enum import IntEnum, StrEnum
from typing import Any
from pathlib import Path
import subprocess
from subprocess import Popen, TimeoutExpired

from numpy.typing import NDArray
from ataraxis_base_utilities import console, ensure_directory_exists


class VideoEncoders(StrEnum):
    """Stores the supported video encoders used when saving camera frames as videos via VideoSaver instances."""

    H264 = "H264"
    """
    For CPU savers this is the libx264 encoder and for GPU savers this is the h264_nvenc encoder.
    """
    H265 = "H265"
    """
    For CPU savers this is the libx265 encoder and for GPU savers this is the hevc_nvenc encoder.
    """


class EncoderSpeedPresets(IntEnum):
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
    """Interfaces with an FFMPEG process to continuously save the input camera frames as an MP4 video file.

    This class uses the FFMPEG library and either Nvidia GPU or CPU to continuously encode and append the input stream
    of camera frames to an MP4 video file stored in non-volatile memory (on disk).

    Notes:
        Every processed frame is encoded using the same quantization, discarding the same amount of information for
        each frame. The lower the quantization parameter, the less information is discarded and the larger the file
        size.

    Args:
        output_directory: The path to the output directory where to store the generated video file.
        gpu: The index of the GPU to use for encoding. Setting this argument to a value of -1 (default) configures the
            instance to instead use the CPU for encoding. Valid GPU indices can be obtained from the 'nvidia-smi'
            terminal command.
        video_encoder: The encoder to use for generating the video file. Must be one of the valid VideoEncoders
            enumeration members.
        encoder_speed_preset: The encoding speed preset to use for generating the video file. Must be one of the valid
            EncoderSpeedPresets enumeration members.
        input_pixel_format: The pixel format used by input frame data. This argument depends on the configuration of
            the camera used to acquire the frames. Must be one of the valid InputPixelFormats enumeration members.
        output_pixel_format: The pixel format to be used by the output video file. Must be one of the valid
            OutputPixelFormats enumeration members.
        quantization_parameter: The integer value to use for the 'quantization parameter' of the encoder. This
            determines how much information to discard from each encoded frame. Lower values mean better video quality:
            0 is best, 51 is worst. Note, the default value is calibrated for the H265 encoder and is likely too low for
            the H264 encoder.

    Attributes:
        _output_directory: Stores the path to the output directory.
        _input_pixel_format: Stores the pixel format used by the input camera frames.
        _ffmpeg_command: Stores the main body of the FFMPEG command used to start the video encoding process.
        _repr_body: Stores the main body of the class representation string.
        _ffmpeg_process: Stores the Popen object that controls the FFMPEG's video encoding process. This is used during
            camera frame encoding to continuously feed the input camera frames to the encoding process.
    """

    def __init__(
        self,
        output_directory: Path,
        gpu: int = -1,
        video_encoder: VideoEncoders | str = VideoEncoders.H265,
        encoder_speed_preset: EncoderSpeedPresets | int = EncoderSpeedPresets.SLOW,
        input_pixel_format: InputPixelFormats | str = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats | str = OutputPixelFormats.YUV420,
        quantization_parameter: int = 15,
    ) -> None:
        # Ensures that the output directory exists
        ensure_directory_exists(output_directory)

        self._output_directory: Path = output_directory
        self._input_pixel_format: str = input_pixel_format.value()

        # Constructs the main body of the ffmpeg command that will be used to generate video file(s). This block
        # lacks the input header and the output file path, which is added by other methods of this class when they
        # are called.
        self._ffmpeg_command: str


        if gpu >= 0:
            # Depending on the requested encoder type and gpu acceleration, selects the specific encoder library to use
            # for video encoding.
            video_encoder = "h264_nvenc" if video_encoder == VideoEncoders.H264 else "hevc_nvenc"

            # Depending on the desired output pixel format and the selected video codec, resolves the appropriate profile
            # to support chromatic coding.
            if video_encoder == "h264_nvenc":
                encoder_profile = "high444p" if output_pixel_format.value == "yuv444p" else "main"
            else:
                encoder_profile = "rext" if output_pixel_format.value == "yuv444p" else "main"

            self._ffmpeg_command = (
                f"-vcodec {video_encoder} -qp {quantization_parameter} -preset {encoder_speed_preset.value} "
                f"-profile:v {encoder_profile} -pixel_format {output_pixel_format.value} -gpu {gpu} -rc constqp"
            )

        else:
            video_encoder = "libx264" if video_encoder == VideoEncoders.H264 else "libx265"

            if video_encoder == "libx265":
                encoder_profile = "main444-8" if output_pixel_format.value == "yuv444p" else "main"
            else:
                encoder_profile = "high444" if output_pixel_format.value == "yuv444p" else "high420"

            # This is unique to CPU codecs. Resolves the 'parameter' specifier based on the codec name. This is used to
            # force CPU encoders to use the QP control mode.
            parameter_specifier = "-x264-params" if video_encoder == "libx264" else "-x265-params"

            # Note, the qp has to be preceded by the '-parameter' specifier for the desired h265 / h265 codec
            self._ffmpeg_command = (
                f"-vcodec {video_encoder} {parameter_specifier} qp={quantization_parameter} "
                f"-preset {encoder_speed_preset.value} -profile {encoder_profile} -pixel_format "
                f"{output_pixel_format.value}"
            )

        # Also generates the body for the representation string to be used by the repr method. This is done here to
        # reduce the number of class attributes.
        self._repr_body: str = (
            f"output_directory={self._output_directory}, hardware_encoding={gpu >= 0}, "
            f"video_format=mp4, input_pixel_format={self._input_pixel_format}, "
            f"video_encoder={video_encoder}, encoding_speed_preset={encoder_speed_preset.value}, "
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
            self.terminate_encoder(timeout=600)

    @property
    def is_live(self) -> bool:
        """Returns True if the class is running an active 'live' encoder and False otherwise."""
        if self._ffmpeg_process is None:
            return False
        return True

    def create_encoder(
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

    def terminate_encoder(self, timeout: float | None = None) -> None:
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
