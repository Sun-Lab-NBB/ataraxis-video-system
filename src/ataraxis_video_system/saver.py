"""Provides a unified API that allows other library modules to save acquired camera frames via the FFMPEG library.

Primarily, this module abstracts the configuration and flow control steps typically involved in saving acquired camera
frames as video files in real time.
"""

from enum import IntEnum, StrEnum
from typing import Any, ClassVar
from pathlib import Path
import subprocess
from subprocess import Popen, TimeoutExpired

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console, ensure_directory_exists


def check_gpu_availability() -> bool:
    """Checks whether the host system has an Nvidia GPU.

    The presence of a GPU is determined by calling the 'nvidia-smi' command. If the command runs successfully, it
    indicates the host-system, has an Nvidia GPU.

    Returns:
        True if the host system has an Nvidia GPU, False otherwise.
    """
    try:
        # Runs nvidia-smi command, uses check to trigger CalledProcessError exception if the runtime fails
        subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False
    else:
        return True


def check_ffmpeg_availability() -> bool:
    """Checks whether the host system has the FFMPEG library installed and available on PATH.

    The presence of the FFMPEG library is determined by calling the 'ffmpeg -version' command.

    Returns:
        True if the host system has the FFMPEG library installed and available on PATH, False otherwise.
    """
    try:
        # Runs ffmpeg version command, uses check to trigger CalledProcessError exception if runtime fails
        subprocess.run(args=["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except Exception:
        return False
    else:
        return True


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
    The preset for color images.
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


class VideoSaver:
    """Interfaces with an FFMPEG process to continuously save the input camera frames as an MP4 video file.

    This class uses the FFMPEG library and either Nvidia GPU or CPU to continuously encode and append the input stream
    of camera frames to an MP4 video file stored in non-volatile memory (on disk).

    Args:
        system_id: The unique identifier code of the VideoSystem instance that uses this saver interface.
        output_file: The path to the .mp4 video file to create at runtime.
        frame_width: The width of the video to be encoded, in pixels.
        frame_height: The height of the video to be encoded, in pixels.
        frame_rate: The frame rate of the video to be created.
        gpu: The index of the GPU to use for encoding. Setting this argument to a value of -1 (default) configures the
            instance to instead use the CPU for encoding. Valid GPU indices can be obtained from the 'nvidia-smi'
            terminal command.
        video_encoder: The encoder to use for generating the video file. Must be one of the valid VideoEncoders
            enumeration members.
        encoder_speed_preset: The encoding speed preset to use for generating the video file. Must be one of the valid
            EncoderSpeedPresets enumeration members.
        input_pixel_format: The pixel format used by the input frame data. This argument depends on the configuration of
            the camera used to acquire the frames. Must be one of the valid InputPixelFormats enumeration members.
        output_pixel_format: The pixel format to be used by the output video file. Must be one of the valid
            OutputPixelFormats enumeration members.
        quantization_parameter: The integer value to use for the 'quantization parameter' of the encoder. This
            determines how much information to discard from each encoded frame. Lower values produce better video
            quality at the expense of longer processing time and larger file size: 0 is best, 51 is worst. Note, the
            default value is calibrated for the H265 encoder and is likely too low for the H264 encoder.

    Attributes:
        _gpu_encoder_preset_map: Maps EncoderSpeedPresets enumeration member values to the appropriate GPU encoder
            speed preset values.
        _cpu_encoder_preset_map: Maps EncoderSpeedPresets enumeration member values to the appropriate CPU encoder
            speed preset values.
        _system_id: Stores the unique identifier code of the VideoSystem instance that uses this saver interface.
        _ffmpeg_command: Stores the main body of the FFMPEG command used to start the video encoding process.
        _repr_body: Stores the main body of the class representation string.
        _ffmpeg_process: Stores the Popen object that controls the FFMPEG's video encoding process. This is used during
            camera frame encoding to continuously feed the input camera frames to the encoding process.
    """

    _gpu_encoder_preset_map: ClassVar[dict[int, str]] = {
        1: "p1",
        2: "p2",
        3: "p3",
        4: "p4",
        5: "p5",
        6: "p6",
        7: "p7",
    }

    _cpu_encoder_preset_map: ClassVar[dict[int, str]] = {
        1: "veryfast",
        2: "faster",
        3: "fast",
        4: "medium",
        5: "slow",
        6: "slower",
        7: "veryslow",
    }

    def __init__(
        self,
        system_id: int,
        output_file: Path,
        frame_width: int,
        frame_height: int,
        frame_rate: float,
        gpu: int = -1,
        video_encoder: VideoEncoders | str = VideoEncoders.H265,
        encoder_speed_preset: EncoderSpeedPresets | int = EncoderSpeedPresets.SLOW,
        input_pixel_format: InputPixelFormats | str = InputPixelFormats.BGR,
        output_pixel_format: OutputPixelFormats | str = OutputPixelFormats.YUV420,
        quantization_parameter: int = 15,
    ) -> None:
        # Stores the caller VideoSystem ID to a class attribute.
        self._system_id: int = system_id

        # Ensures that all enumeration inputs are stored as enumerations:
        video_encoder = VideoEncoders(video_encoder)
        encoder_speed_preset = EncoderSpeedPresets(encoder_speed_preset)
        input_pixel_format = InputPixelFormats(input_pixel_format)
        output_pixel_format = OutputPixelFormats(output_pixel_format)

        # Ensures that the output file's directory exists.
        ensure_directory_exists(output_file)

        # Constructs the encoder-specific portion of the FFMPEG command based on whether GPU or CPU encoding is
        # requested. This portion contains the encoder parameters but lacks the input header and output path.
        encoder_command_portion: str

        # If a GPU index is provided, uses one of the hardware-encoding libraries.
        if gpu >= 0:
            # Selects the encoder library.
            video_codec = "h264_nvenc" if video_encoder == VideoEncoders.H264 else "hevc_nvenc"

            # Resolves the chromatic coding profile.
            if video_codec == "h264_nvenc":
                encoder_profile = "high444p" if output_pixel_format == OutputPixelFormats.YUV444 else "main"
            else:
                encoder_profile = "rext" if output_pixel_format == OutputPixelFormats.YUV444 else "main"

            # Resolves the GPU encoding speed preset.
            encoder_speed = self._gpu_encoder_preset_map[encoder_speed_preset.value]

            # Uses the resolved data to construct the GPU encoding command.
            encoder_command_portion = (
                f"-vcodec {video_codec} -qp {quantization_parameter} -preset {encoder_speed} "
                f"-profile:v {encoder_profile} -pixel_format {output_pixel_format.value} -gpu {gpu} -rc constqp"
            )

        # Otherwise, uses one of the software-encoding libraries.
        else:
            # Selects the encoder library.
            video_codec = "libx264" if video_encoder == VideoEncoders.H264 else "libx265"

            # Resolves the chromatic coding profile.
            if video_codec == "libx265":
                encoder_profile = "main444-8" if output_pixel_format == OutputPixelFormats.YUV444 else "main"
            else:
                encoder_profile = "high444" if output_pixel_format == OutputPixelFormats.YUV444 else "high422"

            # Resolves the CPU encoding speed preset.
            encoder_speed = self._cpu_encoder_preset_map[encoder_speed_preset.value]

            # This is unique to CPU encoders. Resolves the 'parameter' specifier based on the encoder name. This is
            # used to force CPU encoders to use the QP control mode.
            parameter_specifier = "-x264-params" if video_codec == "libx264" else "-x265-params"

            # Note, the qp has to be preceded by the '-parameter' specifier for the desired h265 / h265 encoder
            encoder_command_portion = (
                f"-vcodec {video_codec} {parameter_specifier} qp={quantization_parameter} "
                f"-preset {encoder_speed} -profile {encoder_profile} -pix_fmt "
                f"{output_pixel_format.value}"
            )

        # Constructs the complete FFMPEG command used by the start() method to initialize the encoder process. This
        # includes the input specifications, encoder parameters, and output path.
        self._ffmpeg_command: str = (
            f"ffmpeg -y -f rawvideo -pix_fmt {input_pixel_format.value} -s {frame_width}x{frame_height} "
            f"-r {frame_rate} -i pipe: {encoder_command_portion} {output_file}"
        )

        # Also generates the body for the representation string used by the __repr__() method. This is done here to
        # reduce the number of class attributes.
        self._repr_body: str = (
            f"output_file={output_file}, hardware_encoding={gpu >= 0}, "
            f"input_pixel_format={input_pixel_format.value}, video_encoder={video_encoder}, "
            f"encoding_speed_preset={encoder_speed_preset.value}, quantization_parameter={quantization_parameter}, "
            f"gpu_index={gpu}"
        )

        # Initializes the attribute to store the live FFMPEG encoder process, once it is started.
        self._ffmpeg_process: Popen[bytes] | None = None

    def __repr__(self) -> str:
        """Returns the string representation of the VideoEncoder instance."""
        return f"VideoSaver({self._repr_body}, started={self._ffmpeg_process is None})"

    def __del__(self) -> None:
        """Ensures that the video encoder is stopped before the instance is garbage-collected."""
        self.stop()

    @property
    def is_active(self) -> bool:
        """Returns True if the instance's encoder process is active (running)."""
        return self._ffmpeg_process is not None

    def start(self) -> None:
        """Creates the FFMPEG encoder process and sets up the data stream to pipe incoming camera frames to the
        process.
        """
        # Prevents recreating an already existing process
        if self._ffmpeg_process is not None:
            return

        # Starts the FFMPEG process using the command constructed during initialization and saves it to class attribute.
        self._ffmpeg_process = subprocess.Popen(
            self._ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )

    def stop(self) -> None:
        """Stops the FFMPEG encoder process."""
        # Prevents stopping an already stopped process.
        if self._ffmpeg_process is None:
            return

        # If the process does not terminate 'gracefully,' it is terminated forcefully to prevent deadlocks.
        try:
            _ = self._ffmpeg_process.communicate(timeout=600)
        except TimeoutExpired:  # pragma: no cover
            self._ffmpeg_process.kill()

        # Sets the process variable to None placeholder. This causes the underlying Popen object to be garbage
        # collected.
        self._ffmpeg_process = None

    def save_frame(self, frame: NDArray[np.integer[Any]]) -> None:
        """Sends the input frame to be added to the video file managed by the instance's FFMPEG encoder process.

        Notes:
            This method expects that the input frame data matches the video dimensions and input pixel format used
            during VideoSaver initialization.

        Args:
            frame: The frame's data to be encoded into the video.

        Raises:
            ConnectionError: If the method is called before starting the encoder process via the start() method.
            BrokenPipeError: If the method encounters an error when submitting the frame's data to the FFMPEG process.
        """
        # Raises an error if the encoder process does not exist
        if self._ffmpeg_process is None:
            message = (
                f"Unable to submit the frame's data to the FFMPEG encoder process of the VideoSaver instance for the "
                f"VideoSystem with id {self._system_id} as the process has not been started. Call the start() method "
                f"to start the encoder process before calling the save_frame() method."
            )
            console.error(message=message, error=ConnectionError)

        # Writes the input frame to the encoder's standard input pipe.
        try:
            self._ffmpeg_process.stdin.write(frame.tobytes())  # type: ignore[union-attr]
        except Exception as e:  # pragma: no cover
            message = (
                f"The FFMPEG process of the VideoSaver instance for the VideoSystem with id {self._system_id} "
                f"has failed to process the input frame's data with error: {e}"
            )
            console.error(message=message, error=BrokenPipeError)
