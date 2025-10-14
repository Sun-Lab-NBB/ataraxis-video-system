"""Contains tests for classes and methods provided by the saver.py module."""

import subprocess

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system.saver import (
    VideoSaver,
    VideoEncoders,
    EncoderSpeedPresets,
    InputPixelFormats,
    OutputPixelFormats,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from ataraxis_video_system.camera import MockCamera


@pytest.fixture(scope="session")
def has_nvidia():
    """Checks for NVIDIA GPU availability in the test environment."""
    return check_gpu_availability()


@pytest.fixture(scope="session")
def has_ffmpeg():
    """Checks for FFMPEG availability in the test environment."""
    return check_ffmpeg_availability()


def test_check_gpu_availability():
    """Verifies the functioning of the check_gpu_availability() function."""
    # Tests that the function returns a boolean
    result = check_gpu_availability()
    assert isinstance(result, bool)

    # If nvidia-smi is available, verifies it returns True
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert result is True
    except:
        assert result is False


def test_check_ffmpeg_availability():
    """Verifies the functioning of the check_ffmpeg_availability() function."""
    # Tests that the function returns a boolean
    result = check_ffmpeg_availability()
    assert isinstance(result, bool)

    # If ffmpeg is available, verifies it returns True
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert result is True
    except:
        assert result is False


def test_video_saver_init_repr(tmp_path, has_ffmpeg):
    """Verifies the functioning of the VideoSaver __init__() and __repr__() methods."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    # Tests CPU encoder initialization
    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=640,
        frame_height=480,
        frame_rate=30.0,
        gpu=-1,
        video_encoder=VideoEncoders.H265,
        encoder_speed_preset=EncoderSpeedPresets.MEDIUM,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=OutputPixelFormats.YUV420,
        quantization_parameter=20,
    )

    # Verifies that the saver was initialized properly
    assert saver._system_id == 1
    assert saver._ffmpeg_process is None
    assert not saver.is_active  # Note: is_active returns True when the process is None

    # Verifies the __repr__() method
    assert "VideoSaver(" in repr(saver)
    assert "output_file=" in repr(saver)
    assert "hardware_encoding=False" in repr(saver)


@pytest.mark.parametrize(
    "video_encoder, gpu_index, output_pixel_format",
    [
        (VideoEncoders.H265, -1, OutputPixelFormats.YUV420),
        (VideoEncoders.H264, -1, OutputPixelFormats.YUV420),
        (VideoEncoders.H265, -1, OutputPixelFormats.YUV444),
        (VideoEncoders.H264, -1, OutputPixelFormats.YUV444),
    ],
)
def test_video_saver_cpu_configurations(tmp_path, video_encoder, gpu_index, output_pixel_format, has_ffmpeg):
    """Verifies different CPU encoder configurations for the VideoSaver class."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=320,
        frame_height=240,
        frame_rate=15.0,
        gpu=gpu_index,
        video_encoder=video_encoder,
        encoder_speed_preset=EncoderSpeedPresets.FASTEST,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=output_pixel_format,
        quantization_parameter=25,
    )

    # Verifies the FFMPEG command was constructed properly
    assert "libx264" in saver._ffmpeg_command or "libx265" in saver._ffmpeg_command
    # noinspection PyTypeChecker
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "veryfast" in saver._ffmpeg_command  # FASTEST maps to veryfast for CPU


@pytest.mark.parametrize(
    "video_encoder, output_pixel_format",
    [
        (VideoEncoders.H265, OutputPixelFormats.YUV420),
        (VideoEncoders.H264, OutputPixelFormats.YUV420),
        (VideoEncoders.H265, OutputPixelFormats.YUV444),
        (VideoEncoders.H264, OutputPixelFormats.YUV444),
    ],
)
def test_video_saver_gpu_configurations(tmp_path, video_encoder, output_pixel_format, has_nvidia, has_ffmpeg):
    """Verifies different GPU encoder configurations for the VideoSaver class."""
    if not has_nvidia:
        pytest.skip("Skipping this test as it requires an NVIDIA GPU.")
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=320,
        frame_height=240,
        frame_rate=15.0,
        gpu=0,
        video_encoder=video_encoder,
        encoder_speed_preset=EncoderSpeedPresets.FASTEST,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=output_pixel_format,
        quantization_parameter=25,
    )

    # Verifies the FFMPEG command was constructed properly for GPU encoding
    assert "h264_nvenc" in saver._ffmpeg_command or "hevc_nvenc" in saver._ffmpeg_command
    # noinspection PyTypeChecker
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "p1" in saver._ffmpeg_command  # FASTEST maps to p1 for GPU
    assert "-gpu 0" in saver._ffmpeg_command


def test_video_saver_start_stop(tmp_path, has_ffmpeg):
    """Verifies the functioning of the VideoSaver start() and stop() methods."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
        video_encoder=VideoEncoders.H265,
        encoder_speed_preset=EncoderSpeedPresets.FASTEST,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=OutputPixelFormats.YUV420,
        quantization_parameter=30,
    )

    # Verifies that the process is not running initially
    assert saver._ffmpeg_process is None

    # Starts the encoder process
    saver.start()
    assert saver._ffmpeg_process is not None

    # Verifies that calling start() again does nothing
    process = saver._ffmpeg_process
    saver.start()
    assert saver._ffmpeg_process is process  # Same process object

    # Stops the encoder process
    saver.stop()
    assert saver._ffmpeg_process is None

    # Verifies that calling stop() again does nothing
    saver.stop()
    assert saver._ffmpeg_process is None


def test_video_saver_save_frame(tmp_path, has_ffmpeg):
    """Verifies the functioning of the VideoSaver save_frame() method."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    # Setup
    output_file = tmp_path / "test_video.mp4"
    frame_width = 100
    frame_height = 100

    # Creates a mock camera to generate test frames
    camera = MockCamera(system_id=1, color=True, frame_rate=10, frame_width=frame_width, frame_height=frame_height)
    camera.connect()

    # Creates the video saver
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=frame_width,
        frame_height=frame_height,
        frame_rate=10.0,
        gpu=-1,
        video_encoder=VideoEncoders.H264,
        encoder_speed_preset=EncoderSpeedPresets.FASTEST,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=OutputPixelFormats.YUV420,
        quantization_parameter=35,
    )

    # Starts the encoder
    saver.start()

    # Generates and saves test frames
    for _ in range(20):
        frame = camera.grab_frame()
        saver.save_frame(frame)

    # Stops the encoder to finalize the video
    saver.stop()

    # Verifies that the video file was created
    assert output_file.exists()
    assert output_file.stat().st_size > 0  # File is not empty


def test_video_saver_save_frame_errors(tmp_path, has_ffmpeg):
    """Verifies the error handling of the VideoSaver save_frame() method."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )

    # Creates a test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Verifies that saving a frame without starting the encoder raises an error
    message = (
        f"Unable to submit the frame's data to the FFMPEG encoder process of the VideoSaver instance for the "
        f"VideoSystem with id 1 as the process has not been started. Call the start() method "
        f"to start the encoder process before calling the save_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        saver.save_frame(frame)


def test_video_saver_del(tmp_path, has_ffmpeg):
    """Verifies that the VideoSaver __del__() method properly cleans up resources."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )

    # Starts the encoder
    saver.start()
    assert saver._ffmpeg_process is not None

    # Deletes the saver (should call stop() internally)
    del saver

    # Creates a new saver to verify resources were released
    saver2 = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    # Should be able to start without conflicts
    saver2.start()
    saver2.stop()


def test_encoder_speed_preset_mappings():
    """Verifies that the encoder speed preset mappings are correctly defined."""
    # Verifies all EncoderSpeedPresets values have corresponding mappings
    for preset in EncoderSpeedPresets:
        assert preset.value in VideoSaver._gpu_encoder_preset_map
        assert preset.value in VideoSaver._cpu_encoder_preset_map

    # Verifies the specific mappings
    assert VideoSaver._gpu_encoder_preset_map[EncoderSpeedPresets.FASTEST] == "p1"
    assert VideoSaver._gpu_encoder_preset_map[EncoderSpeedPresets.SLOWEST] == "p7"
    assert VideoSaver._cpu_encoder_preset_map[EncoderSpeedPresets.FASTEST] == "veryfast"
    assert VideoSaver._cpu_encoder_preset_map[EncoderSpeedPresets.SLOWEST] == "veryslow"
