"""Contains tests for classes and methods provided by the saver.py module."""

import subprocess
import time

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from ataraxis_video_system.saver import VideoSaver
from ataraxis_video_system.camera import MockCamera


def test_check_gpu_availability() -> None:
    """Verifies the functioning of the check_gpu_availability() function."""
    # Tests that the function returns a boolean.
    result = check_gpu_availability()
    assert isinstance(result, bool)

    # If nvidia-smi is available, verifies it returns True.
    try:
        subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert result
    except Exception:
        assert not result


def test_check_ffmpeg_availability() -> None:
    """Verifies the functioning of the check_ffmpeg_availability() function."""
    # Tests that the function returns a boolean.
    result = check_ffmpeg_availability()
    assert isinstance(result, bool)

    # If ffmpeg is available, verifies it returns True.
    try:
        subprocess.run(
            args=["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert result
    except Exception:
        assert not result


def test_video_saver_init_repr(tmp_path, has_ffmpeg) -> None:
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
    assert not saver.is_active

    # Verifies the __repr__() method
    assert "VideoSaver(" in repr(saver)
    assert "output_file=" in repr(saver)
    assert "hardware_encoding=False" in repr(saver)


@pytest.mark.parametrize(
    ("video_encoder", "gpu_index", "output_pixel_format"),
    [
        (VideoEncoders.H265, -1, OutputPixelFormats.YUV420),
        (VideoEncoders.H264, -1, OutputPixelFormats.YUV420),
        (VideoEncoders.H265, -1, OutputPixelFormats.YUV444),
        (VideoEncoders.H264, -1, OutputPixelFormats.YUV444),
    ],
)
def test_video_saver_cpu_configurations(tmp_path, video_encoder, gpu_index, output_pixel_format, has_ffmpeg) -> None:
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
    ("video_encoder", "output_pixel_format"),
    [
        (VideoEncoders.H265, OutputPixelFormats.YUV420),
        (VideoEncoders.H264, OutputPixelFormats.YUV420),
        (VideoEncoders.H265, OutputPixelFormats.YUV444),
        (VideoEncoders.H264, OutputPixelFormats.YUV444),
    ],
)
def test_video_saver_gpu_configurations(tmp_path, video_encoder, output_pixel_format, has_nvidia, has_ffmpeg) -> None:
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
    gpu_index = saver._ffmpeg_command.index("-gpu")
    assert saver._ffmpeg_command[gpu_index + 1] == "0"


def test_video_saver_start_stop(tmp_path, has_ffmpeg) -> None:
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


def test_video_saver_save_frame(tmp_path, has_ffmpeg) -> None:
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


def test_video_saver_save_frame_errors(tmp_path, has_ffmpeg) -> None:
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
        "Unable to submit the frame's data to the FFMPEG encoder process of the VideoSaver instance for the "
        "VideoSystem with id 1 as the process has not been started. Call the start() method "
        "to start the encoder process before calling the save_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        saver.save_frame(frame)


def test_video_saver_del(tmp_path, has_ffmpeg) -> None:
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


def test_encoder_speed_preset_mappings() -> None:
    """Verifies that the encoder speed preset properties are correctly defined."""
    # Verifies all EncoderSpeedPresets values produce valid preset strings.
    for preset in EncoderSpeedPresets:
        assert isinstance(preset.gpu_preset, str)
        assert isinstance(preset.cpu_preset, str)

    # Verifies the specific mappings.
    assert EncoderSpeedPresets.FASTEST.gpu_preset == "p1"
    assert EncoderSpeedPresets.SLOWEST.gpu_preset == "p7"
    assert EncoderSpeedPresets.FASTEST.cpu_preset == "veryfast"
    assert EncoderSpeedPresets.SLOWEST.cpu_preset == "veryslow"


def test_video_saver_context_manager(tmp_path, has_ffmpeg) -> None:
    """Verifies the VideoSaver __enter__() and __exit__() context manager methods."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "ctx_test.mp4"
    with VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    ) as saver:
        assert saver.is_active
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        saver.save_frame(frame)

    # After exiting the context, the saver should be stopped.
    assert not saver.is_active


def test_video_saver_save_non_contiguous_frame(tmp_path, has_ffmpeg) -> None:
    """Verifies that VideoSaver handles non-C-contiguous frames by calling tobytes()."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "fortran_test.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
        input_pixel_format=InputPixelFormats.BGR,
    )
    saver.start()

    # Creates a Fortran-ordered (non-C-contiguous) frame.
    frame = np.asfortranarray(np.zeros((100, 100, 3), dtype=np.uint8))
    assert not frame.flags["C_CONTIGUOUS"]
    saver.save_frame(frame)
    saver.stop()

    assert output_file.exists()


def test_video_saver_ffmpeg_error_on_stop(tmp_path, has_ffmpeg) -> None:
    """Verifies that VideoSaver logs FFMPEG error output when the process terminates with a non-zero exit code."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "error_test.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    saver.start()
    time.sleep(0.2)

    # Terminates the FFMPEG process to produce a non-zero exit code with stderr output.
    saver._ffmpeg_process.terminate()

    # stop() should handle the terminated process and trigger the error logging branch.
    saver.stop()
    assert saver._ffmpeg_process is None


def test_video_saver_save_frame_ffmpeg_crash(tmp_path, has_ffmpeg) -> None:
    """Verifies that save_frame raises RuntimeError when the FFMPEG process terminates unexpectedly."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "crash_test.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    saver.start()
    time.sleep(0.1)

    # Kills the FFMPEG process to simulate an unexpected termination.
    saver._ffmpeg_process.kill()
    saver._ffmpeg_process.wait()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        saver.save_frame(frame)

    # Cleans up the dead process reference to prevent stop() from failing.
    saver._ffmpeg_process = None
