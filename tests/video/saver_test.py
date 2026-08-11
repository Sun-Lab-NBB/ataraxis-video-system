"""Contains tests for classes and methods provided by the saver.py module."""

from contextlib import suppress
import subprocess

import numpy as np
import pytest
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import error_format

from ataraxis_video_system import (
    VideoEncoders,
    InputPixelFormats,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_gpu_availability,
    check_ffmpeg_availability,
)
from ataraxis_video_system.video import saver as saver_module
from ataraxis_video_system.video.saver import VideoSaver
from ataraxis_video_system.video.camera import MockCamera


class _NoEofStdin:
    """Wraps the encoder's real stdin pipe so that closing it does not signal EOF to the FFMPEG process."""

    def __init__(self, stream) -> None:
        self.real = stream

    def close(self) -> None:
        """Ignores the close request, keeping the wrapped pipe open so that FFMPEG never exits on its own."""


class _FailingStdin:
    """Stands in for the encoder's stdin pipe and fails every write with a preset error."""

    def __init__(self, error) -> None:
        self._error = error

    def write(self, _data) -> int:
        """Raises the preset error instead of accepting the frame's data."""
        raise self._error


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


@pytest.mark.parametrize(
    "probe_error",
    [
        FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
        subprocess.CalledProcessError(returncode=9, cmd=["nvidia-smi"]),
    ],
    ids=["missing_binary", "rejecting_driver"],
)
def test_check_gpu_availability_degrades_when_the_probe_fails(monkeypatch, probe_error) -> None:
    """Verifies that check_gpu_availability() reports False instead of propagating a failing probe command."""
    probe_calls = []

    def _failing_run(*args, **kwargs):
        """Records the probe invocation and fails it the way a missing or rejecting nvidia-smi binary would."""
        probe_calls.append(kwargs)
        raise probe_error

    monkeypatch.setattr(saver_module.subprocess, "run", _failing_run)

    # The CLI, the MCP camera tools, and VideoSystem all treat this probe as a boolean gate, so a propagated error
    # would abort a runtime that is meant to fall back to CPU encoding.
    assert check_gpu_availability() is False

    # The probe interrogates the NVIDIA driver and relies on 'check' to turn a driver-level rejection into an
    # exception rather than a silent success.
    assert probe_calls[0]["args"][0] == "nvidia-smi"
    assert probe_calls[0]["check"] is True


@pytest.mark.parametrize(
    "probe_error",
    [
        FileNotFoundError(2, "No such file or directory", "ffmpeg"),
        subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"]),
    ],
    ids=["missing_binary", "failing_binary"],
)
def test_check_ffmpeg_availability_degrades_when_the_probe_fails(monkeypatch, probe_error) -> None:
    """Verifies that check_ffmpeg_availability() reports False instead of propagating a failing probe command."""
    probe_calls = []

    def _failing_run(*args, **kwargs):
        """Records the probe invocation and fails it the way a missing or broken FFMPEG installation would."""
        probe_calls.append(kwargs)
        raise probe_error

    monkeypatch.setattr(saver_module.subprocess, "run", _failing_run)

    # The has_ffmpeg test fixture, the CLI's compatibility check, the MCP camera tools, and the VideoSystem saver gate
    # all call this probe directly. A propagated error would take down every caller instead of degrading them to
    # 'FFMPEG is not installed'.
    assert check_ffmpeg_availability() is False

    assert probe_calls[0]["args"][0] == "ffmpeg"
    assert probe_calls[0]["check"] is True


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
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "veryfast" in saver._ffmpeg_command  # FASTEST maps to veryfast for CPU
    # Verifies the output is forced to full (pc) range
    range_index = saver._ffmpeg_command.index("-color_range")
    assert saver._ffmpeg_command[range_index + 1] == "pc"

    # Pins the frame geometry argument. The width and the height are interchangeable in every square-framed test, so
    # nothing else in the suite would catch the two being transposed, which FFMPEG accepts and encodes as a sheared
    # video rather than rejecting.
    size_index = saver._ffmpeg_command.index("-s")
    assert saver._ffmpeg_command[size_index + 1] == "320x240"

    # Pins the quantization parameter, which the encoder-specific parameter specifier carries for CPU encoders.
    specifier = "-x264-params" if "libx264" in saver._ffmpeg_command else "-x265-params"
    specifier_index = saver._ffmpeg_command.index(specifier)
    assert saver._ffmpeg_command[specifier_index + 1] == "qp=25"


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
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "p1" in saver._ffmpeg_command  # FASTEST maps to p1 for GPU
    gpu_index = saver._ffmpeg_command.index("-gpu")
    assert saver._ffmpeg_command[gpu_index + 1] == "0"
    # Pins the frame geometry and the quantization parameter, neither of which any other assertion would catch.
    size_index = saver._ffmpeg_command.index("-s")
    assert saver._ffmpeg_command[size_index + 1] == "320x240"
    qp_index = saver._ffmpeg_command.index("-qp")
    assert saver._ffmpeg_command[qp_index + 1] == "25"
    # Verifies the output is forced to full (pc) range
    range_index = saver._ffmpeg_command.index("-color_range")
    assert saver._ffmpeg_command[range_index + 1] == "pc"


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
    PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=200, allow_sleep=True, block=False)

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
    PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=100, allow_sleep=True, block=False)

    # Kills the FFMPEG process to simulate an unexpected termination.
    saver._ffmpeg_process.kill()
    saver._ffmpeg_process.wait()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        saver.save_frame(frame)

    # Cleans up the dead process reference to prevent stop() from failing.
    saver._ffmpeg_process = None


def test_video_saver_stop_kills_an_unresponsive_encoder(tmp_path, has_ffmpeg) -> None:
    """Verifies that stop() force-kills and reaps an FFMPEG process that outlives the shutdown grace period."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "unresponsive_test.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    saver.start()
    process = saver._ffmpeg_process

    # Without this wrapper, the stdin close inside stop() hands FFMPEG an EOF and it exits on its own, so the kill
    # would land on an already-dead process and prove nothing.
    stdin_stub = _NoEofStdin(stream=process.stdin)
    process.stdin = stdin_stub

    real_wait = process.wait
    wait_timeouts = []

    def _unresponsive_wait(timeout=None):
        """Ignores the grace period, then reaps the process that the kill escalation terminates."""
        wait_timeouts.append(timeout)
        if len(wait_timeouts) == 1:
            # An unbounded first wait would hang the whole session here, as the wrapped stdin keeps FFMPEG alive.
            if timeout is None:
                pytest.fail("VideoSaver.stop() waited on the FFMPEG process without a bounded timeout.")
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
        return real_wait(timeout=30)

    process.wait = _unresponsive_wait

    try:
        saver.stop()

        # The grace period is bounded, so a wedged encoder cannot stall VideoSystem shutdown indefinitely.
        assert wait_timeouts[0] == 600

        # The child was force-killed and then reaped, rather than left behind as a zombie process.
        assert process.poll() is not None
        assert process.returncode != 0

        assert saver._ffmpeg_process is None
        assert not saver.is_active

        # A forced shutdown leaves the instance in the same state a clean one does: stopping again is a no-op, and
        # the saver remains usable for another recording.
        saver.stop()
        assert saver._ffmpeg_process is None
        saver.start()
        assert saver.is_active
    finally:
        # The no-op close transferred the ownership of the real pipe to this test.
        with suppress(OSError):
            stdin_stub.real.close()
        saver.stop()


@pytest.mark.parametrize(
    "write_error",
    [BrokenPipeError(32, "Broken pipe"), ValueError("I/O operation on closed file")],
    ids=["severed_pipe", "closed_pipe"],
)
def test_video_saver_save_frame_reports_a_broken_encoder_pipe(tmp_path, has_ffmpeg, write_error) -> None:
    """Verifies that save_frame() translates any encoder stdin write failure into a diagnosable BrokenPipeError."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "broken_pipe_test.mp4"
    saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    saver.start()

    # The FFMPEG process stays alive, so the termination guard passes and execution reaches the pipe write. Both
    # simulated failures are what a real severed pipe produces: EPIPE when FFMPEG dies mid-write and a ValueError
    # when the pipe object is already closed.
    real_stdin = saver._ffmpeg_process.stdin
    saver._ffmpeg_process.stdin = _FailingStdin(error=write_error)

    try:
        message = (
            f"The FFMPEG process of the VideoSaver instance for the VideoSystem with id 1 has failed to process the "
            f"input frame's data with error: {write_error}"
        )
        with pytest.raises(BrokenPipeError, match=error_format(message)):
            saver.save_frame(np.zeros((100, 100, 3), dtype=np.uint8))

        # The write failure is reported to the caller, but the encoder process is deliberately left running, so the
        # consumer process can decide whether to drop the frame or shut the saver down.
        assert saver.is_active
        assert saver._ffmpeg_process.poll() is None
    finally:
        # Restoring the real pipe is load-bearing: stop() closes stdin to signal EOF, and the stub would instead
        # leave FFMPEG running until the grace period expires.
        saver._ffmpeg_process.stdin = real_stdin
        saver.stop()
