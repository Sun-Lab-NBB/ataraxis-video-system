"""Contains tests for classes and methods provided by the saver.py module."""

from contextlib import suppress
import subprocess

import numpy as np
import pytest
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console, error_format

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


def test_check_gpu_availability_matches_the_host_nvidia_probe() -> None:
    """Verifies that check_gpu_availability() agrees with a direct nvidia-smi probe of the host."""
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


def test_check_ffmpeg_availability_matches_the_host_ffmpeg_probe() -> None:
    """Verifies that check_ffmpeg_availability() agrees with a direct ffmpeg probe of the host."""
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

    monkeypatch.setattr(target=saver_module.subprocess, name="run", value=_failing_run)

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

    monkeypatch.setattr(target=saver_module.subprocess, name="run", value=_failing_run)

    # The has_ffmpeg test fixture, the CLI's compatibility check, the MCP camera tools, and the VideoSystem saver gate
    # all call this probe directly. A propagated error would take down every caller instead of degrading them to
    # 'FFMPEG is not installed'.
    assert check_ffmpeg_availability() is False

    assert probe_calls[0]["args"][0] == "ffmpeg"
    assert probe_calls[0]["check"] is True


def test_video_saver_init_stores_parameters_and_renders_its_repr(tmp_path, has_ffmpeg) -> None:
    """Verifies that constructing a VideoSaver stores the requested parameters and renders them in its repr."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

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

    assert saver._system_id == 1
    assert saver._ffmpeg_process is None
    assert saver._ffmpeg_process is None

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
def test_video_saver_builds_the_cpu_encoder_command(
    tmp_path, video_encoder, gpu_index, output_pixel_format, has_ffmpeg
) -> None:
    """Verifies that a CPU encoder request builds an FFMPEG command carrying the requested settings."""
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

    assert "libx264" in saver._ffmpeg_command or "libx265" in saver._ffmpeg_command
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "veryfast" in saver._ffmpeg_command  # FASTEST maps to veryfast for CPU.
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
def test_video_saver_builds_the_gpu_encoder_command(
    tmp_path, video_encoder, output_pixel_format, has_nvidia, has_ffmpeg
) -> None:
    """Verifies that a GPU encoder request builds an FFMPEG command carrying the requested settings."""
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

    assert "h264_nvenc" in saver._ffmpeg_command or "hevc_nvenc" in saver._ffmpeg_command
    assert output_pixel_format.value in saver._ffmpeg_command
    assert "p1" in saver._ffmpeg_command  # FASTEST maps to p1 for GPU.
    gpu_index = saver._ffmpeg_command.index("-gpu")
    assert saver._ffmpeg_command[gpu_index + 1] == "0"
    # Pins the frame geometry and the quantization parameter, neither of which any other assertion would catch.
    size_index = saver._ffmpeg_command.index("-s")
    assert saver._ffmpeg_command[size_index + 1] == "320x240"
    quantization_index = saver._ffmpeg_command.index("-qp")
    assert saver._ffmpeg_command[quantization_index + 1] == "25"
    range_index = saver._ffmpeg_command.index("-color_range")
    assert saver._ffmpeg_command[range_index + 1] == "pc"


def test_video_saver_start_and_stop_are_idempotent(tmp_path, has_ffmpeg) -> None:
    """Verifies that repeated start() and stop() calls leave the encoder process in the state the first one set."""
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

    assert saver._ffmpeg_process is None

    saver.start()
    assert saver._ffmpeg_process is not None

    process = saver._ffmpeg_process
    saver.start()
    assert saver._ffmpeg_process is process  # Same process object.

    saver.stop()
    assert saver._ffmpeg_process is None

    saver.stop()
    assert saver._ffmpeg_process is None


def test_video_saver_save_frame_writes_a_playable_video_file(tmp_path, has_ffmpeg) -> None:
    """Verifies that the frames handed to save_frame() are encoded into a non-empty video file."""
    if not has_ffmpeg:
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_file = tmp_path / "test_video.mp4"
    frame_width = 100
    frame_height = 100

    camera = MockCamera(system_id=1, color=True, frame_rate=10, frame_width=frame_width, frame_height=frame_height)
    camera.connect()

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

    saver.start()

    for _ in range(20):
        frame = camera.grab_frame()
        saver.save_frame(frame)

    # Stops the encoder to finalize the video.
    saver.stop()

    assert output_file.exists()
    assert output_file.stat().st_size > 0  # File is not empty.


def test_video_saver_save_frame_rejects_an_unstarted_encoder(tmp_path, has_ffmpeg) -> None:
    """Verifies that saving a frame before the encoder process starts is rejected."""
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

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    message = (
        "Unable to submit the frame's data to the FFMPEG encoder process of the VideoSaver instance for the "
        "VideoSystem with id 1 as the process has not been started. Call the start() method "
        "to start the encoder process before calling the save_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        saver.save_frame(frame)


def test_video_saver_del_releases_the_encoder_process(tmp_path, has_ffmpeg) -> None:
    """Verifies that deleting a started VideoSaver runs its shutdown path and leaves the output file reusable."""
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

    saver.start()
    assert saver._ffmpeg_process is not None

    del saver

    # Creates a new saver over the same output file to confirm the path is reusable after the first instance is dropped.
    replacement_saver = VideoSaver(
        system_id=1,
        output_file=output_file,
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )
    replacement_saver.start()
    replacement_saver.stop()


def test_encoder_speed_presets_map_to_cpu_and_gpu_names() -> None:
    """Verifies that every encoder speed preset maps to the CPU and GPU preset names FFMPEG accepts."""
    for preset in EncoderSpeedPresets:
        assert isinstance(preset.gpu_preset, str)
        assert isinstance(preset.cpu_preset, str)

    assert EncoderSpeedPresets.FASTEST.gpu_preset == "p1"
    assert EncoderSpeedPresets.SLOWEST.gpu_preset == "p7"
    assert EncoderSpeedPresets.FASTEST.cpu_preset == "veryfast"
    assert EncoderSpeedPresets.SLOWEST.cpu_preset == "veryslow"


def test_video_saver_context_manager_starts_and_stops_the_encoder(tmp_path, has_ffmpeg) -> None:
    """Verifies that the context manager starts the encoder on entry and stops it on exit."""
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
        assert saver._ffmpeg_process is not None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        saver.save_frame(frame)

    assert saver._ffmpeg_process is None


def test_video_saver_save_frame_accepts_a_non_contiguous_frame(tmp_path, has_ffmpeg) -> None:
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


def test_video_saver_stop_reports_a_non_zero_encoder_exit_code(tmp_path, has_ffmpeg) -> None:
    """Verifies that stop() reaps an FFMPEG process that exited with a non-zero code and clears its reference."""
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

    saver.stop()
    assert saver._ffmpeg_process is None


def test_video_saver_save_frame_reports_a_terminated_encoder(tmp_path, has_ffmpeg) -> None:
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

    # Clears the dead process reference so that the __del__-driven stop() short-circuits instead of reaping again.
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
        assert saver._ffmpeg_process is None

        # A forced shutdown leaves the instance in the same state a clean one does: stopping again is a no-op, and
        # the saver remains usable for another recording.
        saver.stop()
        assert saver._ffmpeg_process is None
        saver.start()
        assert saver._ffmpeg_process is not None
    finally:
        # The no-op close transferred the ownership of the real pipe to this test.
        with suppress(OSError):
            stdin_stub._real.close()
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
        assert saver._ffmpeg_process is not None
        assert saver._ffmpeg_process.poll() is None
    finally:
        # Restoring the real pipe is load-bearing: stop() closes stdin to signal EOF, and the stub defines no close(),
        # so stop() would raise AttributeError before it could reap the encoder.
        saver._ffmpeg_process.stdin = real_stdin
        saver.stop()


def test_video_saver_stop_reports_a_failed_encoder_to_stderr(tmp_path, capsys) -> None:
    """Verifies that stop() reports a failed encoder to stderr even while the console is disabled."""
    saver = VideoSaver(
        system_id=1,
        output_file=tmp_path / "test.mp4",
        frame_width=100,
        frame_height=100,
        frame_rate=10.0,
        gpu=-1,
    )

    # stop() runs inside the spawned consumer process, which re-imports the library and therefore always holds a
    # freshly disabled console. Reproducing that state is what makes this a regression guard, because a console-based
    # report is discarded outright in it.
    console_was_enabled = console.enabled
    console.disable()
    saver._ffmpeg_process = _FailedEncoderProcess(returncode=1)
    saver._stderr_output = b"Invalid data found when processing input"

    try:
        saver.stop()
    finally:
        if console_was_enabled:
            console.enable()

    captured = capsys.readouterr()
    assert "FFMPEG encoder error (system 1, exit code 1)" in captured.err
    assert "Invalid data found when processing input" in captured.err

    # The MCP server carries its JSON-RPC message stream over stdout, so an encoder diagnostic that landed there
    # would render the message it interleaves with unparsable for the connected client.
    assert captured.out == ""


class _FailedEncoderProcess:
    """Stands in for an FFMPEG process that has already exited with an error, exposing no pipes to close.

    Attributes:
        returncode: Stores the exit code the stand-in process reports to every caller that reaps it.
        stdin: Stores the absent input pipe, standing in for the pipe an exited process no longer exposes.
        stderr: Stores the absent error pipe, standing in for the pipe an exited process no longer exposes.
    """

    def __init__(self, returncode) -> None:
        self.returncode = returncode
        self.stdin = None
        self.stderr = None

    def wait(self, timeout=None) -> int:  # noqa: ARG002 - mirrors the Popen.wait signature stop() calls by keyword.
        """Returns the preset exit code immediately, as the process it stands in for has already terminated."""
        return self.returncode


class _NoEofStdin:
    """Wraps the encoder's real stdin pipe so that closing it does not signal EOF to the FFMPEG process.

    Attributes:
        _real: Stores the wrapped stdin pipe, which the test closes once it no longer needs FFMPEG alive.
    """

    def __init__(self, stream) -> None:
        self._real = stream

    def close(self) -> None:
        """Ignores the close request, keeping the wrapped pipe open so that FFMPEG never exits on its own."""


class _FailingStdin:
    """Stands in for the encoder's stdin pipe and fails every write with a preset error.

    Attributes:
        _error: Stores the error raised in place of accepting the frame's data on every write.
    """

    def __init__(self, error) -> None:
        self._error = error

    def write(self, _data) -> int:
        """Raises the preset error instead of accepting the frame's data."""
        raise self._error
