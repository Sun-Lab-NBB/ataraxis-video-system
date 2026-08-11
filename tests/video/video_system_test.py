"""Contains tests for classes and methods provided by the video_system.py module."""

import re
import sys
from queue import Queue
from random import randint
import threading

import cv2
import numpy as np
import pytest
from ataraxis_time import (
    PrecisionTimer,
    TimerPrecisions,
    TimestampFormats,
    get_timestamp,
    convert_timestamp,
)
from tests.fake_opencv import FakeVideoCapture, build_capture_factory
from tests.log_archives import create_test_archive
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, DataLogger, SharedMemoryArray, assemble_log_archives

from ataraxis_video_system import (
    VideoSystem,
    VideoEncoders,
    CameraInterfaces,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_ffmpeg_availability,
)
from ataraxis_video_system.video import video_system as video_system_module
from ataraxis_video_system.video.camera import MockCamera, OpenCVCamera
from ataraxis_video_system.video.timestamps import extract_logged_camera_timestamps
from ataraxis_video_system.video.video_system import _empty_display_queue


class _RecordingSaver:
    """Records the encoder calls the frame saving loop makes, standing in for a VideoSaver without requiring FFMPEG."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.frames: list = []

    def start(self) -> None:
        """Records the encoder startup."""
        self.calls.append("start")

    def save_frame(self, frame) -> None:
        """Records the frame handed to the encoder."""
        self.calls.append("save_frame")
        self.frames.append(frame)

    def stop(self) -> None:
        """Records the encoder shutdown."""
        self.calls.append("stop")


class _ExplodingSaver:
    """Fails on the first frame handed to it, standing in for an encoder whose pipe dies in the middle of a
    recording.
    """

    def __init__(self) -> None:
        self.stopped: bool = False

    def start(self) -> None:
        """Simulates a successful encoder startup."""

    def save_frame(self, frame) -> None:
        """Fails the way the encoder does when its input pipe is closed underneath it."""
        message = f"ffmpeg pipe closed while saving a {frame.shape} frame"
        raise RuntimeError(message)

    def stop(self) -> None:
        """Records the encoder shutdown."""
        self.stopped = True


class _ExplodingCamera:
    """Fails on the first frame grab, standing in for a camera whose link to the hardware drops mid-acquisition."""

    def __init__(self) -> None:
        self.disconnected: bool = False

    def connect(self) -> None:
        """Simulates a successful connection to the camera hardware."""

    def grab_frame(self):
        """Fails the way a camera interface does when the link to the hardware drops."""
        message = "camera link lost"
        raise RuntimeError(message)

    def disconnect(self) -> None:
        """Records the release of the camera handle."""
        self.disconnected = True


def _wait_until(predicate, timeout_seconds: int = 15) -> bool:
    """Blocks until the input predicate holds or the timeout elapses and reports whether the predicate held."""
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    timeout_milliseconds = timeout_seconds * 1000
    while timer.elapsed < timeout_milliseconds:
        if predicate():
            return True
        timer.delay(delay=5, allow_sleep=True, block=False)
    return predicate()


def _drain_queue(target_queue) -> list:
    """Removes and returns every item buffered in the input queue."""
    items = []
    while not target_queue.empty():
        items.append(target_queue.get())
    return items


def _consumer_exit_at_once(*_arguments) -> None:
    """Stands in for the consumer loop and returns at once, so the consumer process dies before it reports
    initialization.
    """


def _consumer_exit_on_command(system_id, saver, saver_queue, logger_queue, terminator_array) -> None:
    """Reports consumer initialization and exits once index 1 of the terminator array is set, standing in for a
    consumer process that dies in the middle of a recording.
    """
    terminator_array.connect()
    terminator_array[3] = 1

    # Bounds the wait, so that a test which never issues the exit command still releases the process.
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    while not terminator_array[1] and timer.elapsed < 60000:
        timer.delay(delay=10, allow_sleep=True, block=False)

    terminator_array.disconnect()


def _producer_report_then_exit(
    system_id, camera, display_frame_rate, saver_queue, logger_queue, terminator_array
) -> None:
    """Reports producer initialization and exits at once, standing in for a producer that dies immediately after
    start() accepts its handshake.
    """
    terminator_array.connect()
    terminator_array[2] = 1
    terminator_array.disconnect()


@pytest.fixture
def data_logger(tmp_path) -> DataLogger:
    """Creates a DataLogger instance and returns it to the caller."""
    return DataLogger(output_directory=tmp_path, instance_name=str(randint(a=0, b=100000000000)))


def test_init_repr(tmp_path, data_logger) -> None:
    """Verifies the functioning of the VideoSystem __init__() and __repr__() methods."""
    vs_instance = VideoSystem(
        system_id=np.uint8(1),
        data_logger=data_logger,
        name="test_camera",
        output_directory=tmp_path / "test_output_directory",
        camera_interface=CameraInterfaces.MOCK,
        camera_index=0,
    )

    # Verifies class properties
    assert vs_instance.system_id == np.uint8(1)
    assert not vs_instance.started
    assert vs_instance.video_file_path is not None

    # Verifies the __repr__() method.
    representation_string: str = (
        f"VideoSystem(system_id={np.uint8(1)}, started={False}, camera=MockCamera, frame_saving={True})"
    )
    assert repr(vs_instance) == representation_string


def test_init_errors(data_logger) -> None:
    """Verifies the error handling behavior of the VideoSystem initialization method."""
    # Invalid system_id input - causes conversion error
    invalid_system_id = "str"
    with pytest.raises((TypeError, ValueError)):
        VideoSystem(
            system_id=invalid_system_id,  # type: ignore[arg-type]
            data_logger=data_logger,
            name="test_camera",
            output_directory=data_logger.output_directory,
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid data_logger input
    invalid_data_logger = None
    message = (
        f"Unable to initialize the VideoSystem instance with id 1. Expected an initialized "
        f"DataLogger instance as the 'data_logger' argument value, but encountered {invalid_data_logger} of type "
        f"{type(invalid_data_logger).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=invalid_data_logger,  # type: ignore[arg-type]
            name="test_camera",
            output_directory=data_logger.output_directory,
        )

    # Invalid output_directory input
    invalid_output_directory = "Not a Path"
    message = (
        f"Unable to initialize the VideoSystem instance with id 1. Expected a Path instance or None "
        f"as the 'output_directory' argument's value, but encountered {invalid_output_directory} of type "
        f"{type(invalid_output_directory).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=invalid_output_directory,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )


@pytest.mark.parametrize("invalid_name", ["", None, 5])
def test_init_rejects_an_invalid_name(data_logger, tmp_path, invalid_name) -> None:
    """Verifies that VideoSystem rejects a camera name that is not a non-empty string."""
    message = (
        f"Unable to initialize the VideoSystem instance with id 1. Expected a non-empty string as the 'name' "
        f"argument value, but encountered {invalid_name!r} of type {type(invalid_name).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name=invalid_name,  # type: ignore[arg-type]
            output_directory=tmp_path / "test_output_directory",
            camera_interface=CameraInterfaces.MOCK,
        )


def test_init_rejects_a_camera_acquiring_wider_than_eight_bit_frames(data_logger, tmp_path, monkeypatch) -> None:
    """Verifies that a camera acquiring frames above 8 bits per component is rejected at construction."""

    def _wide_frame(self):
        """Returns the 16-bit frame a GenICam camera configured for Mono12 or Mono16 delivers."""
        return np.zeros((4, 4), dtype=np.uint16)

    monkeypatch.setattr(MockCamera, "grab_frame", _wide_frame)

    # Every InputPixelFormats member describes 8 bits per component, so a wider frame hands the saver twice the bytes
    # the FFMPEG command declares and is demuxed as two frames, breaking frame-to-timestamp alignment.
    message = (
        "Unable to configure the camera interface for the VideoSystem with id 1. The managed camera acquires frames "
        "using the 'uint16' data type, but this library encodes 8-bit frames only. Reconfigure the camera to use an "
        "8-bit pixel format, such as Mono8 or BGR8."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=tmp_path / "test_output_directory",
            camera_interface=CameraInterfaces.MOCK,
        )


def test_init_builds_the_opencv_interface(data_logger, monkeypatch) -> None:
    """Verifies that the OpenCV branch of the interface selection builds, validates, and queries its camera."""
    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: FakeVideoCapture()}))

    # The output directory is left unset, so the selection is exercised without requiring FFMPEG on the host.
    video_system = VideoSystem(
        system_id=np.uint8(60),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,
        camera_interface=CameraInterfaces.OPENCV,
        camera_index=0,
    )

    assert isinstance(video_system._camera, OpenCVCamera)
    assert video_system._camera.frame_width == 640
    assert video_system._camera.frame_height == 480
    assert not video_system._camera.is_connected


def test_empty_display_queue_releases_every_retained_frame() -> None:
    """Verifies that the frames a departed display thread never consumed are released rather than retained."""
    display_queue: Queue = Queue()
    for _ in range(5):
        display_queue.put(np.zeros((4, 4, 3), dtype=np.uint8))

    _empty_display_queue(display_queue=display_queue)

    assert display_queue.empty()
    # The unfinished-task count is what a join() on the queue waits for, so each discarded frame is also retired.
    assert display_queue.unfinished_tasks == 0


def test_camera_configuration_errors(data_logger, tmp_path) -> None:
    """Verifies the error handling behavior of camera configuration during VideoSystem initialization."""
    output_directory = tmp_path / "test_output_directory"

    # Invalid camera index
    invalid_index = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Expected a "
        f"zero or positive integer as the 'camera_index' argument value, but got {invalid_index} of type "
        f"{type(invalid_index).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_index=invalid_index,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid frame rate
    invalid_frame_rate = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Expected a "
        f"positive integer or None as the 'frame_rate' argument value, but got "
        f"{invalid_frame_rate} of type {type(invalid_frame_rate).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            frame_rate=invalid_frame_rate,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid frame width
    invalid_frame_width = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Expected a "
        f"positive integer or None as the 'frame_width' argument value, but got {invalid_frame_width} of type "
        f"{type(invalid_frame_width).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            frame_width=invalid_frame_width,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid frame height
    invalid_frame_height = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Expected a "
        f"positive integer or None as the 'frame_height' argument value, but got {invalid_frame_height} of type "
        f"{type(invalid_frame_height).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            frame_height=invalid_frame_height,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid camera interface
    invalid_interface = "invalid"
    message_pattern = "Unable to configure the camera interface.*unsupported camera_interface"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=invalid_interface,  # type: ignore[arg-type]
        )


def test_video_saver_configuration(data_logger, tmp_path, has_nvidia) -> None:
    """Verifies the functioning of video saver configuration during VideoSystem initialization."""
    output_directory = tmp_path / "test_output_directory"

    # Tests GPU encoding if NVIDIA GPU is available, otherwise tests CPU encoding
    if has_nvidia and check_ffmpeg_availability():
        vs = VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            gpu=0,
            video_encoder=VideoEncoders.H265,
            encoder_speed_preset=EncoderSpeedPresets.FASTEST,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
        )
        assert vs._saver is not None
    elif check_ffmpeg_availability():
        vs = VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            gpu=-1,
            video_encoder=VideoEncoders.H265,
            encoder_speed_preset=EncoderSpeedPresets.FASTEST,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
        )
        assert vs._saver is not None


def test_init_rejects_a_host_without_ffmpeg(data_logger, tmp_path, monkeypatch) -> None:
    """Verifies that a host without FFMPEG is rejected at construction, before the video saver is built."""
    # The probe is answered rather than consulted, so the rejection is verified on a host that does have FFMPEG.
    monkeypatch.setattr(video_system_module, "check_ffmpeg_availability", lambda: False)

    output_directory = tmp_path / "missing_ffmpeg_output"
    message = (
        "Unable to configure the video saver for the VideoSystem with id 121. VideoSaver requires a third-party "
        "software, FFMPEG, to be available on the system's Path. Make sure FFMPEG is installed and callable from a "
        "Python shell. See https://www.ffmpeg.org/download.html for more information."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(121),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
        )

    # The saver creates its output directory as part of its construction, so the absence of that directory proves the
    # guard preempts the saver instead of tearing one down after the fact.
    assert not output_directory.exists()


def test_init_rejects_gpu_encoding_without_an_nvidia_gpu(data_logger, tmp_path, monkeypatch) -> None:
    """Verifies that GPU encoding requested on a host without an NVIDIA GPU is rejected at construction."""
    # Both probes are answered, so the outcome depends on the requested GPU index alone rather than on the host.
    monkeypatch.setattr(video_system_module, "check_ffmpeg_availability", lambda: True)
    monkeypatch.setattr(video_system_module, "check_gpu_availability", lambda: False)

    output_directory = tmp_path / "missing_gpu_output"
    message = (
        "Unable to configure the video saver for the VideoSystem with id 122. The saver is configured to use the GPU "
        "video encoder, which currently only supports NVIDIA GPUs. Calling 'nvidia-smi' to verify the presence of "
        "NVIDIA GPUs did not run successfully, indicating that there are no available NVIDIA GPUs on the host system. "
        "Use a CPU encoder or make sure nvidia-smi is callable from a Python shell."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(122),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            gpu=0,
        )

    # The guard is gated on the GPU index: the same host builds a CPU-encoding saver without consulting the GPU probe.
    video_system = VideoSystem(
        system_id=np.uint8(122),
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        gpu=-1,
    )
    assert video_system._saver is not None


def test_video_saver_configuration_errors(data_logger, tmp_path) -> None:
    """Verifies the error handling behavior of video saver configuration during VideoSystem initialization."""
    output_directory = tmp_path / "test_output_directory"

    # Invalid gpu index
    invalid_gpu = "str"
    message = (
        f"Unable to configure the video saver for the VideoSystem with id 1. Expected an "
        f"integer as the 'gpu' argument value, but got {invalid_gpu} of type {type(invalid_gpu).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            gpu=invalid_gpu,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid video encoder
    invalid_encoder = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'video_encoder'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            video_encoder=invalid_encoder,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid encoder preset
    invalid_preset = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'encoder_speed_preset'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            encoder_speed_preset=invalid_preset,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid output pixel format
    invalid_format = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'output_pixel_format'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            output_pixel_format=invalid_format,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid quantization parameter
    invalid_qp = "str"
    message = (
        f"Unable to configure the video saver for the VideoSystem with id 1. Expected an "
        f"integer between 0 and 51 as the 'quantization_parameter' argument value, but got "
        f"{invalid_qp} of type {type(invalid_qp).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            quantization_parameter=invalid_qp,  # type: ignore[arg-type]
            camera_interface=CameraInterfaces.MOCK,
        )

    # The encoder rejects a negative quantization parameter outright, so the guard admits 0 through 51 alone rather
    # than deferring the choice to the encoder.
    message = (
        "Unable to configure the video saver for the VideoSystem with id 1. Expected an "
        "integer between 0 and 51 as the 'quantization_parameter' argument value, but got -1 of type int."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            quantization_parameter=-1,
            camera_interface=CameraInterfaces.MOCK,
        )


def test_start_stop(data_logger, tmp_path) -> None:
    """Verifies the functioning of the start(), stop(), start_frame_saving() and stop_frame_saving() methods of the
    VideoSystem class.

    Also verifies internal DataLogger bindings and logged timestamp extraction methods.
    """
    # Creates two VideoSystem instances with different configurations
    output_directory = tmp_path / "test_output_directory"

    video_system_1 = VideoSystem(
        system_id=np.uint8(101),
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=10,
        display_frame_rate=None,
        quantization_parameter=40,
    )

    video_system_2 = VideoSystem(
        system_id=np.uint8(202),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,  # No saving configured
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=5,
    )

    # Starts all instances
    data_logger.start()
    video_system_1.start()
    # Ensures that calling start twice does nothing.
    video_system_1.start()
    video_system_2.start()

    assert video_system_1.started
    assert video_system_2.started

    # Tests frame saving control
    timer = PrecisionTimer(precision="s")
    video_system_1.start_frame_saving()
    timer.delay(delay=2, allow_sleep=True, block=False)
    video_system_1.stop_frame_saving()

    # Ensures that the saving commands are harmless for system 2, which has no video saver configured.
    video_system_2.start_frame_saving()
    video_system_2.stop_frame_saving()

    # Re-enables saving for system 1
    video_system_1.start_frame_saving()
    timer.delay(delay=2, allow_sleep=True, block=False)
    video_system_1.stop_frame_saving()

    # Stops the video systems
    video_system_1.stop()
    video_system_2.stop()
    # Ensures that calling stop twice does nothing.
    video_system_2.stop()

    assert not video_system_1.started
    assert not video_system_2.started

    # Compresses logs for timestamp extraction
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, memory_mapping=False)

    # Extracts frame timestamps for system 1 (which saved frames)
    log_path_1 = data_logger.output_directory / "101_log.npz"
    frame_timestamps_1 = extract_logged_camera_timestamps(log_path=log_path_1, workers=1)
    # With fps of 10 and running for ~4 seconds total, should have acquired around 40 frames
    assert 35 <= len(frame_timestamps_1) <= 45

    # Tests the system without frame saving
    video_system_3 = VideoSystem(
        system_id=np.uint8(234),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,  # No output directory
        camera_interface=CameraInterfaces.MOCK,
    )
    video_system_3.start()
    timer.delay(delay=1, allow_sleep=True, block=False)
    video_system_3.stop()
    data_logger.stop()


@pytest.mark.xdist_group(name="video_system")
def test_start_reports_a_producer_that_never_initializes(data_logger, monkeypatch) -> None:
    """Verifies that a producer that does not report initialization in time is reclaimed and reported."""
    # A spawned interpreter cannot report initialization within the first 20-millisecond poll tick, so a zero timeout
    # takes the stall arm deterministically.
    monkeypatch.setattr(video_system_module, "_PROCESS_INITIALIZATION_TIMEOUT", 0)

    # The output directory is left unset, so the failed startup leaves no encoder subprocess behind.
    video_system = VideoSystem(
        system_id=np.uint8(123),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=30,
        frame_width=32,
        frame_height=32,
    )

    message = (
        "Unable to start the VideoSystem with id 123. The producer process has unexpectedly shut down or stalled for "
        "more than 20 seconds during initialization. This likely indicates a problem with the camera interface "
        "instance managed by the process."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        video_system.start()

    assert not video_system.started

    # The child is terminated and reaped rather than abandoned as an orphan.
    assert not video_system._producer_process.is_alive()

    # The failed startup also unlinked its buffer, which is what allows its name to be claimed exclusively again.
    probe_array = SharedMemoryArray.create_array(
        name="123_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=False
    )
    probe_array.destroy()

    # The failed startup left nothing to reclaim, so a follow-up shutdown is a no-op instead of a second teardown that
    # would reach the destroyed buffer.
    video_system.stop()


@pytest.mark.xdist_group(name="video_system")
def test_start_reports_a_consumer_that_never_initializes(data_logger, tmp_path, monkeypatch) -> None:
    """Verifies that a consumer that dies during initialization is reported as the consumer rather than the
    producer.
    """
    # The stand-in consumer never starts the encoder, so the saver is built without requiring FFMPEG on the host.
    monkeypatch.setattr(video_system_module, "check_ffmpeg_availability", lambda: True)
    monkeypatch.setattr(VideoSystem, "_frame_saving_loop", staticmethod(_consumer_exit_at_once))

    video_system = VideoSystem(
        system_id=np.uint8(124),
        data_logger=data_logger,
        name="test_camera",
        output_directory=tmp_path / "consumer_startup_output",
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=30,
        frame_width=32,
        frame_height=32,
    )

    message = (
        "Unable to start the VideoSystem with id 124. The consumer process has unexpectedly shut down or stalled for "
        "more than 20 seconds during initialization. This likely indicates a problem with the VideoSaver instance "
        "managed by the process."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        video_system.start()

    assert not video_system.started

    # The producer that did initialize is terminated alongside the consumer that did not.
    assert not video_system._producer_process.is_alive()

    probe_array = SharedMemoryArray.create_array(
        name="124_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=False
    )
    probe_array.destroy()

    video_system.stop()


@pytest.mark.xdist_group(name="video_system")
def test_frame_production_loop_streams_frames_and_honors_the_saving_gate() -> None:
    """Verifies that the production loop logs its onset, reports readiness, and gates frames on the saving flag."""
    terminator_array = SharedMemoryArray.create_array(
        name="125_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )
    camera = MockCamera(system_id=125, frame_rate=30, frame_width=32, frame_height=32, color=True)
    saver_queue: Queue = Queue()
    logger_queue: Queue = Queue()
    loop_thread = threading.Thread(
        target=VideoSystem._frame_production_loop,
        kwargs={
            "system_id": np.uint8(125),
            "camera": camera,
            "display_frame_rate": 0,
            "saver_queue": saver_queue,
            "logger_queue": logger_queue,
            "terminator_array": terminator_array,
        },
        daemon=True,
    )

    # Brackets the onset stamp the loop logs, which is the anchor every extracted frame timestamp is measured from.
    onset_lower_bound = get_timestamp(output_format=TimestampFormats.INTEGER)
    try:
        loop_thread.start()

        # Index 2 is the readiness handshake the start() method polls for before it returns.
        assert _wait_until(predicate=lambda: terminator_array[2] == 1)
        onset_upper_bound = get_timestamp(output_format=TimestampFormats.INTEGER)

        # Frames reach the consumer only while index 1 is set, which is the mechanism start_frame_saving() drives.
        assert saver_queue.qsize() == 0
        terminator_array[1] = 1
        assert _wait_until(predicate=lambda: saver_queue.qsize() >= 5)

        terminator_array[1] = 0
        gated_size = saver_queue.qsize()

        # Spans several frame periods, so a loop that kept forwarding frames past the flag would be caught.
        PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=300, allow_sleep=True, block=False)

        # Allows for the single frame that may have been in flight when the flag was cleared.
        assert saver_queue.qsize() <= gated_size + 1
    finally:
        if terminator_array.is_connected:
            terminator_array[0] = 1
        loop_thread.join(timeout=30)
        terminator_array.destroy()

    assert not loop_thread.is_alive()

    # The loop releases the camera handle on its way out, even though it was stopped through the array.
    assert not camera.is_connected

    # The producer logs exactly one entry, the onset anchor. The per-frame entries are the consumer's responsibility.
    logger_entries = _drain_queue(target_queue=logger_queue)
    assert len(logger_entries) == 1
    assert logger_entries[0].source_id == np.uint8(125)
    assert logger_entries[0].acquisition_time == np.uint64(0)
    onset_microseconds = convert_timestamp(
        timestamp=logger_entries[0].serialized_data, output_format=TimestampFormats.INTEGER
    )
    assert onset_lower_bound <= onset_microseconds <= onset_upper_bound

    # Every forwarded frame carries the camera's own geometry and a timestamp that advances with the acquisition.
    frame_entries = _drain_queue(target_queue=saver_queue)
    assert len(frame_entries) >= 5
    previous_stamp = 0
    for frame, frame_stamp in frame_entries:
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.uint8
        assert frame_stamp > previous_stamp
        previous_stamp = frame_stamp


@pytest.mark.xdist_group(name="video_system")
def test_frame_production_loop_releases_a_departed_display_thread(monkeypatch) -> None:
    """Verifies that the production loop stops feeding and releases the display queue once its thread ends."""
    display_queues: list[Queue] = []
    displayed_frames: list = []

    def _drain_two_frames_then_return(display_queue, system_id) -> None:
        """Stands in for the display loop and returns after two frames, as the real loop does when the user dismisses
        the display window.
        """
        display_queues.append(display_queue)
        for _ in range(2):
            displayed_frames.append(display_queue.get())
            display_queue.task_done()

    monkeypatch.setattr(VideoSystem, "_frame_display_loop", staticmethod(_drain_two_frames_then_return))

    terminator_array = SharedMemoryArray.create_array(
        name="126_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )
    camera = MockCamera(system_id=126, frame_rate=30, frame_width=32, frame_height=32, color=True)
    saver_queue: Queue = Queue()
    loop_thread = threading.Thread(
        target=VideoSystem._frame_production_loop,
        kwargs={
            "system_id": np.uint8(126),
            "camera": camera,
            "display_frame_rate": 30,
            "saver_queue": saver_queue,
            "logger_queue": Queue(),
            "terminator_array": terminator_array,
        },
        daemon=True,
    )

    try:
        loop_thread.start()
        assert _wait_until(predicate=lambda: terminator_array[2] == 1)
        terminator_array[1] = 1

        # The stand-in consumes two frames and returns, which is what leaves the departed-thread branch to be taken.
        assert _wait_until(predicate=lambda: len(displayed_frames) == 2)
        acquired_frames = saver_queue.qsize()

        # The acquisition continues past the display thread's exit, spanning several further display cycles.
        assert _wait_until(predicate=lambda: saver_queue.qsize() >= acquired_frames + 6)
    finally:
        if terminator_array.is_connected:
            terminator_array[0] = 1
        loop_thread.join(timeout=30)
        terminator_array.destroy()

    # The frames the display thread received while it lived arrive with the camera's own geometry.
    assert len(displayed_frames) == 2
    for frame in displayed_frames:
        assert frame.shape == (32, 32, 3)

    # The queue the departed thread drained is emptied and dropped rather than grown by one frame per display cycle
    # for the rest of the acquisition, which is the memory the release exists to return to the producer process.
    assert display_queues[0].qsize() == 0


@pytest.mark.xdist_group(name="video_system")
def test_frame_production_loop_shuts_down_a_live_display_thread(monkeypatch) -> None:
    """Verifies that the production loop hands a live display thread its sentinel and waits for that thread to end."""
    received: list = []
    finished: list[bool] = []

    def _display_until_the_sentinel(display_queue, system_id) -> None:
        """Stands in for the display loop and ends on the shutdown sentinel, as the real loop does."""
        while True:
            item = display_queue.get()
            display_queue.task_done()
            received.append(item)
            if not isinstance(item, np.ndarray):
                break

        # Outlasts the producer's own return, so a producer that did not wait for this thread would be observed with
        # the completion flag still unset.
        PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=250, allow_sleep=True, block=False)
        finished.append(True)

    monkeypatch.setattr(VideoSystem, "_frame_display_loop", staticmethod(_display_until_the_sentinel))

    terminator_array = SharedMemoryArray.create_array(
        name="127_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )
    camera = MockCamera(system_id=127, frame_rate=30, frame_width=32, frame_height=32, color=True)
    loop_thread = threading.Thread(
        target=VideoSystem._frame_production_loop,
        kwargs={
            "system_id": np.uint8(127),
            "camera": camera,
            "display_frame_rate": 30,
            "saver_queue": Queue(),
            "logger_queue": Queue(),
            "terminator_array": terminator_array,
        },
        daemon=True,
    )

    try:
        loop_thread.start()
        assert _wait_until(predicate=lambda: terminator_array[2] == 1)
        assert _wait_until(predicate=lambda: len(received) >= 2)
    finally:
        if terminator_array.is_connected:
            terminator_array[0] = 1
        loop_thread.join(timeout=30)
        terminator_array.destroy()

    assert not loop_thread.is_alive()

    # The shutdown sentinel is what ends the display loop, and the producer blocks on that thread until it finishes.
    # A producer that skipped either step would hang the process or abandon the thread mid-frame.
    assert received[-1] is None
    assert finished == [True]


@pytest.mark.xdist_group(name="video_system")
def test_frame_production_loop_surfaces_a_camera_failure(capsys) -> None:
    """Verifies that a camera failure is written to the terminal and still releases the camera and the array."""
    camera = _ExplodingCamera()
    terminator_array = SharedMemoryArray.create_array(
        name="128_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )

    try:
        with pytest.raises(RuntimeError, match="camera link lost"):
            VideoSystem._frame_production_loop(
                system_id=np.uint8(128),
                camera=camera,
                display_frame_rate=0,
                saver_queue=Queue(),
                logger_queue=Queue(),
                terminator_array=terminator_array,
            )

        # The producer process dies visibly rather than silently, which is what makes the watchdog's report
        # actionable, and it releases both of its handles on the way out.
        assert "camera link lost" in capsys.readouterr().err
        assert camera.disconnected
        assert not terminator_array.is_connected
    finally:
        terminator_array.destroy()


@pytest.mark.xdist_group(name="video_system")
def test_frame_saving_loop_pairs_every_frame_with_its_timestamp() -> None:
    """Verifies that the saving loop reports readiness and logs one timestamp per frame handed to the encoder."""
    terminator_array = SharedMemoryArray.create_array(
        name="129_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )
    saver = _RecordingSaver()
    saver_queue: Queue = Queue()
    logger_queue: Queue = Queue()

    frame_stamps = (1000, 2500, 4100, 5000, 7300, 9001, 11000, 13400, 15000, 17250)
    frames = tuple(np.full(shape=(4, 4, 3), fill_value=index, dtype=np.uint8) for index in range(len(frame_stamps)))
    for frame, frame_stamp in zip(frames, frame_stamps, strict=True):
        saver_queue.put((frame, frame_stamp))

    # The end-of-stream sentinel the shutdown appends, which is the loop's only exit.
    saver_queue.put(None)

    try:
        VideoSystem._frame_saving_loop(
            system_id=np.uint8(129),
            saver=saver,
            saver_queue=saver_queue,
            logger_queue=logger_queue,
            terminator_array=terminator_array,
        )

        # The loop releases the shared handle in its finally, so the handshake is read through a fresh connection.
        # Index 3 is the readiness signal the start() method polls for before it returns.
        terminator_array.connect()
        assert terminator_array[3] == 1
    finally:
        terminator_array.destroy()

    # The encoder is started once, fed every frame in queue order, and stopped after the last frame rather than
    # abandoned while it still holds an unfinalized video file.
    assert saver.calls == ["start", *["save_frame"] * len(frames), "stop"]
    for handed_frame, expected_frame in zip(saver.frames, frames, strict=True):
        np.testing.assert_array_equal(handed_frame, expected_frame)

    # Exactly one log entry per encoded frame, in input order, each carrying that frame's own acquisition timestamp.
    # A pairing that shifted by even one entry would describe every frame of the recording with the wrong timestamp.
    logger_entries = _drain_queue(target_queue=logger_queue)
    assert len(logger_entries) == len(frame_stamps)
    for entry, expected_stamp in zip(logger_entries, frame_stamps, strict=True):
        assert entry.source_id == np.uint8(129)
        assert entry.acquisition_time == np.uint64(expected_stamp)
        assert entry.serialized_data.size == 0


@pytest.mark.xdist_group(name="video_system")
def test_frame_saving_loop_surfaces_an_encoder_failure(capsys) -> None:
    """Verifies that an encoder failure is written to the terminal and still stops the encoder and the array."""
    terminator_array = SharedMemoryArray.create_array(
        name="130_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=True
    )
    saver = _ExplodingSaver()
    saver_queue: Queue = Queue()
    saver_queue.put((np.zeros(shape=(4, 4, 3), dtype=np.uint8), 1000))

    try:
        with pytest.raises(RuntimeError, match="ffmpeg pipe closed"):
            VideoSystem._frame_saving_loop(
                system_id=np.uint8(130),
                saver=saver,
                saver_queue=saver_queue,
                logger_queue=Queue(),
                terminator_array=terminator_array,
            )

        # A mid-recording encoder failure still runs the cleanup, so the consumer process dies visibly with its
        # encoder stopped instead of dying silently while the encoder still holds the file.
        assert "ffmpeg pipe closed" in capsys.readouterr().err
        assert saver.stopped
        assert not terminator_array.is_connected
    finally:
        terminator_array.destroy()


@pytest.mark.xdist_group(name="video_system")
def test_watchdog_reports_and_cleans_up_after_a_dead_producer(data_logger) -> None:
    """Verifies that a producer that dies mid-recording is reported and has its resources reclaimed."""
    video_system = VideoSystem(
        system_id=np.uint8(131),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=30,
        frame_width=32,
        frame_height=32,
    )

    # The watchdog raises on its own thread, so the terminal error is captured through the thread excepthook. The hook
    # is saved and restored explicitly, since pytest installs its own around the call phase and monkeypatch's undo
    # would reinstate an already-exited catcher.
    thread_errors: list = []
    original_excepthook = threading.excepthook
    threading.excepthook = thread_errors.append
    try:
        video_system.start()
        video_system._producer_process.kill()
        assert _wait_until(predicate=lambda: len(thread_errors) > 0, timeout_seconds=30)
    finally:
        threading.excepthook = original_excepthook

        # The watchdog and this call claim the same teardown, so this is a no-op after a watchdog that got there
        # first, and it reclaims the resources of a watchdog that did not.
        video_system.stop()

    message = (
        "The producer process for the VideoSystem with id 131 has been prematurely shut down. This likely indicates "
        "that the process has encountered a runtime error that terminated the process."
    )
    assert thread_errors[0].exc_type is RuntimeError
    assert re.search(error_format(message), str(thread_errors[0].exc_value)) is not None

    # The instance marks itself stopped rather than continuing to report a recording that is no longer running.
    assert not video_system.started

    # The buffer reference is dropped, which is what makes a concurrent stop() take its own early return instead of
    # reaching a buffer this thread has already destroyed.
    assert video_system._terminator_array is None

    # The buffer itself was unlinked, so its name can be claimed exclusively again.
    probe_array = SharedMemoryArray.create_array(
        name="131_terminator_array", prototype=np.zeros(shape=4, dtype=np.uint8), exists_ok=False
    )
    probe_array.destroy()


@pytest.mark.xdist_group(name="video_system")
def test_watchdog_reports_a_dead_consumer(data_logger, tmp_path, monkeypatch) -> None:
    """Verifies that a consumer that dies mid-recording is reported as the consumer rather than the producer."""
    # The stand-in consumer never starts the encoder, so the saver is built without requiring FFMPEG on the host.
    monkeypatch.setattr(video_system_module, "check_ffmpeg_availability", lambda: True)
    monkeypatch.setattr(VideoSystem, "_frame_saving_loop", staticmethod(_consumer_exit_on_command))

    video_system = VideoSystem(
        system_id=np.uint8(132),
        data_logger=data_logger,
        name="test_camera",
        output_directory=tmp_path / "dead_consumer_output",
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=30,
        frame_width=32,
        frame_height=32,
    )

    thread_errors: list = []
    original_excepthook = threading.excepthook
    threading.excepthook = thread_errors.append
    try:
        video_system.start()

        # The stand-in consumer treats index 1 as its exit command, so this ends the consumer alone and leaves the
        # producer running, which is the state the consumer arm of the watchdog exists to discriminate.
        video_system._terminator_array[1] = 1
        assert _wait_until(predicate=lambda: len(thread_errors) > 0, timeout_seconds=30)
    finally:
        threading.excepthook = original_excepthook
        video_system.stop()

    message = (
        "The consumer process for the VideoSystem with id 132 has been prematurely shut down. This likely indicates "
        "that the process has encountered a runtime error that terminated the process."
    )
    assert thread_errors[0].exc_type is RuntimeError
    assert re.search(error_format(message), str(thread_errors[0].exc_value)) is not None
    assert not video_system.started
    assert video_system._terminator_array is None


@pytest.mark.xdist_group(name="video_system")
def test_watchdog_activates_only_after_the_startup_completes(data_logger, monkeypatch) -> None:
    """Verifies that the watchdog leaves a VideoSystem alone until start() marks the instance as started."""
    observations: list[tuple[bool, bool]] = []

    # The producer stand-in reports initialization and exits at once, so start() accepts the handshake and hands the
    # watchdog a system whose producer is already dead.
    monkeypatch.setattr(VideoSystem, "_frame_production_loop", staticmethod(_producer_report_then_exit))

    video_system = VideoSystem(
        system_id=np.uint8(133),
        data_logger=data_logger,
        name="test_camera",
        output_directory=None,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=30,
        frame_width=32,
        frame_height=32,
    )

    class _WindowWideningThread(threading.Thread):
        """Holds the caller inside start() while the watchdog thread it started polls, widening the window in which
        the watchdog observes a system whose startup has not yet completed.
        """

        def start(self) -> None:
            """Starts the thread and records the state the watchdog observed before the startup completed."""
            super().start()
            producer_died = _wait_until(predicate=lambda: not video_system._producer_process.is_alive())

            # Leaves the watchdog several 20-millisecond poll cycles to act on the dead producer it monitors.
            PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=200, allow_sleep=True, block=False)
            observations.append((producer_died, video_system._terminator_array is not None))

    monkeypatch.setattr(video_system_module, "Thread", _WindowWideningThread)

    thread_errors: list = []
    original_excepthook = threading.excepthook
    threading.excepthook = thread_errors.append
    try:
        video_system.start()
        assert _wait_until(predicate=lambda: len(thread_errors) > 0, timeout_seconds=30)
    finally:
        threading.excepthook = original_excepthook
        video_system.stop()

    # The watchdog polled repeatedly while the started flag was unset and left the buffer alone, even though the
    # producer it monitors had already died. Acting there would have destroyed the buffer the startup was still
    # wiring up and left the instance reporting a recording it never began.
    assert observations == [(True, True)]

    # Once the startup completed, the same dead producer was reported and every resource was reclaimed.
    assert thread_errors[0].exc_type is RuntimeError
    assert not video_system.started
    assert video_system._terminator_array is None


def test_display_frame_rate_validation(data_logger, tmp_path) -> None:
    """Verifies the validation of the display_frame_rate parameter."""
    output_directory = tmp_path / "test_output_directory"

    # Tests invalid display frame rate (string instead of int)
    invalid_display_rate = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Encountered "
        f"an unsupported 'display_frame_rate' argument value {invalid_display_rate} of type "
        f"{type(invalid_display_rate).__name__}. The display frame rate override has to be None or a positive "
        f"integer that does not exceed the camera acquisition frame rate (30)."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            frame_rate=30,
            display_frame_rate=invalid_display_rate,  # type: ignore[arg-type]
        )

    # Tests display frame rate exceeding acquisition rate
    excessive_display_rate = 60
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Encountered "
        f"an unsupported 'display_frame_rate' argument value {excessive_display_rate} of type "
        f"{type(excessive_display_rate).__name__}. The display frame rate override has to be None or a positive "
        f"integer that does not exceed the camera acquisition frame rate (30)."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            name="test_camera",
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            frame_rate=30,
            display_frame_rate=excessive_display_rate,  # Exceeds frame_rate
        )


def test_init_disables_frame_display_on_macos(data_logger, monkeypatch) -> None:
    """Verifies that macOS degrades a requested frame display to disabled instead of rejecting the request."""
    # The constructor reads sys.platform at call time, which makes the macOS branch reachable from any host. Every
    # other platform read in the package resolves at import time and is therefore unaffected by this override.
    monkeypatch.setattr(sys, "platform", "darwin")

    # The output directory is left unset, so the degradation is exercised without requiring FFMPEG on the host.
    with pytest.warns(UserWarning, match="Displaying frames is currently not supported") as warning_records:
        video_system = VideoSystem(
            system_id=np.uint8(120),
            data_logger=data_logger,
            name="test_camera",
            output_directory=None,
            camera_interface=CameraInterfaces.MOCK,
            frame_rate=30,
            display_frame_rate=30,
        )

    # Construction succeeds and the requested rate is discarded, in contrast to the rates that
    # test_display_frame_rate_validation has rejected outright.
    assert video_system._display_frame_rate == 0

    # The warning names the system that lost its display, which is what tells a user whose camera was disabled.
    assert len(warning_records) == 1
    assert "Disabling frame display for the VideoSystem with id 120." in str(warning_records[0].message)


@pytest.mark.xdist_group(name="group1")
def test_init_opencv_interface(has_opencv, data_logger, tmp_path) -> None:
    """Verifies that VideoSystem can be initialized with the OpenCV camera interface."""
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")
    if not check_ffmpeg_availability():
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_directory = tmp_path / "opencv_test"
    vs = VideoSystem(
        system_id=np.uint8(50),
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.OPENCV,
        camera_index=0,
    )
    assert vs._saver is not None


@pytest.mark.usefixtures("gentl_simulator")
def test_init_harvesters_interface(data_logger, tmp_path) -> None:
    """Verifies that VideoSystem can be initialized with the Harvesters camera interface."""
    if not check_ffmpeg_availability():
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_directory = tmp_path / "harvesters_test"
    # The simulated devices report no acquisition rate of their own, so the encoder rate is supplied explicitly.
    vs = VideoSystem(
        system_id=np.uint8(51),
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.HARVESTERS,
        camera_index=0,
        frame_rate=10,
        frame_width=200,
        frame_height=200,
    )
    assert vs._saver is not None
    assert vs._camera.frame_rate == 10
    assert vs._camera.frame_width == 200
    assert vs._camera.frame_height == 200


@pytest.mark.xdist_group(name="group2")
def test_init_harvesters_interface_hardware(has_harvesters, data_logger, tmp_path) -> None:
    """Verifies that VideoSystem can be initialized against real GenICam hardware."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GenICam camera).")
    if not check_ffmpeg_availability():
        pytest.skip("Skipping this test as it requires FFMPEG.")

    output_directory = tmp_path / "harvesters_hardware_test"
    vs = VideoSystem(
        system_id=np.uint8(52),
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.HARVESTERS,
        camera_index=0,
    )
    assert vs._saver is not None


def test_extract_logged_camera_timestamps_errors(tmp_path) -> None:
    """Verifies the error handling of the extract_logged_camera_timestamps() function."""
    # Tests with a non-existent file
    non_existent_path = tmp_path / "non_existent.npz"
    message = (
        f"Unable to extract camera frame timestamp data from the log file {non_existent_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=non_existent_path)

    # Tests with invalid file extension
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.touch()
    message = (
        f"Unable to extract camera frame timestamp data from the log file {invalid_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(log_path=invalid_path)


def test_camera_timestamp_extraction(data_logger, tmp_path) -> None:
    """Verifies timestamp extraction with start/stop segments of frame saving.

    Ensures that timestamps are correctly extracted when frame saving is enabled and disabled
    multiple times during a single session.
    """
    system_id = np.uint8(99)
    frame_rate = 10  # Lower frame rate for easier validation

    output_directory = tmp_path / "test_segmented_timestamps"
    output_directory.mkdir(parents=True, exist_ok=True)

    video_system = VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=frame_rate,
        frame_width=320,
        frame_height=240,
        color=True,
    )

    # Starts the systems.
    data_logger.start()
    video_system.start()

    timer = PrecisionTimer(precision="s")

    # First segment: 1 second of recording
    video_system.start_frame_saving()
    timer.delay(delay=1, allow_sleep=True, block=False)
    video_system.stop_frame_saving()

    # Pause: 1 second without recording
    timer.delay(delay=1, allow_sleep=True, block=False)

    # Second segment: 2 seconds of recording
    video_system.start_frame_saving()
    timer.delay(delay=2, allow_sleep=True, block=False)
    video_system.stop_frame_saving()

    # Pause: 1 second without recording
    timer.delay(delay=1, allow_sleep=True, block=False)

    # Third segment: 1 second of recording
    video_system.start_frame_saving()
    timer.delay(delay=1, allow_sleep=True, block=False)
    video_system.stop_frame_saving()

    # Stops the systems.
    video_system.stop()
    data_logger.stop()

    # Processes the logs.
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, memory_mapping=False)

    # Extracts timestamps
    log_file_path = data_logger.output_directory / f"{system_id}_log.npz"
    timestamps = extract_logged_camera_timestamps(log_path=log_file_path, workers=1)

    # Total recording time: 1 + 2 + 1 = 4 seconds
    # Expected frames: approximately 40 (4 * 10 fps)
    actual_frames = len(timestamps)

    # Allows for timing variations.
    assert actual_frames >= 30, f"Expected approximately 40 frames, got {actual_frames}"

    # Checks for gaps in the timestamps that may indicate the recording pauses.
    # Uses a permissive threshold because the actual gap sizes depend on implementation details.
    if len(timestamps) > 10:
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        max_interval = max(intervals)
        average_interval = np.mean(intervals)

        # The maximum interval might be larger due to pauses, but shouldn't be
        # excessive (e.g., not more than 10x the average for this controlled test)
        assert max_interval < average_interval * 10, "Detected unexpectedly large gap in timestamps"


def test_extract_logged_camera_timestamps_empty_archive(tmp_path) -> None:
    """Verifies that extract_logged_camera_timestamps returns an empty array for archives with no messages."""
    # Builds an archive holding the onset message alone, which is what a recording that acquired no frame produces.
    archive_path = tmp_path / f"101{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(archive_path=archive_path, source_id=101, onset_us=1700000000000000, frame_timestamps_us=[])

    timestamps = extract_logged_camera_timestamps(log_path=archive_path, workers=1)

    assert timestamps.size == 0


def test_extract_logged_camera_timestamps_parallel(data_logger, tmp_path) -> None:
    """Verifies that extract_logged_camera_timestamps works correctly with parallel processing."""
    system_id = np.uint8(77)
    frame_rate = 30

    output_directory = tmp_path / "parallel_timestamps"
    output_directory.mkdir(parents=True, exist_ok=True)

    video_system = VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        name="test_camera",
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=frame_rate,
        frame_width=100,
        frame_height=100,
        color=True,
    )

    # Runs the system long enough to generate > 2000 frame messages for parallel processing.
    data_logger.start()
    video_system.start()

    timer = PrecisionTimer(precision="s")
    video_system.start_frame_saving()
    timer.delay(delay=75, allow_sleep=True, block=False)
    video_system.stop_frame_saving()

    video_system.stop()
    data_logger.stop()

    # Assembles log archives.
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, memory_mapping=False)

    # Extracts timestamps using parallel workers.
    log_file_path = data_logger.output_directory / f"{system_id}_log.npz"
    if log_file_path.exists():
        timestamps_parallel = extract_logged_camera_timestamps(log_path=log_file_path, workers=-1)
        timestamps_sequential = extract_logged_camera_timestamps(log_path=log_file_path, workers=1)

        # Both methods should return the same timestamps.
        assert len(timestamps_parallel) == len(timestamps_sequential)
        np.testing.assert_array_equal(timestamps_parallel, timestamps_sequential)
        assert len(timestamps_parallel) > 2000
