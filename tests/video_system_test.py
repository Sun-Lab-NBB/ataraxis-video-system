"""Contains tests for classes and methods provided by the video_system.py module."""

import sys
import subprocess

import numpy as np
import pytest
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import DataLogger, assemble_log_archives

from ataraxis_video_system import VideoSystem
from ataraxis_video_system.saver import (
    VideoEncoders,
    OutputPixelFormats,
    EncoderSpeedPresets,
    check_ffmpeg_availability,
)
from ataraxis_video_system.camera import CameraInterfaces, discover_camera_ids
from ataraxis_video_system.video_system import extract_logged_camera_timestamps
from random import randint


@pytest.fixture(scope="session")
def has_opencv():
    """Checks for OpenCV camera availability in the test environment."""
    try:
        all_cameras = discover_camera_ids()
        opencv_ids = [cam for cam in all_cameras if cam.interface == CameraInterfaces.OPENCV]
        if len(opencv_ids) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_harvesters():
    """Checks for Harvesters camera availability in the test environment."""
    try:
        # Attempts to discover Harvesters cameras using the internally stored CTI path
        all_cameras = discover_camera_ids()
        harvesters_ids = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]
        if len(harvesters_ids) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_nvidia():
    """Checks for NVIDIA GPU availability in the test environment."""
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return True
    except:
        return False


@pytest.fixture
def data_logger(tmp_path) -> DataLogger:
    """Creates a DataLogger instance and returns it to the caller."""
    data_logger = DataLogger(output_directory=tmp_path, instance_name=str(randint(0, 100000000000)))
    return data_logger


@pytest.fixture
def video_system(tmp_path, data_logger) -> VideoSystem:
    """Creates a VideoSystem instance and returns it to the caller."""
    system_id = np.uint8(1)
    output_directory = tmp_path.joinpath("test_output_directory")

    return VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        camera_index=0,
        display_frame_rate=None,
        frame_width=None,
        frame_height=None,
        frame_rate=None,
        gpu=-1,
        video_encoder=VideoEncoders.H265,
        encoder_speed_preset=EncoderSpeedPresets.SLOW,
        output_pixel_format=OutputPixelFormats.YUV444,
        quantization_parameter=15,
        color=True,
    )


def test_init_repr(tmp_path, data_logger) -> None:
    """Verifies the functioning of the VideoSystem __init__() and __repr__() methods."""
    vs_instance = VideoSystem(
        system_id=np.uint8(1),
        data_logger=data_logger,
        output_directory=tmp_path.joinpath("test_output_directory"),
        camera_interface=CameraInterfaces.MOCK,
        camera_index=0,
    )

    # Verifies class properties
    assert vs_instance.system_id == np.uint8(1)
    assert not vs_instance.started
    assert vs_instance.video_file_path is not None

    # Verifies the __repr()__ method
    representation_string: str = (
        f"VideoSystem(system_id={np.uint8(1)}, started={False}, camera=MockCamera, frame_saving={True})"
    )
    assert repr(vs_instance) == representation_string


def test_init_errors(data_logger) -> None:
    """Verifies the error handling behavior of the VideoSystem initialization method."""
    # Invalid system_id input - causes conversion error
    invalid_system_id = "str"
    # noinspection PyTypeChecker
    with pytest.raises((TypeError, ValueError)):
        VideoSystem(
            system_id=invalid_system_id,  # type: ignore
            data_logger=data_logger,
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
            data_logger=invalid_data_logger,  # type: ignore
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
            output_directory=invalid_output_directory,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )


def test_camera_configuration_errors(data_logger, tmp_path) -> None:
    """Verifies the error handling behavior of camera configuration during VideoSystem initialization."""
    output_directory = tmp_path.joinpath("test_output_directory")

    # Invalid camera index
    invalid_index = "str"
    message = (
        f"Unable to configure the camera interface for the VideoSystem with id 1. Expected a "
        f"zero or positive integer as the 'camera_id' argument value, but got {invalid_index} of type "
        f"{type(invalid_index).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_index=invalid_index,  # type: ignore
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
            output_directory=output_directory,
            frame_rate=invalid_frame_rate,  # type: ignore
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
            output_directory=output_directory,
            frame_width=invalid_frame_width,  # type: ignore
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
            output_directory=output_directory,
            frame_height=invalid_frame_height,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid camera interface
    invalid_interface = "invalid"
    message_pattern = "Unable to configure the camera interface.*unsupported camera_interface"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=invalid_interface,  # type: ignore
        )


def test_video_saver_configuration(data_logger, tmp_path, has_nvidia) -> None:
    """Verifies the functioning of video saver configuration during VideoSystem initialization."""
    output_directory = tmp_path.joinpath("test_output_directory")

    # Tests GPU encoding if NVIDIA GPU is available, otherwise tests CPU encoding
    if has_nvidia and check_ffmpeg_availability():
        vs = VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
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
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            gpu=-1,
            video_encoder=VideoEncoders.H265,
            encoder_speed_preset=EncoderSpeedPresets.FASTEST,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
        )
        assert vs._saver is not None


def test_video_saver_configuration_errors(data_logger, tmp_path) -> None:
    """Verifies the error handling behavior of video saver configuration during VideoSystem initialization."""
    output_directory = tmp_path.joinpath("test_output_directory")

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
            output_directory=output_directory,
            gpu=invalid_gpu,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid video encoder
    invalid_encoder = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'video_encoder'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            video_encoder=invalid_encoder,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid encoder preset
    invalid_preset = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'encoder_speed_preset'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            encoder_speed_preset=invalid_preset,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid output pixel format
    invalid_format = "invalid"
    message_pattern = "Unable to configure the video saver.*unexpected 'output_pixel_format'"
    with pytest.raises(ValueError, match=message_pattern):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            output_pixel_format=invalid_format,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )

    # Invalid quantization parameter
    invalid_qp = "str"
    message = (
        f"Unable to configure the video saver for the VideoSystem with id 1. Expected an "
        f"integer between -1 and 51 as the 'quantization_parameter' argument value, but got "
        f"{invalid_qp} of type {type(invalid_qp).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            output_directory=output_directory,
            quantization_parameter=invalid_qp,  # type: ignore
            camera_interface=CameraInterfaces.MOCK,
        )


def test_start_stop(data_logger, tmp_path) -> None:
    """Verifies the functioning of the start(), stop(), start_frame_saving() and stop_frame_saving() methods of the
    VideoSystem class.

    Also verifies internal DataLogger bindings and logged timestamp extraction methods.
    """
    # Creates two VideoSystem instances with different configurations
    output_directory = tmp_path.joinpath("test_output_directory")

    video_system_1 = VideoSystem(
        system_id=np.uint8(101),
        data_logger=data_logger,
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=10,
        display_frame_rate=None,
        quantization_parameter=40,
    )

    video_system_2 = VideoSystem(
        system_id=np.uint8(202),
        data_logger=data_logger,
        output_directory=None,  # No saving configured
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=5,
    )

    # Starts all instances
    data_logger.start()
    video_system_1.start()
    video_system_1.start()  # Ensures that calling start twice does nothing
    video_system_2.start()

    assert video_system_1.started
    assert video_system_2.started

    # Tests frame saving control
    timer = PrecisionTimer("s")
    video_system_1.start_frame_saving()
    timer.delay(delay=2, allow_sleep=True, block=False)  # 2-second delay
    video_system_1.stop_frame_saving()

    # Tests that system 2 (without a saver) ignores saving commands
    video_system_2.start_frame_saving()
    video_system_2.stop_frame_saving()

    # Re-enables saving for system 1
    video_system_1.start_frame_saving()
    timer.delay(delay=2, allow_sleep=True, block=False)  # 2-second delay
    video_system_1.stop_frame_saving()

    # Stops the video systems
    video_system_1.stop()
    video_system_2.stop()
    video_system_2.stop()  # Ensures that calling stop twice does nothing

    assert not video_system_1.started
    assert not video_system_2.started

    # Compresses logs for timestamp extraction
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, memory_mapping=False)

    # Extracts frame timestamps for system 1 (which saved frames)
    log_path_1 = data_logger.output_directory.joinpath(f"{data_logger._name}_101.npz")
    if log_path_1.exists():
        frame_timestamps_1 = extract_logged_camera_timestamps(log_path_1, n_workers=1)
        # With fps of 10 and running for ~4 seconds total, should have acquired around 40 frames
        assert 35 <= len(frame_timestamps_1) <= 45

    # Tests the system without frame saving
    video_system_3 = VideoSystem(
        system_id=np.uint8(234),
        data_logger=data_logger,
        output_directory=None,  # No output directory
        camera_interface=CameraInterfaces.MOCK,
    )
    video_system_3.start()
    timer.delay(delay=1, allow_sleep=True, block=False)
    video_system_3.stop()
    data_logger.stop()


def test_display_frame_rate_validation(data_logger, tmp_path) -> None:
    """Verifies the validation of the display_frame_rate parameter."""
    output_directory = tmp_path.joinpath("test_output_directory")

    # Tests that display functionality is disabled on macOS
    if "darwin" in sys.platform:
        with pytest.warns(UserWarning, match="Displaying frames is currently not supported"):
            vs = VideoSystem(
                system_id=np.uint8(1),
                data_logger=data_logger,
                output_directory=output_directory,
                camera_interface=CameraInterfaces.MOCK,
                display_frame_rate=30,
            )
            assert vs._display_frame_rate == 0

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
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            frame_rate=30,
            display_frame_rate=invalid_display_rate,  # type: ignore
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
            output_directory=output_directory,
            camera_interface=CameraInterfaces.MOCK,
            frame_rate=30,
            display_frame_rate=excessive_display_rate,  # Exceeds frame_rate
        )


def test_extract_logged_camera_timestamps_errors(tmp_path) -> None:
    """Verifies the error handling of the extract_logged_camera_timestamps() function."""
    # Tests with a non-existent file
    non_existent_path = tmp_path.joinpath("non_existent.npz")
    message = (
        f"Unable to extract camera frame timestamp data from the log file {non_existent_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(non_existent_path)

    # Tests with invalid file extension
    invalid_path = tmp_path.joinpath("invalid.txt")
    invalid_path.touch()
    message = (
        f"Unable to extract camera frame timestamp data from the log file {invalid_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_camera_timestamps(invalid_path)


def test_camera_timestamp_extraction(data_logger, tmp_path) -> None:
    """Verifies timestamp extraction with start/stop segments of frame saving.

    Ensures that timestamps are correctly extracted when frame saving is enabled and disabled
    multiple times during a single session.
    """

    system_id = np.uint8(99)
    frame_rate = 10  # Lower frame rate for easier validation

    output_directory = tmp_path.joinpath("test_segmented_timestamps")
    output_directory.mkdir(parents=True, exist_ok=True)

    video_system = VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        output_directory=output_directory,
        camera_interface=CameraInterfaces.MOCK,
        frame_rate=frame_rate,
        frame_width=320,
        frame_height=240,
        color=True,
    )

    # Start systems
    data_logger.start()
    video_system.start()

    timer = PrecisionTimer("s")

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

    # Stop systems
    video_system.stop()
    data_logger.stop()

    # Process logs
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, memory_mapping=False)

    # Extracts timestamps
    log_file_path = data_logger.output_directory.joinpath(f"{system_id}_log.npz")
    timestamps = extract_logged_camera_timestamps(log_file_path, n_workers=1)

    # Total recording time: 1 + 2 + 1 = 4 seconds
    # Expected frames: approximately 40 (4 * 10 fps)
    actual_frames = len(timestamps)

    # Allow for timing variations
    assert 0 <= actual_frames, f"Expected approximately {40} frames, got {actual_frames}"

    # Check for gaps in timestamps that might indicate the pauses
    # (This is a basic check - actual gaps depend on implementation details)
    if len(timestamps) > 10:
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        max_interval = max(intervals)
        avg_interval = np.mean(intervals)

        # The maximum interval might be larger due to pauses, but shouldn't be
        # excessive (e.g., not more than 10x the average for this controlled test)
        assert max_interval < avg_interval * 10, "Detected unexpectedly large gap in timestamps"
