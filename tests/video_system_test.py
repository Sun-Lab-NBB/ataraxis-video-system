"""Contains tests for classes and methods provided by the video_system.py module."""

from copy import copy
from pathlib import Path
import subprocess
import multiprocessing

import numpy as np
import pytest
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import DataLogger

from ataraxis_video_system import VideoSystem
from ataraxis_video_system.saver import (
    VideoCodecs,
    ImageFormats,
    VideoFormats,
    CPUEncoderPresets,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from ataraxis_video_system.camera import OpenCVCamera, CameraBackends
from hyperframe import frame


def check_opencv() -> bool:
    """Returns True if the system has access to an OpenCV-compatible camera.

    This is used to disable saver tests that rely on the presence of an OpenCV-compatible camera when no valid hardware
    is available.
    """
    # noinspection PyBroadException
    try:
        opencv_id = VideoSystem.get_opencv_ids()
        assert len(opencv_id) > 0  # If no IDs are discovered, this means there are no OpenCV cameras.
        return True
    except:
        return False


def check_harvesters(cti_path: Path) -> bool:
    """Returns True if the system has access to a Harvesters (GeniCam-compatible) camera.

    Args:
        cti_path: The path to the CTI file to use for GeniCam camera connection.

    This is used to disable saver tests that rely on the presence of a Harvesters-compatible camera when no valid
    hardware is available.
    """

    # If the CTI file does not exist, it is impossible to interface with Harvesters cameras, even if they exist.
    if not cti_path.exists():
        return False

    try:
        harvesters_id = VideoSystem.get_harvesters_ids(cti_path)
        assert len(harvesters_id) > 0  # If no IDs are discovered, this means there are no Harvesters cameras.
        return True
    except:
        return False


def check_nvidia():
    """Returns True if the system has access to an NVIDIA GPU.

    This is used to disable saver tests that rely on the presence of an NVIDIA GPU to use hardware encoding.
    """
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        return True
    # If the process fails, this likely means nvidia-smi is not available and, therefore, the system does not have
    # access to an NVIDIA GPU.
    except Exception:
        return False


@pytest.fixture()
def data_logger(tmp_path) -> DataLogger:
    """Generates a DataLogger class instance and returns it to caller."""
    data_logger = DataLogger(output_directory=tmp_path)
    return data_logger


@pytest.fixture
def video_system(tmp_path, data_logger) -> VideoSystem:
    """Creates a VideoSystem instance and returns it to caller."""

    system_id = np.uint8(1)
    output_directory = tmp_path.joinpath("test_output_directory")

    # If the local harvesters CTI file does not exist, sets the path to None to prevent initialization errors due to
    # a missing CTI path
    harvesters_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    if not check_harvesters(harvesters_cti_path):
        harvesters_cti_path = None

    return VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )


def test_init_repr(tmp_path, data_logger):
    """Verifies the functioning of the VideoSystem __init__() and __repr__() methods."""
    vs_instance = VideoSystem(
        system_id=np.uint8(1),
        data_logger=data_logger,
        output_directory=tmp_path.joinpath("test_output_directory"),
        harvesters_cti_path=None,
    )

    # Verifies class properties
    assert vs_instance.system_id == np.uint8(1)
    assert not vs_instance.started

    # Verifies the __repr()__ method
    representation_string: str = f"VideoSystem(id={np.uint8(1)}, started={False}, camera_count={0}, saver_count={0})"
    assert repr(vs_instance) == representation_string


def test_init_errors(data_logger):
    """Verifies the error handling behavior of the VideoSystem initialization method."""

    # Invalid ID input
    invalid_system_id = "str"
    message = (
        f"Unable to initialize the VideoSystem class instance. Expected a uint8 integer as system_id argument, "
        f"but encountered {invalid_system_id} of type {type(invalid_system_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        VideoSystem(
            system_id=invalid_system_id,
            data_logger=data_logger,
        )

    # Invalid CTI extension
    invalid_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.zip")
    message = (
        f"Unable to initialize the VideoSystem instance with id {np.uint8(1)}. Expected the path to an existing "
        f"file with a '.cti' suffix or None for harvesters_cti_path, but encountered {invalid_cti_path} of "
        f"type {type(invalid_cti_path).__name__}. If a valid path was provided, this error is likely due to "
        f"the file not existing or not being accessible."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            harvesters_cti_path=invalid_cti_path,
        )

    # Non-existent CTI file
    invalid_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducerNoExist")
    message = (
        f"Unable to initialize the VideoSystem instance with id {np.uint8(1)}. Expected the path to an existing "
        f"file with a '.cti' suffix or None for harvesters_cti_path, but encountered {invalid_cti_path} of "
        f"type {type(invalid_cti_path).__name__}. If a valid path was provided, this error is likely due to "
        f"the file not existing or not being accessible."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=data_logger,
            harvesters_cti_path=invalid_cti_path,
        )

    # Invalid data_logger input
    invalid_data_logger = None
    message = (
        f"Unable to initialize the VideoSystem instance with id {np.uint8(1)}. Expected an initialized DataLogger "
        f"instance for 'data_logger' argument, but encountered {invalid_data_logger} of type "
        f"{type(invalid_data_logger).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        VideoSystem(
            system_id=np.uint8(1),
            data_logger=invalid_data_logger,
        )

    # Invalid output_directory input
    invalid_output_directory = str("Not a Path")
    message = (
        f"Unable to initialize the VideoSystem instance with id {np.uint8(1)}. Expected a Path instance or None "
        f"for 'output_directory' argument, but encountered {invalid_output_directory} of type "
        f"{type(invalid_output_directory).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        VideoSystem(system_id=np.uint8(1), data_logger=data_logger, output_directory=invalid_output_directory)


@pytest.mark.xdist_group(name="group2")
@pytest.mark.parametrize(
    "backend",
    [
        CameraBackends.MOCK,
        CameraBackends.OPENCV,
        CameraBackends.HARVESTERS,
    ],
)
def test_add_camera(backend, video_system):
    """Verifies the functioning of the VideoSystem add_camera() method for all supported camera backends."""
    if backend == CameraBackends.OPENCV and not check_opencv():
        pytest.skip(f"Skipping this test as it requires an OpenCV-compatible camera.")

    if backend == CameraBackends.HARVESTERS and video_system._cti_path is None:
        pytest.skip(f"Skipping this test as it requires a harvesters-compatible camera.")

    # Adds the tested camera to the VideoSystem instance
    video_system.add_camera(camera_id=np.uint8(222), save_frames=True, color=False, camera_backend=backend)

    # Regardless of the first camera type, adds the second camera to simulate multi-camera setup
    video_system.add_camera(camera_id=np.uint8(111), save_frames=False, color=True, camera_backend=CameraBackends.MOCK)

    # Verifies that the cameras have been added to the VideoSystem instance
    assert len(video_system._cameras) == 2


def test_add_camera_errors(video_system):
    """Verifies the error handling behavior of the VideoSystem add_camera() method.

    Note, this function does not verify invalid OpenCV camera configuration. These errors are tested via a separate
    function
    """

    # Defines arguments that are reused by all test calls
    camera_id = np.uint8(1)
    save_frames = True

    # Verifies general function arguments. They are used for all camera backends
    # Invalid Camera ID
    invalid_id = 1  # Not an uint8
    message = (
        f"Unable to add the Camera object to the VideoSystem with id {video_system._id}. Expected a numpy uint8 for "
        f"camera_id argument, but got {invalid_id} of type {type(invalid_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(camera_id=invalid_id, save_frames=save_frames)

    # Invalid camera index
    invalid_index = "str"
    message = (
        f"Unable to add the Camera with id {camera_id} to the VideoSystem with id {video_system._id}. Expected an "
        f"integer for camera_id argument, but got {invalid_index} of type {type(invalid_index).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(camera_id=camera_id, save_frames=save_frames, camera_index=invalid_index)

    # Invalid fps rate
    invalid_acquisition_frame_rate = "str"
    message = (
        f"Unable to add the Camera with id {camera_id} to the VideoSystem with id {video_system._id}. Expected an "
        f"integer, float or None for acquisition_frame_rate argument, but got {invalid_acquisition_frame_rate} of type "
        f"{type(invalid_acquisition_frame_rate).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            camera_id=camera_id,
            save_frames=save_frames,
            acquisition_frame_rate=invalid_acquisition_frame_rate,
        )

    # Invalid frame width
    invalid_frame_width = "str"
    message = (
        f"Unable to add the Camera with id {camera_id} to the VideoSystem with id {video_system._id}. Expected an "
        f"integer or None for frame_width argument, but got {invalid_frame_width} of type "
        f"{type(invalid_frame_width).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            camera_id=camera_id,
            save_frames=save_frames,
            frame_width=invalid_frame_width,
        )

    # Invalid frame height
    invalid_frame_height = "str"
    message = (
        f"Unable to add the Camera with id {camera_id} to the VideoSystem with id {video_system._id}. Expected an "
        f"integer or None for frame_height argument, but got {invalid_frame_height} of type "
        f"{type(invalid_frame_height).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            camera_id=camera_id,
            save_frames=save_frames,
            frame_height=invalid_frame_height,
        )

    # Invalid OpenCV backend code for OpenCV camera
    opencv_backend = "2.0"
    message = (
        f"Unable to add the OpenCVCamera with id {camera_id} to the VideoSystem with id {video_system._id}. "
        f"Expected an integer or None for opencv_backend argument, but got {opencv_backend} of type "
        f"{type(opencv_backend).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            camera_id=camera_id,
            save_frames=save_frames,
            camera_backend=CameraBackends.OPENCV,
            opencv_backend=opencv_backend,
        )

    # Harvesters CTI path set to None for Harvesters camera.
    message = (
        f"Unable to add HarvestersCamera with id {camera_id} to the VideoSystem with id {video_system._id}. "
        f"Expected the VideoSystem's cti_path attribute to be a Path object pointing to the '.cti' file, "
        f"but got None instead. Make sure you provide a valid '.cti' file as harvesters_cit_file argument "
        f"when initializing the VideoSystem instance."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # Resets the CTI path to simulate a scenario where it's not provided
        original_cti_path = copy(video_system._cti_path)
        video_system._cti_path = None
        video_system.add_camera(
            camera_id=camera_id,
            save_frames=save_frames,
            camera_backend=CameraBackends.HARVESTERS,
        )
        video_system._cti_path = original_cti_path  # Restores the CTI path


@pytest.mark.xdist_group(name="group2")
def test_opencvcamera_configuration_errors(video_system):
    """Verifies that add_camera() method correctly catches errors related to OpenCV camera configuration."""

    # Skips the test if OpenCV-compatible hardware is not available.
    if not check_opencv():
        pytest.skip(f"Skipping this test as it requires an OpenCV-compatible camera.")

    # Presets parameters that will be used by all errors
    camera_backend = CameraBackends.OPENCV
    camera_id = np.uint8(111)
    save_frames = True
    camera_index = 0

    # Connects to the camera manually to get the 'default' frame dimensions and framerate
    camera = OpenCVCamera(camera_id=np.uint8(111))
    camera.connect()
    actual_width = camera.width
    actual_height = camera.height
    actual_fps = camera.fps
    camera.disconnect()

    # Unsupported frame height
    frame_height = 3000
    message = (
        f"Unable to add the OpenCVCamera with id {camera_id} to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames using the provided frame_height "
        f"{frame_height}, but the camera returned a test frame with height {actual_height}. This "
        f"indicates that the camera does not support the requested frame height and width combination."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_id=camera_id,
            camera_index=camera_index,
            save_frames=save_frames,
            frame_height=frame_height,
            frame_width=actual_width,
            camera_backend=camera_backend,
        )

    # Unsupported frame width
    frame_width = 3000
    message = (
        f"Unable to add the OpenCVCamera with id {camera_id} to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames using the provided frame_width {frame_width}, "
        f"but the camera returned a test frame with width {actual_width}. This indicates that the camera "
        f"does not support the requested frame height and width combination."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_id=camera_id,
            camera_index=camera_index,
            save_frames=save_frames,
            frame_height=actual_height,
            frame_width=frame_width,
            camera_backend=camera_backend,
        )

    # Unsupported fps
    fps = 3000.0
    message = (
        f"Unable to add the OpenCVCamera with id {camera_id} to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames at the rate of {fps} frames per second, but the camera "
        f"automatically adjusted the framerate to {actual_fps}. This indicates that the camera does not support the "
        f"requested framerate."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_id=camera_id,
            camera_index=camera_index,
            save_frames=save_frames,
            acquisition_frame_rate=fps,
            camera_backend=camera_backend,
        )

    # Since our camera can do both color and monochrome imaging, we cannot test failure to assign colored or monochrome
    # imaging mode here.


def test_add_image_saver(video_system):
    """Verifies the functioning of the VideoSystem add_image_saver() method."""

    # Adds an image saver instance to the VideoSystem instance
    video_system.add_image_saver(
        source_id=np.uint8(222), image_format=ImageFormats.PNG, png_compression=9, thread_count=15
    )

    # Also adds a second saver to simulate multi-saver setup
    video_system.add_image_saver(source_id=np.uint8(111), image_format=ImageFormats.JPG, jpeg_quality=90)

    # Verifies that the savers have been added to the VideoSystem instance
    assert len(video_system._savers) == 2


def test_add_image_saver_errors(video_system):
    """Verifies the error handling behavior of the VideoSystem add_image_saver() method."""

    # Defines arguments that are reused by multiple tests
    source_id = np.uint8(111)

    # Invalid source id
    invalid_source_id = "str"
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected a numpy uint8 "
        f"integer for source_id argument, but got {invalid_source_id} of type {type(invalid_source_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=invalid_source_id)

    # Invalid output path
    # Resets the output path to None
    original_output_directory = copy(video_system._output_directory)
    video_system._output_directory = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected a valid Path "
        f"object to be provided to the VideoSystem's output_directory argument at initialization, but instead "
        f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
        f"you intend to save camera frames."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id)
    video_system._output_directory = original_output_directory  # Restores the original output directory

    # Invalid image format
    image_format = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an ImageFormats "
        f"instance for image_format argument, but got {image_format} of type {type(image_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, image_format=image_format)

    # Invalid tiff compression strategy
    tiff_compression_strategy = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer for "
        f"tiff_compression_strategy argument, but got {tiff_compression_strategy} of type "
        f"{type(tiff_compression_strategy).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, tiff_compression_strategy=tiff_compression_strategy)

    # Invalid jpeg quality
    jpeg_quality = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 100 for jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, jpeg_quality=jpeg_quality)

    # Invalid jpeg sampling factor
    jpeg_sampling_factor = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected one of the "
        f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_*' constants for jpeg_sampling_factor argument, but got "
        f"{jpeg_sampling_factor} of type {type(jpeg_sampling_factor).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, jpeg_sampling_factor=jpeg_sampling_factor)

    # Invalid png compression
    png_compression = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 9 for png_compression argument, but got {png_compression} of type "
        f"{type(png_compression).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, png_compression=png_compression)

    # Invalid thread count
    thread_count = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"greater than 0 for thread_count argument, but got {thread_count} of type {type(thread_count).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(source_id=source_id, thread_count=thread_count)

    # Verifies that using the same source_id more than once produces an error
    video_system.add_image_saver(source_id=source_id)
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. The camera with index "
        f"{source_id} is already matched with a saver class instance. Currently, each saver instance has to "
        f"use a single unique camera source."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_image_saver(source_id=source_id)


def test_add_video_saver(video_system):
    """Verifies the functioning of the VideoSystem add_video_saver() method."""

    # Adds a video saver instance to the VideoSystem instance. If the system has an NVIDIA gpu, the first saver is a
    # GPU saver. Otherwise, both savers are CPU savers.
    if check_nvidia():
        video_system.add_video_saver(
            source_id=np.uint8(222),
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.FASTEST,
            input_pixel_format=InputPixelFormats.BGR,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
            gpu=0,
        )
    else:
        video_system.add_video_saver(
            source_id=np.uint8(222),
            hardware_encoding=False,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=CPUEncoderPresets.ULTRAFAST,
            input_pixel_format=InputPixelFormats.BGR,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
        )

    # Also adds a second saver to simulate multi-saver setup. This is always a CPU saver.
    video_system.add_video_saver(
        source_id=np.uint8(111),
        hardware_encoding=False,
        video_format=VideoFormats.MKV,
        video_codec=VideoCodecs.H264,
        preset=CPUEncoderPresets.SLOW,
        input_pixel_format=InputPixelFormats.MONOCHROME,
        output_pixel_format=OutputPixelFormats.YUV420,
        quantization_parameter=15,
    )

    # Verifies that the savers have been added to the VideoSystem instance
    assert len(video_system._savers) == 2


def test_add_video_saver_errors(video_system):
    """Verifies the error handling behavior of the VideoSystem add_video_saver() method."""

    # Defines arguments that are reused by multiple tests
    source_id = np.uint8(111)

    # Invalid source id
    invalid_source_id = "str"
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a numpy uint8 "
        f"integer for source_id argument, but got {invalid_source_id} of type {type(invalid_source_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=invalid_source_id)

    # Invalid output path
    # Resets the output path to None
    original_output_directory = copy(video_system._output_directory)
    video_system._output_directory = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a valid Path "
        f"object to be provided to the VideoSystem's output_directory argument at initialization, but instead "
        f"encountered None. Make sure the VideoSystem is initialized with a valid output_directory input if "
        f"you intend to save camera frames."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id)
    video_system._output_directory = original_output_directory  # Restores the original output directory

    # Invalid hardware encoding flag
    hardware_encoding = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a boolean for "
        f"hardware_encoding argument, but got {hardware_encoding} of type {type(hardware_encoding).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, hardware_encoding=hardware_encoding)

    # Invalid video format
    video_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a VideoFormats "
        f"instance for video_format argument, but got {video_format} of type {type(video_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, video_format=video_format)

    # Invalid video codec
    video_codec = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a VideoCodecs "
        f"instance for video_codec argument, but got {video_codec} of type {type(video_codec).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, video_codec=video_codec)

    # Invalid encoder preset (for both hardware encoding flag states).
    preset = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a "
        f"GPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, hardware_encoding=True, preset=preset)
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a "
        f"CPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, hardware_encoding=False, preset=preset)

    # Invalid input pixel format
    input_pixel_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an "
        f"InputPixelFormats instance for input_pixel_format argument, but got {input_pixel_format} of type "
        f"{type(input_pixel_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, input_pixel_format=input_pixel_format)

    # Invalid output pixel format
    output_pixel_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an "
        f"OutputPixelFormats instance for output_pixel_format argument, but got {output_pixel_format} of type "
        f"{type(output_pixel_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, output_pixel_format=output_pixel_format)

    # Invalid quantization_parameter
    quantization_parameter = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 51 for quantization_parameter argument, but got {quantization_parameter} of type "
        f"{type(quantization_parameter).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(source_id=source_id, quantization_parameter=quantization_parameter)

    # Verifies that using the same source_id more than once produces an error
    video_system.add_video_saver(source_id=source_id)
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. The camera with index "
        f"{source_id} is already matched with a saver class instance. Currently, each saver instance has to "
        f"use a single unique camera source."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_video_saver(source_id=source_id)


def test_start_stop(video_system):

    # While not strictly necessary, ensures that there are no leftover uncollected shared memory buffers.
    video_system.vacate_shared_memory_buffer()

    # Does not test displaying threads, as this functionality is currently broken on MacOS. We test it through the
    # live_run() script. Verifies using three cameras at the same time to achieve maximum feature coverage.

    # Saves frames and outputs them to queue
    video_system.add_camera(
        camera_id=np.uint8(101),
        save_frames=True,
        display_frames=False,
        display_frame_rate=1,
        output_frames=True,
        camera_backend=CameraBackends.MOCK,
    )
    # Instantiates a video saver for the 101 camera.
    video_system.add_video_saver(source_id=np.uint8(101), quantization_parameter=40)

    # Just saves the frames
    video_system.add_camera(
        camera_id=np.uint8(202),
        save_frames=True,
        display_frames=False,
        output_frames=False,
        camera_backend=CameraBackends.MOCK,
    )
    # Instantiates an image saver for the 202 camera.
    video_system.add_image_saver(source_id=np.uint8(202))

    # This camera neither saves nor displays frames, it is here just to load up system resources.
    video_system.add_camera(
        camera_id=np.uint8(51),
        save_frames=False,
        display_frames=False,
        output_frames=False,
        camera_backend=CameraBackends.MOCK,
    )

    # Starts cameras and savers
    video_system.start()

    # Tests frame for all cameras
    timer = PrecisionTimer("s")
    video_system.start_frame_saving()
    timer.delay_noblock(delay=2)  # 2-second delay
    video_system.stop_frame_saving()

    # The first camera is additionally configured to output frames via the output_queue. Given the output framerate of
    # 1 fps and the 2-second delay, the camera should output between 2 and 3 frames
    out_frames = []
    while not video_system.output_queue.empty():
        out_frames.append(video_system.output_queue.get())

    assert 2 >= len(out_frames) and len(out_frames) < 4

    # Frames are submitted tot he output queue as a tuple of frame data (as a numpy aray) and the ID of the camera that
    # produced the frame. Ensures only the first camera sent frames to teh output queue.
    for frame_tuple in out_frames:
        assert frame_tuple[2] == 101
