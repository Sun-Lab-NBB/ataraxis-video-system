"""Contains tests for classes and methods provided by the video_system.py module."""

import sys
from copy import copy
from pathlib import Path
import subprocess

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


@pytest.fixture(scope="session")
def cti_path():
    _cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    return _cti_path


@pytest.fixture(scope="session")
def has_opencv():
    """Static check for OpenCV camera availability."""
    try:
        opencv_id = VideoSystem.get_opencv_ids()
        if len(opencv_id) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_harvesters(cti_path):
    """Static check for Harvesters camera availability."""
    if not cti_path.exists():
        return False

    try:
        harvesters_id = VideoSystem.get_harvesters_ids(cti_path)
        if len(harvesters_id) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_nvidia():
    """Static check for NVIDIA GPU availability."""
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


@pytest.fixture()
def data_logger(tmp_path) -> DataLogger:
    """Generates a DataLogger class instance and returns it to the caller."""
    data_logger = DataLogger(output_directory=tmp_path, exist_ok=True, instance_name=str(tmp_path.stem))
    return data_logger


@pytest.fixture
def video_system(tmp_path, data_logger, has_harvesters) -> VideoSystem:
    """Creates a VideoSystem instance and returns it to the caller."""
    system_id = np.uint8(1)
    output_directory = tmp_path.joinpath("test_output_directory")

    # If the local harvesters CTI file does not exist, sets the path to None to prevent initialization errors due to
    # a missing CTI path
    harvesters_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    if not has_harvesters:
        harvesters_cti_path = None

    return VideoSystem(
        system_id=system_id,
        data_logger=data_logger,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )


def test_init_repr(tmp_path, data_logger) -> None:
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
    assert vs_instance.output_directory is None

    # Verifies the __repr()__ method
    representation_string: str = f"VideoSystem(id={np.uint8(1)}, started={False}, camera=None, saver=None)"
    assert repr(vs_instance) == representation_string


def test_init_errors(data_logger) -> None:
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
    invalid_output_directory = "Not a Path"
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
def test_add_camera(backend, video_system, has_opencv) -> None:
    """Verifies the functioning of the VideoSystem add_camera() method for all supported camera backends."""
    if backend == CameraBackends.OPENCV and not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    if backend == CameraBackends.HARVESTERS and video_system._cti_path is None:
        pytest.skip("Skipping this test as it requires a harvesters-compatible camera.")

    # Adds the tested camera to the VideoSystem instance
    video_system.add_camera(save_frames=True, color=False, camera_backend=backend)


def test_add_camera_errors(video_system) -> None:
    """Verifies the error handling behavior of the VideoSystem add_camera() method.

    Note, this function does not verify invalid OpenCV camera configuration. These errors are tested via a separate
    function
    """
    # Defines arguments that are reused by all test calls
    save_frames = True

    # Invalid camera index
    invalid_index = "str"
    message = (
        f"Unable to add a Camera to the VideoSystem with id {video_system._id}. Expected an "
        f"integer for camera_id argument, but got {invalid_index} of type {type(invalid_index).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(save_frames=save_frames, camera_index=invalid_index)

    # Invalid fps rate
    invalid_acquisition_frame_rate = "str"
    message = (
        f"Unable to add a Camera to the VideoSystem with id {video_system._id}. Expected an "
        f"integer, float or None for acquisition_frame_rate argument, but got {invalid_acquisition_frame_rate} of type "
        f"{type(invalid_acquisition_frame_rate).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            save_frames=save_frames,
            acquisition_frame_rate=invalid_acquisition_frame_rate,
        )

    # Invalid frame width
    invalid_frame_width = "str"
    message = (
        f"Unable to add a Camera to the VideoSystem with id {video_system._id}. Expected an "
        f"integer or None for frame_width argument, but got {invalid_frame_width} of type "
        f"{type(invalid_frame_width).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            save_frames=save_frames,
            frame_width=invalid_frame_width,
        )

    # Invalid frame height
    invalid_frame_height = "str"
    message = (
        f"Unable to add a Camera to the VideoSystem with id {video_system._id}. Expected an "
        f"integer or None for frame_height argument, but got {invalid_frame_height} of type "
        f"{type(invalid_frame_height).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            save_frames=save_frames,
            frame_height=invalid_frame_height,
        )

    # Invalid OpenCV backend code for OpenCV camera
    opencv_backend = "2.0"
    message = (
        f"Unable to add the OpenCVCamera to the VideoSystem with id {video_system._id}. "
        f"Expected an integer or None for opencv_backend argument, but got {opencv_backend} of type "
        f"{type(opencv_backend).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            save_frames=save_frames,
            camera_backend=CameraBackends.OPENCV,
            opencv_backend=opencv_backend,
        )

    # Harvesters CTI path set to None for Harvesters camera.
    message = (
        f"Unable to add HarvestersCamera to the VideoSystem with id {video_system._id}. "
        f"Expected the VideoSystem's cti_path attribute to be a Path object pointing to the '.cti' file, "
        f"but got None instead. Make sure you provide a valid '.cti' file as harvesters_cit_file argument "
        f"when initializing the VideoSystem instance."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # Resets the CTI path to simulate a scenario where it's not provided
        original_cti_path = copy(video_system._cti_path)
        video_system._cti_path = None
        video_system.add_camera(
            save_frames=save_frames,
            camera_backend=CameraBackends.HARVESTERS,
        )
        video_system._cti_path = original_cti_path  # Restores the CTI path

    # Invalid camera backend
    invalid_backend = None
    message = (
        f"Unable to instantiate the Camera due to encountering an unsupported "
        f"camera_backend argument {invalid_backend} of type {type(invalid_backend).__name__}. "
        f"camera_backend has to be one of the options available from the CameraBackends enumeration."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(save_frames=save_frames, camera_backend=invalid_backend)

    # Invalid output framerate override
    invalid_output_fr = None
    message = (
        f"Unable to instantiate the Camera due to encountering an unsupported "
        f"output_frame_rate argument {invalid_output_fr} of type {type(invalid_output_fr).__name__}. "
        f"Output framerate override has to be an integer or floating point number that does not exceed the "
        f"camera acquisition framerate (30.0)."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_camera(
            save_frames=save_frames,
            camera_backend=CameraBackends.MOCK,
            acquisition_frame_rate=30,
            output_frames=True,
            output_frame_rate=invalid_output_fr,
        )

    # Invalid display framerate override. This is not tested on MacOS as there is currently a static guard against
    # displaying frames on that OS.
    if "darwin" not in sys.platform:
        invalid_display_fr = None
        message = (
            f"Unable to instantiate the Camera due to encountering an unsupported "
            f"display_frame_rate argument {invalid_display_fr} of type {type(invalid_display_fr).__name__}. "
            f"Display framerate override has to be an integer or floating point number that does not exceed the "
            f"camera acquisition framerate (30.0)."
        )
        with pytest.raises(TypeError, match=error_format(message)):
            # noinspection PyTypeChecker
            video_system.add_camera(
                save_frames=save_frames,
                camera_backend=CameraBackends.MOCK,
                acquisition_frame_rate=30,
                display_frames=True,
                display_frame_rate=invalid_display_fr,
            )


@pytest.mark.xdist_group(name="group1")
def test_opencvcamera_configuration_errors(video_system, has_opencv) -> None:
    """Verifies that the add_camera () method correctly catches errors related to OpenCV camera configuration."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Presets parameters that will be used by all errors
    camera_backend = CameraBackends.OPENCV
    save_frames = True
    camera_index = 0

    # Connects to the camera manually to get the 'default' frame dimensions and framerate
    camera = OpenCVCamera(camera_id=np.uint8(111))
    camera.connect()
    actual_width = camera.width
    actual_height = camera.height
    actual_fps = camera.fps
    camera.disconnect()

    # Unsupported frame width
    frame_width = 3000
    message = (
        f"Unable to add the OpenCVCamera to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames using the provided frame_width {frame_width}, "
        f"but the camera returned a test frame with width {actual_width}. This indicates that the camera "
        f"does not support the requested frame height and width combination."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_index=camera_index,
            save_frames=save_frames,
            frame_width=frame_width,
            camera_backend=camera_backend,
        )

    # Unsupported frame height
    frame_height = 3000
    message = (
        f"Unable to add the OpenCVCamera to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames using the provided frame_height "
        f"{frame_height}, but the camera returned a test frame with height {actual_height}. This "
        f"indicates that the camera does not support the requested frame height and width combination."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_index=camera_index,
            save_frames=save_frames,
            frame_height=frame_height,
            camera_backend=camera_backend,
        )

    # Unsupported fps
    fps = 3000.0
    message = (
        f"Unable to add the OpenCVCamera to the VideoSystem with id {video_system._id}. "
        f"Attempted configuring the camera to acquire frames at the rate of {fps} frames per second, but the camera "
        f"automatically adjusted the framerate to {actual_fps}. This indicates that the camera does not support the "
        f"requested framerate."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.add_camera(
            camera_index=camera_index,
            save_frames=save_frames,
            acquisition_frame_rate=fps,
            camera_backend=camera_backend,
        )

    # Since our camera can do both color and monochrome imaging, we cannot test failure to assign colored or monochrome
    # imaging mode here.


def test_add_image_saver(video_system) -> None:
    """Verifies the functioning of the VideoSystem add_image_saver() method."""
    # Adds an image saver instance to the VideoSystem instance
    video_system.add_image_saver(image_format=ImageFormats.PNG, png_compression=9, thread_count=15)


def test_add_image_saver_errors(video_system) -> None:
    """Verifies the error handling behavior of the VideoSystem add_image_saver() method."""

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
        video_system.add_image_saver()
    video_system._output_directory = original_output_directory  # Restores the original output directory

    # Invalid image format
    image_format = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an ImageFormats "
        f"instance for image_format argument, but got {image_format} of type {type(image_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(image_format=image_format)

    # Invalid tiff compression strategy
    tiff_compression_strategy = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer for "
        f"tiff_compression_strategy argument, but got {tiff_compression_strategy} of type "
        f"{type(tiff_compression_strategy).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(tiff_compression_strategy=tiff_compression_strategy)

    # Invalid jpeg quality
    jpeg_quality = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 100 for jpeg_quality argument, but got {jpeg_quality} of type {type(jpeg_quality)}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(jpeg_quality=jpeg_quality)

    # Invalid jpeg sampling factor
    jpeg_sampling_factor = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected one of the "
        f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_*' constants for jpeg_sampling_factor argument, but got "
        f"{jpeg_sampling_factor} of type {type(jpeg_sampling_factor).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(jpeg_sampling_factor=jpeg_sampling_factor)

    # Invalid png compression
    png_compression = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 9 for png_compression argument, but got {png_compression} of type "
        f"{type(png_compression).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(png_compression=png_compression)

    # Invalid thread count
    thread_count = None
    message = (
        f"Unable to add the ImageSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"greater than 0 for thread_count argument, but got {thread_count} of type {type(thread_count).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_image_saver(thread_count=thread_count)


def test_add_video_saver(video_system, has_nvidia) -> None:
    """Verifies the functioning of the VideoSystem add_video_saver() method."""
    # Adds a video saver instance to the VideoSystem instance. If the system has an NVIDIA gpu, the first saver is a
    # GPU saver. Otherwise, it is a CPU saver
    if has_nvidia:
        video_system.add_video_saver(
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
            hardware_encoding=False,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=CPUEncoderPresets.ULTRAFAST,
            input_pixel_format=InputPixelFormats.BGR,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=5,
        )


def test_add_video_saver_errors(video_system) -> None:
    """Verifies the error handling behavior of the VideoSystem add_video_saver() method."""
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
        video_system.add_video_saver()
    video_system._output_directory = original_output_directory  # Restores the original output directory

    # Invalid hardware encoding flag
    hardware_encoding = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a boolean for "
        f"hardware_encoding argument, but got {hardware_encoding} of type {type(hardware_encoding).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(hardware_encoding=hardware_encoding)

    # Invalid video format
    video_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a VideoFormats "
        f"instance for video_format argument, but got {video_format} of type {type(video_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(video_format=video_format)

    # Invalid video codec
    video_codec = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a VideoCodecs "
        f"instance for video_codec argument, but got {video_codec} of type {type(video_codec).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(video_codec=video_codec)

    # Invalid encoder preset (for both hardware encoding flag states).
    preset = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a "
        f"GPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(hardware_encoding=True, preset=preset)
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected a "
        f"CPUEncoderPresets instance for preset argument, but got {preset} of type {type(preset).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(hardware_encoding=False, preset=preset)

    # Invalid input pixel format
    input_pixel_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an "
        f"InputPixelFormats instance for input_pixel_format argument, but got {input_pixel_format} of type "
        f"{type(input_pixel_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(input_pixel_format=input_pixel_format)

    # Invalid output pixel format
    output_pixel_format = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an "
        f"OutputPixelFormats instance for output_pixel_format argument, but got {output_pixel_format} of type "
        f"{type(output_pixel_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(output_pixel_format=output_pixel_format)

    # Invalid quantization_parameter
    quantization_parameter = None
    message = (
        f"Unable to add the VideoSaver object to the VideoSystem with id {video_system._id}. Expected an integer "
        f"between 0 and 51 for quantization_parameter argument, but got {quantization_parameter} of type "
        f"{type(quantization_parameter).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        video_system.add_video_saver(quantization_parameter=quantization_parameter)


def test_start_stop(data_logger, tmp_path, has_harvesters) -> None:
    """Primarily, verifies the functioning of the start(), stop() and all hidden runtime methods of the VideoSystem
    class.

    Also, verifies internal DataLogger bindings and logged timestamp extraction methods, as well as converting
    directories of images to video files.
    """

    # Does not test displaying threads, as this functionality is currently broken on macOS. We test it through the
    # live_run() script. Verifies using three cameras at the same time to achieve maximum feature coverage.

    # Resolves shared assets
    harvesters_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    if not has_harvesters:
        harvesters_cti_path = None
    output_directory = tmp_path.joinpath("test_output_directory")

    video_system_1 = VideoSystem(
        system_id=np.uint8(101),
        data_logger=data_logger,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )

    video_system_2 = VideoSystem(
        system_id=np.uint8(202),
        data_logger=data_logger,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )

    # Saves frames and outputs them to queue
    video_system_1.add_camera(
        save_frames=True,
        acquisition_frame_rate=10,  # With MOCK this is trivial, real cameras may employ 'shenanigans'
        display_frames=False,
        output_frames=True,
        output_frame_rate=1,
        camera_backend=CameraBackends.MOCK,
    )
    video_system_1.add_video_saver(quantization_parameter=40)

    # Just saves the frames
    video_system_2.add_camera(
        save_frames=True,
        acquisition_frame_rate=5,
        display_frames=False,
        output_frames=False,
        camera_backend=CameraBackends.MOCK,
    )
    video_system_2.add_image_saver()

    # Starts all classes
    data_logger.start()
    video_system_1.start()
    video_system_1.start()  # Ensures that calling start twice does nothing
    video_system_2.start()

    # Enables outputting frames
    timer = PrecisionTimer("s")
    video_system_1.start_frame_output()
    video_system_2.start_frame_output()
    timer.delay_noblock(delay=2)  # 2-second delay
    video_system_1.stop_frame_output()
    video_system_2.stop_frame_output()

    # The first camera is additionally configured to output frames via the output_queue. Given the output framerate of
    # 1 fps and the 2-second delay, the camera should output around 2 frames
    out_frames = []
    while not video_system_1.output_queue.empty():
        frame_tuple = video_system_1.output_queue.get()
        out_frames.append(frame_tuple[0])
    assert len(out_frames) == 2
    assert video_system_2.output_queue.empty()  # Confirms system 2 ignored the output command

    # Enables saving, but not outputting frames
    timer = PrecisionTimer("s")
    video_system_1.start_frame_saving()
    video_system_2.start_frame_saving()
    timer.delay_noblock(delay=2)  # 2-second delay
    video_system_1.stop_frame_saving()
    video_system_2.stop_frame_saving()

    # Confirms that no frames were sent to the output queue by either system
    assert video_system_1.output_queue.empty()
    assert video_system_2.output_queue.empty()

    # Stops the logger and video systems
    video_system_1.stop()
    video_system_2.stop()
    video_system_2.stop()  # Ensures that calling stop twice does nothing

    # Also, tests log decompression
    data_logger.compress_logs(remove_sources=True, memory_mapping=False)

    # Extracts the frame timestamps for each system and confirms they match the expected numbers
    frame_data_1 = video_system_1.extract_logged_data()
    frame_data_2 = video_system_2.extract_logged_data()
    assert 19 <= len(frame_data_1) < 21  # fps of 10, ran for 2 seconds should have acquired 20 frames
    assert 9 <= len(frame_data_2) < 11  # fps of 5, ran for 2 seconds should have acquired 10 frames

    # Finally, verifies converting frames acquired as images to video files
    video_system_1.encode_video_from_images(
        directory=video_system_2.output_directory, video_name="test_video", cleanup=True, target_fps=5
    )

    # Also test starting video_system without frame saving
    video_system_3 = VideoSystem(
        system_id=np.uint8(234),
        data_logger=data_logger,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )
    video_system_3.add_camera(save_frames=False, output_frames=True, camera_backend=CameraBackends.MOCK)
    video_system_3.start()
    timer.delay_noblock(delay=2)  # 2-second delay
    video_system_3.stop()
    data_logger.stop()


def test_start_errors(video_system, tmp_path) -> None:
    """Verifies the error handling of the VideoSystem start() method.

    Also verifies error-handling in the log extraction and image to video encoding methods.
    """

    # No camera
    message = (
        f"Unable to start the VideoSystem with id {np.uint8(1)}. The VideoSystem must be equipped with a Camera "
        f"before it can be started. Use add_camera() method to add a Camera class to the VideoSystem. If you "
        f"need to convert a directory of images to video, use the encode_video_from_images() method instead."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        video_system.start()

    # Camera that saves frames, but no saver
    message = (
        f"Unable to start the VideoSystem with id {np.uint8(1)}. The managed Camera is configured to save frames "
        f"and has to be matched to a Saver instance, but no Saver was added. Use add_image_saver() or "
        f"add_video_saver() method to add a Saver instance to save camera frames."
    )
    video_system.add_camera(save_frames=True, camera_backend=CameraBackends.MOCK)
    with pytest.raises(RuntimeError, match=error_format(message)):
        video_system.start()

    # Attempting to read a non-compressed (non-existent) log archive
    message = (
        f"Unable to extract frame data for VideoSystem with id {np.uint8(1)} from the log file. "
        f"This likely indicates that the logs have not been compressed via DataLogger's compress_logs() method "
        f"and are not available for processing. Call log compression method before calling this method."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        video_system.extract_logged_data()

    # Attempting to encode an 'image' directory without a VideoSaver
    message = (
        f"The VideoSystem with ID {np.uint8(1)} is unable to encode a directory of images as a video file. The "
        f"VideoSystem requires a VideoSaver class to generate video files. Call the add_video_saver() method "
        f"before calling this method()."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system.encode_video_from_images(directory=tmp_path, target_fps=10, video_name="test")
