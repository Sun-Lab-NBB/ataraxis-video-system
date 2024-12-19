"""Contains tests for classes and methods provided by the video_system.py module."""

import copy
import queue
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import DataLogger

from ataraxis_video_system import VideoSystem
from ataraxis_video_system.saver import VideoCodecs, ImageFormats, VideoFormats
from ataraxis_video_system.camera import OpenCVCamera, CameraBackends, HarvestersCamera

"""
1) init
2) init_error
3) repr+ property
4) add_saver 
5) add_saver + errors
-
"""


@pytest.fixture()
def logger_queue_fixture(tmp_path):
    """
    Fixture for DataLogger's input queue.
    """
    data_logger = DataLogger(output_directory=tmp_path)
    return DataLogger.input_queue


@pytest.fixture
def video_system_fixture(logger_queue_fixture):
    """
    Fixture for creating an instance of VideoSystem.
    """
    system_id = np.uint8(1)
    system_name = "Test System"
    system_description = "Video system initialization test"
    logger_queue = logger_queue_fixture
    output_directory = Path("")
    harvesters_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")

    return VideoSystem(
        system_id=system_id,
        system_name=system_name,
        system_description=system_description,
        logger_queue=logger_queue,
        output_directory=output_directory,
        harvesters_cti_path=harvesters_cti_path,
    )


def test_init_errors(video_system_fixture):
    invalid_system_name = 12

    message = (
        f"Unable to initialize the VideoSystem class instance. Expected a string for system_name, but got "
        f"{invalid_system_name} of type {type(invalid_system_name).__name__}."
    )
    with pytest.raises(TypeError, match=message):
        video_system = VideoSystem(
            system_id=video_system_fixture._id,
            system_name=invalid_system_name,
            system_description=video_system_fixture._description,
            logger_queue=video_system_fixture._logger_queue,
            output_directory=video_system_fixture._output_directory,
        )


def test_repr(video_system_fixture):
    """Verifies that the correct string representation of the VideoSystem class instance is returned"""

    representation_string: str = (
        f"VideoSystem(name={video_system_fixture._name}, running={video_system_fixture._started}, "
        f"camera_count={len(video_system_fixture._cameras)}, saver_count={len(video_system_fixture._savers)})"
    )

    assert repr(video_system_fixture) == representation_string


def test_add_camera(video_system_fixture):
    """Verifies that a camera cannot be created if the input arguments do not correspond to criteria of a given backend"""

    camera_name = "Test Camera"
    camera_id = 2
    acquisition_frame_rate = 30
    frame_width = 600
    frame_height = 400
    save_frames = True

    invalid_name = 1
    message = (
        f"Unable to add the Camera object to the {video_system_fixture._name} VideoSystem. Expected a string for camera_name "
        f"argument, but got {invalid_name} of type {type(invalid_name).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(camera_name=invalid_name, save_frames=save_frames)

    invalid_camera_id = "str"

    message = (
        f"Unable to add the {camera_name} Camera object to the {video_system_fixture._name} VideoSystem. Expected an integer "
        f"for camera_id argument, but got {invalid_camera_id} of type {type(invalid_camera_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(camera_name=camera_name, camera_id=invalid_camera_id, save_frames=save_frames)

    invalid_acquisition_frame_rate = "str"
    message = (
        f"Unable to add the {camera_name} Camera object to the {video_system_fixture._name} VideoSystem. Expected an integer, "
        f"float or None for acquisition_frame_rate argument, but got {invalid_acquisition_frame_rate} of type "
        f"{type(invalid_acquisition_frame_rate).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            acquisition_frame_rate=invalid_acquisition_frame_rate,
            save_frames=save_frames,
        )

    invalid_frame_width = "str"
    message = (
        f"Unable to add the {camera_name} Camera object to the {video_system_fixture._name} VideoSystem. Expected an integer "
        f"or None for frame_width argument, but got {invalid_frame_width} of type {type(invalid_frame_width).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            acquisition_frame_rate=acquisition_frame_rate,
            frame_width=invalid_frame_width,
            save_frames=save_frames,
        )

    invalid_frame_height = "str"
    message = (
        f"Unable to add the {camera_name} Camera object to the {video_system_fixture._name} VideoSystem. Expected an integer "
        f"or None for frame_height argument, but got {invalid_frame_height} of type {type(invalid_frame_height).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            acquisition_frame_rate=acquisition_frame_rate,
            frame_width=frame_width,
            frame_height=invalid_frame_height,
            save_frames=save_frames,
        )

    opencv_backend = "2.0"
    message = (
        f"Unable to add the {camera_name} OpenCVCamera object to the {video_system_fixture._name} VideoSystem. Expected "
        f"an integer or None for opencv_backend argument, but got {opencv_backend} of type "
        f"{type(opencv_backend).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        video_system_fixture.add_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            acquisition_frame_rate=acquisition_frame_rate,
            frame_width=frame_width,
            frame_height=frame_height,
            opencv_backend=opencv_backend,
            save_frames=save_frames,
            output_frames=True,
        )


def test_opencv_backend_assignment(video_system_fixture):
    """Test that opencv_backend is correctly assigned when None is provided and camera_backend is OPENCV."""

    # I'm not sure how to do this because addcamera does not have opencv_backend. Like for each backend how do
    # do I create a valid camera? Do I create a an OpenCVCamera or HarvestsersCamera then add it in?

    camera_backend = CameraBackends.OPENCV
    opencv_backend = None

    video_system_fixture.add_camera(
        camera_name="Test Camera",
        camera_id=0,
        camera_backend=camera_backend,
        opencv_backend=opencv_backend,
        save_frames=True,
    )
    camera = video_system_fixture._cameras[-1]
    expected_backend_value = int(cv2.CAP_ANY)

    assert camera.opencv_backend == expected_backend_value


#     harvester_ctipath = None

#     message = (
#         f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
#         f"pointing to the '.cti' file for cti_path argument, but got {harvester_ctipath} of "
#         f"type {type(harvester_ctipath).__name__}."
#     )
#     with pytest.raises(ValueError, match=error_format(message)):
#         VideoSystem.create_camera(
#             camera_name=camera_name,
#             camera_id=camera_id,
#             cti_path=harvester_ctipath,
#             frames_per_second=frames_per_second,
#             frame_width=frame_width,
#             frame_height=frame_height,
#             camera_backend=CameraBackends.HARVESTERS,
#         )

#     harvester_invalid_suffix = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.zip")

#     message = (
#         f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
#         f"pointing to the '.cti' file for cti_path argument, but got {harvester_invalid_suffix} of "
#         f"type {type(harvester_invalid_suffix).__name__}."
#     )
#     with pytest.raises(ValueError, match=error_format(message)):
#         VideoSystem.create_camera(
#             camera_name=camera_name,
#             camera_id=camera_id,
#             cti_path=harvester_invalid_suffix,
#             frames_per_second=frames_per_second,
#             frame_width=frame_width,
#             frame_height=frame_height,
#             camera_backend=CameraBackends.HARVESTERS,
#         )


# def test_openCV_default_parameters():
#     """Verifies that the default openCV backend is used if the openCV backend is not specified"""

#     camera = VideoSystem.create_camera(
#         camera_name="Test Camera",
#         camera_id=2,
#         frames_per_second=30,
#         frame_width=600,
#         frame_height=400,
#         opencv_backend=None,
#     )

#     opencv_backend = int(cv2.CAP_ANY)
#     assert opencv_backend == int(cv2.CAP_ANY)


# def test_mock_default_parameters():
#     """Verifies that the default mock camera values for the frame height, width and fps are used if not specified"""

#     camera = VideoSystem.create_camera(
#         camera_name="Test Camera",
#         camera_id=2,
#         camera_backend=CameraBackends.MOCK,
#         frame_height=None,
#         frame_width=None,
#         frames_per_second=None,
#     )

#     assert camera.height == 400
#     assert camera.width == 600
#     assert camera.fps == 30


# def test_invalid_backend():
#     camera_name = "Test Camera"
#     invalid_backend_value = "INVALID_BACKEND"

#     message = (
#         f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
#         f"camera_backend argument {invalid_backend_value} of type {type(invalid_backend_value).__name__}. "
#         f"camera_backend has to be one of the options available from the CameraBackends enumeration."
#     )

#     with pytest.raises(ValueError, match=error_format(message)):
#         VideoSystem.create_camera(
#             camera_name=camera_name,
#             camera_id=2,
#             frames_per_second=30,
#             frame_width=600,
#             frame_height=400,
#             camera_backend=invalid_backend_value,
#         )


# def test_create_image_saver():
#     """Verifies that all attributes of the image saver class are valid before creation."""

#     output_directory = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
#     image_format = ImageFormats.TIFF
#     tiff_compression = cv2.IMWRITE_TIFF_COMPRESSION_LZW
#     jpeg_quality = 50
#     jpeg_sampling_factor = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
#     png_compression = 8

#     invalid_output_directory = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"

#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected a Path instance for output_directory "
#         f"argument, but got {invalid_output_directory} of type {type(invalid_output_directory).__name__}."
#     )

#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(output_directory=invalid_output_directory)

#     invalid_image_format = None
#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected an ImageFormats instance for "
#         f"image_format argument, but got {invalid_image_format} of type {type(invalid_image_format).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(output_directory=output_directory, image_format=invalid_image_format)

#     invalid_tiff_compression = "str"
#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected an integer for tiff_compression "
#         f"argument, but got {invalid_tiff_compression} of type {type(invalid_tiff_compression).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(
#             output_directory=output_directory, image_format=image_format, tiff_compression=invalid_tiff_compression
#         )

#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 100 for "
#         f"jpeg_quality argument, but got {101} of type {type(101)}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(
#             output_directory=output_directory,
#             image_format=image_format,
#             tiff_compression=tiff_compression,
#             jpeg_quality=101,
#         )

#     invalid_jpeg_sampling_factor = None

#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected one of the "
#         f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_' constants for jpeg_sampling_factor argument, but got "
#         f"{invalid_jpeg_sampling_factor} of type {type(invalid_jpeg_sampling_factor).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(
#             output_directory=output_directory,
#             image_format=image_format,
#             tiff_compression=tiff_compression,
#             jpeg_quality=jpeg_quality,
#             jpeg_sampling_factor=invalid_jpeg_sampling_factor,
#         )

#     invalid_png_compression = 11
#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 9 for "
#         f"png_compression argument, but got {invalid_png_compression} of type "
#         f"{type(invalid_png_compression).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(
#             output_directory=output_directory,
#             image_format=image_format,
#             tiff_compression=tiff_compression,
#             jpeg_quality=jpeg_quality,
#             jpeg_sampling_factor=jpeg_sampling_factor,
#             png_compression=invalid_png_compression,
#         )

#     invalid_thread_count = "str"
#     message = (
#         f"Unable to instantiate an ImageSaver class object. Expected an integer for thread_count "
#         f"argument, but got {invalid_thread_count} of type {type(invalid_thread_count).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_image_saver(
#             output_directory=output_directory,
#             image_format=image_format,
#             tiff_compression=tiff_compression,
#             jpeg_quality=jpeg_quality,
#             jpeg_sampling_factor=jpeg_sampling_factor,
#             png_compression=png_compression,
#             thread_count=invalid_thread_count,
#         )


# def test_create_video_saver(tmp_path):
#     """Verifies that all attributes of the video saver class are valid before creation."""

#     output_directory = Path(tmp_path)
#     hardware_encoding = True
#     video_format = VideoFormats.MP4
#     video_codec = VideoCodecs.H265

#     video_invalid_output_directory = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"

#     message = (
#         f"Unable to instantiate a Saver class object. Expected a Path instance for output_directory argument, "
#         f"but got {video_invalid_output_directory} of type {type(video_invalid_output_directory).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_video_saver(output_directory=video_invalid_output_directory)

#     message = (
#         f"Unable to instantiate a VideoSaver class object. Expected a boolean for hardware_encoding "
#         f"argument, but got {'str'} of type {type('str').__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_video_saver(output_directory=output_directory, hardware_encoding="str")

#     invalid_video_format = None

#     message = (
#         f"Unable to instantiate a VideoSaver class object. Expected a VideoFormats instance for "
#         f"video_format argument, but got {invalid_video_format} of type {type(invalid_video_format).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_video_saver(
#             output_directory=output_directory, hardware_encoding=hardware_encoding, video_format=invalid_video_format
#         )

#     invalid_video_codec = None

#     message = (
#         f"Unable to instantiate a VideoSaver class object. Expected a VideoCodecs instance for "
#         f"video_codec argument, but got {invalid_video_codec} of type {type(invalid_video_codec).__name__}."
#     )
#     with pytest.raises(TypeError, match=error_format(message)):
#         VideoSystem.create_video_saver(
#             output_directory=output_directory,
#             hardware_encoding=hardware_encoding,
#             video_format=video_format,
#             video_codec=invalid_video_codec,
#         )

#     def test_start(self):
#         pass
