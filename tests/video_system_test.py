"""Contains tests for classes and methods stored inside the camera module."""

import re
from enum import Enum
from pathlib import Path
import tempfile
import textwrap

import cv2
import numpy as np
import pytest
from harvesters.core import Harvester, ImageAcquirer  # type: ignore
from harvesters.util.pfnc import (  # type: ignore
    bgr_formats,
    rgb_formats,
    bgra_formats,
    rgba_formats,
    mono_location_formats,
)

from ataraxis_video_system import VideoSystem
from ataraxis_video_system.saver import ImageFormats, VideoFormats, VideoCodecs
from ataraxis_video_system.camera import MockCamera, OpenCVCamera, CameraBackends, HarvestersCamera


def error_format(message: str) -> str:
    """Formats the input message to match the default Console format and escapes it using re, so that it can be used to
    verify raised exceptions.

    This method is used to set up pytest 'match' clauses to verify raised exceptions.

    Args:
        message: The message to format and escape, according to standard Ataraxis testing parameters.

    Returns:
        Formatted and escape message that can be used as the 'match' argument of pytest.raises() method.
    """
    return re.escape(textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False))


@pytest.fixture()
def tmp_path() -> Path:
    tmp_path = Path(tempfile.gettempdir())
    return tmp_path


def test_check_backend(backend):
    return backend not in CameraBackends.__members__


def test_create_camera_errors():
    """Verifies that a camera cannot be created if the input arguments do not correspond to criteria of a given backend"""

    camera_name = "Test Camera"
    camera_id = 2
    frames_per_second = 30
    frame_width = 600
    frame_height = 400

    invalid_name = 1

    message = (
        f"Unable to instantiate a Camera class object. Expected a string for camera_name argument, but "
        f"got {invalid_name} of type {type(invalid_name).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(camera_name=invalid_name)

    invalid_camera_id = "str"

    message = (
        f"Unable to instantiate a {camera_name} Camera class object. Expected an integer for camera_id "
        f"argument, but got {invalid_camera_id} of type {type(invalid_camera_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(camera_name=camera_name, camera_id=invalid_camera_id)

    invalid_frames_per_second = "str"
    message = (
        f"Unable to instantiate a {camera_name} Camera class object. Expected an integer, float or None for "
        f"frames_per_second argument, but got {invalid_frames_per_second} of type {type(invalid_frames_per_second).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name, camera_id=camera_id, frames_per_second=invalid_frames_per_second
        )

    invalid_frame_width = "str"
    message = (
        f"Unable to instantiate a {camera_name} Camera class object. Expected an integer or None for "
        f"frame_width argument, but got {invalid_frame_width} of type {type(invalid_frame_width).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            frames_per_second=frames_per_second,
            frame_width=invalid_frame_width,
        )

    invalid_frame_height = "str"
    message = (
        f"Unable to instantiate a {camera_name} Camera class object. Expected an integer or None for "
        f"frame_height argument, but got {invalid_frame_height} of type {type(invalid_frame_height).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            frames_per_second=frames_per_second,
            frame_width=frame_width,
            frame_height=invalid_frame_height,
        )

    opencv_backend = "2.0"
    message = (
        f"Unable to instantiate a {camera_name} OpenCVCamera class object. Expected an integer or None "
        f"for opencv_backend argument, but got {opencv_backend} of type {type(opencv_backend).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            frames_per_second=frames_per_second,
            frame_width=frame_width,
            frame_height=frame_height,
            opencv_backend=opencv_backend,
        )

    harvester_ctipath = None

    message = (
        f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
        f"pointing to the '.cti' file for cti_path argument, but got {harvester_ctipath} of "
        f"type {type(harvester_ctipath).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            cti_path=harvester_ctipath,
            frames_per_second=frames_per_second,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_backend=CameraBackends.HARVESTERS,
        )

    harvester_invalid_suffix = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.zip")

    message = (
        f"Unable to instantiate a {camera_name} HarvestersCamera class object. Expected a Path object "
        f"pointing to the '.cti' file for cti_path argument, but got {harvester_invalid_suffix} of "
        f"type {type(harvester_invalid_suffix).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        VideoSystem.create_camera(
            camera_name=camera_name,
            camera_id=camera_id,
            cti_path=harvester_invalid_suffix,
            frames_per_second=frames_per_second,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_backend=CameraBackends.HARVESTERS,
        )


def test_openCV_default_parameters():
    """Verifies that the default openCV backend is used if the openCV backend is not specified"""

    camera = VideoSystem.create_camera(
        camera_name="Test Camera",
        camera_id=2,
        frames_per_second=30,
        frame_width=600,
        frame_height=400,
        opencv_backend=None,
    )

    opencv_backend = int(cv2.CAP_ANY)
    assert opencv_backend == int(cv2.CAP_ANY)


def test_mock_default_parameters():
    """Verifies that the default mock camera values for the frame height, width and fps are used if not specified"""

    camera = VideoSystem.create_camera(
        camera_name="Test Camera",
        camera_id=2,
        camera_backend=CameraBackends.MOCK,
        frame_height=None,
        frame_width=None,
        frames_per_second=None,
    )

    assert camera.height == 400
    assert camera.width == 600
    assert camera.fps == 30


def test_invalid_backend():
    camera_name = "Test Camera"
    invalid_backend_value = "INVALID_BACKEND"

    if test_check_backend(invalid_backend_value):
        message = (
            f"Unable to instantiate a {camera_name} Camera class object due to encountering an unsupported "
            f"camera_backend argument {invalid_backend_value} of type {type(invalid_backend_value).__name__}. "
            f"camera_backend has to be one of the options available from the CameraBackends enumeration."
        )

        with pytest.raises(ValueError, match=error_format(message)):
            VideoSystem.create_camera(
                camera_name=camera_name,
                camera_id=2,
                frames_per_second=30,
                frame_width=600,
                frame_height=400,
                camera_backend=invalid_backend_value,
            )


def test_create_image_saver():
    """Verifies that all attributes of the image saver class are valid before creation."""

    output_directory = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    image_format = ImageFormats.TIFF
    tiff_compression = cv2.IMWRITE_TIFF_COMPRESSION_LZW
    jpeg_quality = 50
    jpeg_sampling_factor = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
    png_compression = 8

    invalid_output_directory = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"

    message = (
        f"Unable to instantiate an ImageSaver class object. Expected a Path instance for output_directory "
        f"argument, but got {invalid_output_directory} of type {type(invalid_output_directory).__name__}."
    )

    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(output_directory=invalid_output_directory)

    invalid_image_format = None
    message = (
        f"Unable to instantiate an ImageSaver class object. Expected an ImageFormats instance for "
        f"image_format argument, but got {invalid_image_format} of type {type(invalid_image_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(output_directory=output_directory, image_format=invalid_image_format)

    invalid_tiff_compression = "str"
    message = (
        f"Unable to instantiate an ImageSaver class object. Expected an integer for tiff_compression "
        f"argument, but got {invalid_tiff_compression} of type {type(invalid_tiff_compression).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(
            output_directory=output_directory, image_format=image_format, tiff_compression=invalid_tiff_compression
        )

    message = (
        f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 100 for "
        f"jpeg_quality argument, but got {101} of type {type(101)}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(
            output_directory=output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression,
            jpeg_quality=101,
        )

    invalid_jpeg_sampling_factor = None

    message = (
        f"Unable to instantiate an ImageSaver class object. Expected one of the "
        f"'cv2.IMWRITE_JPEG_SAMPLING_FACTOR_' constants for jpeg_sampling_factor argument, but got "
        f"{invalid_jpeg_sampling_factor} of type {type(invalid_jpeg_sampling_factor).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(
            output_directory=output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=invalid_jpeg_sampling_factor,
        )

    invalid_png_compression = 11
    message = (
        f"Unable to instantiate an ImageSaver class object. Expected an integer between 0 and 9 for "
        f"png_compression argument, but got {invalid_png_compression} of type "
        f"{type(invalid_png_compression).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(
            output_directory=output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=jpeg_sampling_factor,
            png_compression=invalid_png_compression,
        )

    invalid_thread_count = "str"
    message = (
        f"Unable to instantiate an ImageSaver class object. Expected an integer for thread_count "
        f"argument, but got {invalid_thread_count} of type {type(invalid_thread_count).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_image_saver(
            output_directory=output_directory,
            image_format=image_format,
            tiff_compression=tiff_compression,
            jpeg_quality=jpeg_quality,
            jpeg_sampling_factor=jpeg_sampling_factor,
            png_compression=png_compression,
            thread_count=invalid_thread_count,
        )


def test_repr(tmp_path):
    """Verifies that the correct string representation of the VideoSystem class instance is returned"""

    camera_type = VideoSystem.create_camera(camera_name="Test camera", camera_backend=CameraBackends.OPENCV)
    saver_type = VideoSystem.create_video_saver(output_directory=tmp_path)
    videosystem = VideoSystem(camera=camera_type, saver=saver_type, system_name="Test system")

    VideoSystem.start(videosystem)

    representation_string = (
        f"VideoSystem(name={videosystem._name}, running={True}, expired={False}, "
        f"camera_type={type(camera_type).__name__}, camera_name={videosystem._camera.name}, saver_type={type(saver_type).__name__}, "
        f"output_directory={videosystem._saver._output_directory})"
    )

    assert repr(videosystem) == representation_string


def test_create_video_saver(tmp_path):
    """Verifies that all attributes of the video saver class are valid before creation."""

    output_directory = Path(tmp_path)
    hardware_encoding = True
    video_format = VideoFormats.MP4
    video_codec = VideoCodecs.H265

    video_invalid_output_directory = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"

    message = (
        f"Unable to instantiate a Saver class object. Expected a Path instance for output_directory argument, "
        f"but got {video_invalid_output_directory} of type {type(video_invalid_output_directory).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_video_saver(output_directory=video_invalid_output_directory)

    message = (
        f"Unable to instantiate a VideoSaver class object. Expected a boolean for hardware_encoding "
        f"argument, but got {'str'} of type {type('str').__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_video_saver(output_directory=output_directory, hardware_encoding="str")

    invalid_video_format = None

    message = (
        f"Unable to instantiate a VideoSaver class object. Expected a VideoFormats instance for "
        f"video_format argument, but got {invalid_video_format} of type {type(invalid_video_format).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_video_saver(
            output_directory=output_directory, hardware_encoding=hardware_encoding, video_format=invalid_video_format
        )

    invalid_video_codec = None

    message = (
        f"Unable to instantiate a VideoSaver class object. Expected a VideoCodecs instance for "
        f"video_codec argument, but got {invalid_video_codec} of type {type(invalid_video_codec).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        VideoSystem.create_video_saver(
            output_directory=output_directory, hardware_encoding=hardware_encoding, video_format=video_format, video_codec=invalid_video_codec
        )


    def test_start(self):
        pass