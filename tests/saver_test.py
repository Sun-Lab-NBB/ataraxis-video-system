"""Contains tests for classes and methods stored inside the Image saver systems modules."""

import os
import re
import time
import random
from pathlib import Path
import tempfile
import textwrap
import subprocess

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
from ataraxis_base_utilities import console

from ataraxis_video_system import VideoSystem, camera
from ataraxis_video_system.saver import (
    ImageSaver,
    VideoSaver,
    VideoCodecs,
    ImageFormats,
    VideoFormats,
    CPUEncoderPresets,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from ataraxis_video_system.camera import MockCamera


@pytest.fixture()
def tmp_path() -> Path:
    tmp_path = Path(tempfile.gettempdir())
    return tmp_path


def error_format(message: str) -> str:
    """Formats the input message to match the default Console format and escapes it using re, so that it can be used to
    verify raised exceptions.

    This method is used to set up pytest 'match' clauses to verify raised exceptions.

    Args:
        message: The message to format and escape, according  to standard Ataraxis testing parameters.

    Returns:
        Formatted and escape message that can be used as the 'match' argument of pytest.raises() method.
    """
    return re.escape(textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False))


def test_check_nvidia():
    """Checks if the user's computer has an in-built GPU"""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)

    except Exception:
        return False


def test_check_openCV():
    """Checks if there is a valid openCV camera connection"""
    try:
        opencv_id = VideoSystem.get_opencv_ids()
        assert len(opencv_id) > 0

    except:
        return False


def test_check_harvesters(tmp_path):
    """Checks if there is a valid Harvesters camera connection"""

    if not os.path.exists(tmp_path):
        return False

    try:
        harvesters_id = VideoSystem.get_harvesters_ids()
        assert len(harvesters_id) > 0

    except:
        return False


def test_image_repr(tmp_path):
    """Verifies that a string representation of the ImageSaver object is returned"""

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)

    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)

    camera.connect()

    representation_string = (
        f"ImageSaver(output_directory={saver._output_directory}, image_format={saver._image_format.value},"
        f"tiff_compression_strategy={saver._tiff_parameters[1]}, jpeg_quality={saver._jpeg_parameters[1]},"
        f"jpeg_sampling_factor={saver._jpeg_parameters[3]}, png_compression_level={saver._png_parameters[1]}, "
        f"thread_count={saver._thread_count})"
    )

    assert repr(saver) == representation_string


def test_shutdown(tmp_path):
    """Verifies that the method releases class resources during shutdown. The method stops the worker
    thread and waits for all pending tasks to complete."""
    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)

    assert saver._running

    saver.shutdown()

    assert not saver._running


def test_image_save_frame(tmp_path):
    """Verifies that the connected camera has a valid id to initiate the frame saving process."""

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)

    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)
    frame_id = "a"

    camera.connect()

    frame = camera.grab_frame()

    message = (
        f"Unable to save the image with the ID {frame_id} as the ID is not valid. The ID must be a "
        f"digit-convertible string, such as 0001."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        saver.save_frame(frame_id=frame_id, frame=frame)


# noinspection PyRedundantParentheses
@pytest.mark.parametrize(
    "image_format",
    [(ImageFormats.TIFF), (ImageFormats.PNG), (ImageFormats.JPG)],
)
def test_save_image(image_format, tmp_path):
    """Verifies that both monochrome and colored image frames are saved in sequence to the chosen output path. JPEG images
    are set to have a jpeg quality of 100 to replicate lossless compression. The difference between the image frames obtained
    from the mock camera and the saved images is set to have a tolerance of 3 pixels."""

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=2, height=2)
    saver = ImageSaver(output_directory=tmp_path.joinpath("TestSaveImage"), image_format=image_format, jpeg_quality=100)

    camera.connect()

    frame_data = camera.grab_frame()

    image_id = "235"

    output_path = Path(saver._output_directory, f"{image_id}.{saver._image_format.value}")

    saver.save_frame(image_id, frame_data)

    time.sleep(3)

    image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

    assert np.allclose(frame_data, image, atol=3)


@pytest.mark.parametrize(
    "video_codec, hardware_encoding, output_pixel_format, preset",
    [
        (VideoCodecs.H265, True, OutputPixelFormats.YUV444, GPUEncoderPresets.MEDIUM),
        (VideoCodecs.H265, True, OutputPixelFormats.YUV420, GPUEncoderPresets.MEDIUM),
        (VideoCodecs.H264, True, OutputPixelFormats.YUV444, GPUEncoderPresets.MEDIUM),
        (VideoCodecs.H264, True, OutputPixelFormats.YUV420, GPUEncoderPresets.MEDIUM),
        (VideoCodecs.H265, False, OutputPixelFormats.YUV444, CPUEncoderPresets.MEDIUM),
        (VideoCodecs.H265, False, OutputPixelFormats.YUV420, CPUEncoderPresets.MEDIUM),
        (VideoCodecs.H264, False, OutputPixelFormats.YUV444, CPUEncoderPresets.MEDIUM),
        (VideoCodecs.H264, False, OutputPixelFormats.YUV420, CPUEncoderPresets.MEDIUM),
    ],
)
def test_input_pipe(video_codec, hardware_encoding, output_pixel_format, preset, tmp_path):
    """Verifies that only one live encoder can be created and the video is saved to the correct path"""

    if hardware_encoding and not test_check_nvidia():
        pytest.skip("GPU not found.")

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=2, height=2)
    saver = VideoSaver(
        output_directory=tmp_path.joinpath("TestInputPipe"),
        hardware_encoding=hardware_encoding,
        video_format=VideoFormats.MP4,
        video_codec=video_codec,
        preset=GPUEncoderPresets.MEDIUM,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=output_pixel_format,
    )

    camera.connect()

    assert not saver.is_live

    saver.create_live_video_encoder(frame_width=400, frame_height=400, video_id="2", video_frames_per_second=45)

    assert saver.is_live

    for _ in range(20):
        frame_data = camera.grab_frame()
        saver.save_frame(_frame_id=1, frame=frame_data)

    if random.randint(1, 2) == 1:
        del saver
    else:
        saver.terminate_live_encoder()

    assert tmp_path.joinpath("TestInputPipe/2.mp4").exists()


def test_video_repr(tmp_path):
    """Verifies that a string representation of the VideoEncoder object is returned."""

    saver = VideoSaver(
        output_directory=tmp_path.joinpath("TestVideoRepr"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    saver._ffmpeg_process = None
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=False)"
    assert repr(saver) == representation_string

    saver._ffmpeg_process = True
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=True)"
    assert repr(saver) == representation_string


def test_error_live_encoder(tmp_path):
    """Verifies that only 1 live encoder can be active at a time"""

    saver = VideoSaver(
        output_directory=tmp_path.joinpath("TestErrorLiveEncoder"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    saver.create_live_video_encoder(video_id="2", frame_width=400, frame_height=400, video_frames_per_second=45)

    message = (
        f"Unable to create live video encoder for video {2}. FFMPEG process already exists and a "
        f"video saver class can have at most one 'live' encoder at a time. Call the terminate_live_encoder() "
        f"method to terminate the existing encoder before creating a new one."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.create_live_video_encoder(video_id="2", frame_width=400, frame_height=400, video_frames_per_second=45)


def test_video_save_frame(tmp_path):
    """Verifies that frames are properly saved to the output path provided and are saved in order"""

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)

    saver = VideoSaver(
        output_directory=tmp_path.joinpath("TestVideoSaveFrame"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    camera.connect()

    frame = camera.grab_frame()

    message = (
        f"Unable to submit the frame to a 'live' FFMPEG encoder process as the process does not exist. Call "
        f"create_live_video_encoder() method to create a 'live' encoder before calling save_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.save_frame(_frame_id=123, frame=frame)


def test_create_video_from_image_folder(tmp_path):
    """Verifies that all image files in the image directory are extracted and sorted based on their integer IDs"""

    if (not test_check_openCV()) or (not test_check_harvesters):
        pytest.skip("Supported camera not found.")

    test_directory = tmp_path.joinpath("TestCreateVideoFromImageFolder")

    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=400, height=400)
    saver = VideoSaver(
        output_directory=test_directory.joinpath("Test"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )
    camera.connect()

    image_saver = ImageSaver(output_directory=test_directory.joinpath("TestImages"), image_format=ImageFormats.PNG)
    for frame_id in range(20):
        frame_data = camera.grab_frame()
        image_saver.save_frame(frame=frame_data, frame_id=str(frame_id))

    # Discover and sort images from the directory
    supported_image_formats = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    video_id = "2"

    images = sorted(
        [
            img
            for img in test_directory.joinpath("TestImages").iterdir()
            if img.is_file() and img.suffix.lower() in supported_image_formats and img.stem.isdigit()
        ],
        key=lambda x: int(x.stem),
    )
    assert len(images) > 0

    empty_folder = test_directory.joinpath("EmptyFolder")
    console._ensure_directory_exists(empty_folder)
    message = (
        f"Unable to create video {video_id} from images. No valid image candidates discovered when crawling "
        f"the image directory ({empty_folder}). Valid candidates are images using one of the supported "
        f"file-extensions ({sorted(supported_image_formats)}) with "
        f"digit-convertible names (e.g: 0001.jpg)."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.create_video_from_image_folder(video_frames_per_second=5, image_directory=empty_folder, video_id=video_id)

    assert all(int(images[i].stem) <= int(images[i + 1].stem) for i in range(len(images) - 1))

    saver.create_video_from_image_folder(
        image_directory=test_directory.joinpath("TestImages"), video_id=video_id, video_frames_per_second=5
    )

    video_path = Path(saver._output_directory, f"{video_id}.{saver._video_format}")
    assert video_path.exists()


def test_terminate_live_encoder(tmp_path):
    saver = VideoSaver(
        output_directory=tmp_path.joinpath("TestTerminateLiveEncoder"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    """Verifies that the function returns nothing when terminate_live_encoder() is called when an ffmpeg 
    process does not exist"""
    saver._ffmpeg_process = None

    saver.terminate_live_encoder(timeout=1)

    assert saver._ffmpeg_process is None

    """Verifies that the video saver system can properly terminates alive' FFMPEG encoder process"""
    saver._ffmpeg_process = subprocess.Popen(["sleep", "30"])

    assert saver._ffmpeg_process.poll() is None

    saver.terminate_live_encoder(timeout=1)

    assert saver._ffmpeg_process is None
