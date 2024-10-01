"""Contains tests for classes and methods stored inside the Image saver systems modules."""

import os
import re
import time
from pathlib import Path
import tempfile
import textwrap

from PIL import Image
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

from ataraxis_video_system.saver import (
    ImageSaver,
    VideoSaver,
    VideoCodecs,
    ImageFormats,
    VideoFormats,
    InputPixelFormats,
    OutputPixelFormats,
)
from ataraxis_video_system.camera import MockCamera, OpenCVCamera


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


def test_image_repr(tmp_path):
    """Verifies that a string representation of the ImageSaver object is returned"""

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

    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)

    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)

    assert saver._running

    saver.shutdown()

    assert not saver._running


def test_save_frame(tmp_path):
    """Verifies that the connected camera has a valid id to initiate the frame saving process."""

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
        _ = saver.save_frame(frame_id=frame_id, frame=frame)


@pytest.mark.parametrize(
    "format",
    [(ImageFormats.TIFF), (ImageFormats.PNG), (ImageFormats.JPG)],
)
def test_save_image(format, tmp_path):
    """Verifies that both monochrome and colored image frames are saved in sequence to the chosen output path. JPEG images
    are set to have a jpeg quality of 100 to replicate lossless compression. The difference between the image frames obtained
    from the mock camera and the saved images is set to have a tolerance of 3 pixels."""

    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=2, height=2)
    saver = ImageSaver(output_directory=Path("/Users/natalieyeung/Desktop/Test"), image_format=format, jpeg_quality=100)

    camera.connect()

    frame_data = camera.grab_frame()

    image_id = "235"

    output_path = Path(saver._output_directory, f"{image_id}.{saver._image_format.value}")

    saver.save_frame(image_id, frame_data)

    time.sleep(3)

    image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

    assert np.allclose(frame_data, image, atol=3)


# @pytest.mark.parametrize(
#     "video_codec, hardware_encoding, expected_encoder, output_pixel_format, encoder_profile",
#     [
#         (VideoCodecs.H264, True, "h264_nvenc", "yuv444p", "high444p"),
#         (VideoCodecs.H265, True, "h264_nvenc", "yuv420p", "main"),
#         (VideoCodecs.H265, True, "hevc_nvenc", "yuv444p", "rext"),
#         (VideoCodecs.H265, True, "hevc_nvenc", "yuv420p", "main"),
#         (VideoCodecs.H265, False, "libx265", "yuv444p", "main444-8"),
#         (VideoCodecs.H265, False, "libx265", "yuv444p", "main"),
#         (VideoCodecs.H264, False, "libx264", "yuv444p", "high444"),
#         (VideoCodecs.H264, False, "libx264", "yuv420p", "high420"),
#     ],
# )


def test_is_live():
    """Verifies that the method returns True if the class is running an active 'live' encoder and False otherwise."""

    saver = VideoSaver(
        output_directory=Path("/Users/natalieyeung/Desktop/Test"),
        hardware_encoding=OutputPixelFormats.YUV444,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    saver._ffmpeg_process = None
    assert saver.is_live == False

    saver._ffmpeg_process = True
    assert saver.is_live == True


def test_video_repr():
    """Verifies that a string representation of the VideoEncoder object is returned."""

    saver = VideoSaver(
        output_directory=Path("/Users/natalieyeung/Desktop/Test"),
        hardware_encoding=OutputPixelFormats.YUV444,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    saver._ffmpeg_process = None
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=False)"

    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=False)"
    assert repr(saver) == representation_string

    saver._ffmpeg_process = True
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=True)"
    assert repr(saver) == representation_string


def test_create_live_video_encoder():
    pass


def test_saveframe(video_codec, hardware_encoding, expected_encoder, output_pixel_format, encoder_profile):
    pass
