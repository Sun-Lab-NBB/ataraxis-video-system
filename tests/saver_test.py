"""Contains tests for classes and methods provided by the saver.py module."""

import time
import subprocess

import cv2
import numpy as np
import pytest
from ataraxis_base_utilities import error_format, ensure_directory_exists

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


def test_image_saver_repr(tmp_path):
    """Verifies the functioning of the ImageSaver class __repr__ method."""

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)
    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)
    camera.connect()

    # Verifies the representation string
    representation_string = (
        f"ImageSaver(output_directory={saver._output_directory}, image_format={saver._image_format.value},"
        f"tiff_compression_strategy={saver._tiff_parameters[1]}, jpeg_quality={saver._jpeg_parameters[1]},"
        f"jpeg_sampling_factor={saver._jpeg_parameters[3]}, png_compression_level={saver._png_parameters[1]}, "
        f"thread_count={saver._thread_count})"
    )
    assert repr(saver) == representation_string


def test_image_saver_shutdown(tmp_path):
    """Verifies the functioning of the ImageSaver shutdown() method."""

    # Setup and ensures the class is initialized
    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)
    assert saver._running

    # Shutdown
    saver.shutdown()
    assert not saver._running


def test_image_saver_save_frame_errors(tmp_path):
    """Verifies the error handling of the ImageSaver save_frame() method."""

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)
    saver = ImageSaver(output_directory=tmp_path, image_format=ImageFormats.PNG)

    # Generates a test frame
    camera.connect()
    frame = camera.grab_frame()

    # Verifies that the method correctly fails when an invalid frame ID is provided
    frame_id = "a"
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
    """Verifies the functioning of the ImageSaver save_frame() method.

    Notes:
        JPEG images are set to have a jpeg quality of 100 to replicate lossless compression. The difference between the
        image frames obtained from the mock camera and the saved images is set to have a tolerance of 3 intensity-values
        for every pixel.
    """

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=2, height=2)
    saver = ImageSaver(output_directory=tmp_path, image_format=image_format, jpeg_quality=100)

    # Connects to the camera and grabs the test frame
    camera.connect()
    frame_data = camera.grab_frame()

    # Sends the frame to be saved
    image_id = "235"
    output_path = tmp_path.joinpath(f"{image_id}.{saver._image_format.value}")
    saver.save_frame(image_id, frame_data)

    time.sleep(1)  # Short delay to ensure the frame is saved before checking integrity (see below)

    # Reads the saved frame and ensures it is similar enough to the original generated frame.
    image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert np.allclose(frame_data, image, atol=3)  # Every pixel has to be within 3 intensity values from the origin


def test_video_saver_repr(tmp_path):
    """Verifies the functioning of the VideoSaver class __repr__ method."""

    # Setup
    saver = VideoSaver(
        output_directory=tmp_path,
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    # Verifies the representation string of a non-live saver
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=False)"
    assert repr(saver) == representation_string

    # Initializes the live video saver and verifies that this is reflected in the representation string
    saver.create_live_video_encoder(frame_width=100, frame_height=100, video_id="TestVideo", video_frames_per_second=30)
    representation_string = f"VideoSaver({saver._repr_body}, live_encoder=True)"
    assert repr(saver) == representation_string


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
def test_video_saver_save_frame(video_codec, hardware_encoding, output_pixel_format, preset, tmp_path):
    """Verifies the functioning of the VideoSaver save_frame() and create_live_video_encoder() methods."""

    # Skips GPU-bound tests if no valid NVIDIA GPU is available.
    if hardware_encoding and not check_nvidia():
        pytest.skip(f"Skipping this test as it requires an NVIDIA GPU.")

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=2, height=2)
    output_path = tmp_path
    saver = VideoSaver(
        output_directory=output_path,
        hardware_encoding=hardware_encoding,
        video_format=VideoFormats.MP4,
        video_codec=video_codec,
        preset=GPUEncoderPresets.MEDIUM,
        input_pixel_format=InputPixelFormats.BGR,
        output_pixel_format=output_pixel_format,
    )
    camera.connect()

    # Tests live video encoder creation.
    assert not saver.is_live
    saver.create_live_video_encoder(frame_width=400, frame_height=400, video_id="TestID", video_frames_per_second=45)
    assert saver.is_live

    # Generates and saves 20 test frames
    for _ in range(20):
        frame_data = camera.grab_frame()
        saver.save_frame(_frame_id=1, frame=frame_data)

    # Terminates the live encoder before the class is garbage-collected. This also finalizes video saving.
    saver.terminate_live_encoder()

    # Verifies that calling the terminator method twice does nothing.
    saver.terminate_live_encoder(timeout=1)
    assert saver._ffmpeg_process is None

    # Ensures that the frames were saved as a video file.
    assert output_path.joinpath("TestID.mp4").exists()


def test_video_saver_save_frame_errors(tmp_path):
    """Verifies the error handling of the VideoSaver save_frame() and create_live_video_encoder() methods."""

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=False, fps=1000, width=400, height=400)
    saver = VideoSaver(
        output_directory=tmp_path,
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    # Generates a test frame
    camera.connect()
    frame = camera.grab_frame()

    # Verifies that saving a frame without an active live encoder correctly raises an error.
    message = (
        f"Unable to submit the frame to a 'live' FFMPEG encoder process as the process does not exist. Call "
        f"create_live_video_encoder() method to create a 'live' encoder before calling save_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.save_frame(_frame_id=123, frame=frame)

    # Initializes a live encoder process
    saver.create_live_video_encoder(video_id="2", frame_width=400, frame_height=400, video_frames_per_second=45)

    # Ensures that only one live encoder can be active at a time.
    message = (
        f"Unable to create live video encoder for video {2}. FFMPEG process already exists and a "
        f"video saver class can have at most one 'live' encoder at a time. Call the terminate_live_encoder() "
        f"method to terminate the existing encoder before creating a new one."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.create_live_video_encoder(video_id="2", frame_width=400, frame_height=400, video_frames_per_second=45)


def test_create_video_from_image_folder(tmp_path):
    """Verifies the functioning of the VideoSaver create_video_from_image_folder() method."""

    # Defines subdirectories used in testing
    video_directory = tmp_path.joinpath("TestVideo")
    images_directory = tmp_path.joinpath("TestImages")

    # Setup
    camera = MockCamera(name="Test Camera", camera_id=1, color=True, fps=1000, width=400, height=400)
    saver = VideoSaver(
        output_directory=video_directory,
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )
    camera.connect()

    # Generates 20 standalone images to be converted to the video file
    image_saver = ImageSaver(output_directory=images_directory, image_format=ImageFormats.PNG)
    for frame_id in range(20):
        frame_data = camera.grab_frame()
        image_saver.save_frame(frame=frame_data, frame_id=str(frame_id))

    # Converts the images to a video file and verifies that the video file exists
    video_id = "TestID"
    saver.create_video_from_image_folder(
        image_directory=images_directory, video_id=video_id, video_frames_per_second=5, cleanup=True
    )
    video_path = video_directory.joinpath(f"{video_id}.{saver._video_format}")
    assert video_path.exists()


def test_create_video_from_image_errors(tmp_path):
    """Verifies the error handling of the VideoSaver create_video_from_image_folder() method."""

    # Setup
    saver = VideoSaver(
        output_directory=tmp_path.joinpath("Video"),
        hardware_encoding=False,
        video_format=VideoFormats.MP4,
        video_codec=VideoCodecs.H265,
        input_pixel_format=InputPixelFormats.BGRA,
    )

    # Ensures that trying to convert an empty folder into a video correctly raises an error.
    video_id = "TestID"
    empty_folder = tmp_path.joinpath("EmptyFolder")
    ensure_directory_exists(empty_folder)
    message = (
        f"Unable to create video {video_id} from images. No valid image candidates discovered when crawling "
        f"the image directory ({empty_folder}). Valid candidates are images using one of the supported "
        f"file-extensions ({sorted(saver._supported_image_formats)}) with digit-convertible names (e.g: 0001.jpg)."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        saver.create_video_from_image_folder(video_frames_per_second=5, image_directory=empty_folder, video_id=video_id)
