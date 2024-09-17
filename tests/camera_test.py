"""Contains tests for classes and methods stored inside the camera module."""

import re
import textwrap

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

from ataraxis_video_system.camera import MockCamera, OpenCVCamera, HarvestersCamera


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


@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
    ],
)
def test_mock_camera_init(color, fps, width, height) -> None:
    """Verifies Mock camera initialization under different conditions."""
    camera = MockCamera(name="Test Camera", camera_id=1, color=color, fps=fps, width=width, height=height)
    assert camera.width == width
    assert camera.height == height
    assert camera.fps == fps
    assert camera.name == "Test Camera"


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 640, 480),
        (False, 30, 1280, 720),
    ],
)
def test_openCV_camera_init(color, fps, width, height) -> None:
    """Verifies Mock camera initialization under different conditions."""
    camera = OpenCVCamera(name="Test Camera", camera_id=0, color=color, fps=fps, width=width, height=height)

    camera.connect()

    assert camera.fps == fps
    assert camera.width == width
    assert camera.height == height

    assert camera.name == "Test Camera"


def test_mock_connect_disconnect():
    camera = MockCamera()

    """Verifies that the camera is not connected when initialized"""
    assert not camera.is_connected

    """Verifies that the camera should be connected and disconnected when the connect() and disconnect() modules are called accordingly"""
    camera.connect()
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected


def test_mock_acquisition():
    camera = MockCamera()

    """ Verifies that the camera should not be acquiring images when initialized"""
    assert not camera.is_acquiring

    camera._acquiring = True
    assert camera.is_acquiring


def test_OpenCV_connect_disconnect():
    camera = OpenCVCamera(name="Test camera")

    """Verifies that the camera is not connected when initialized"""
    assert not camera.is_connected

    """Verifies that the camera should be connected and disconnected when the connect() and disconnect() modules are called accordingly"""
    camera.connect()
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected


def test_OpenCV_acquisition():
    camera = OpenCVCamera(name="Test camera")

    """Verifies that the camera should not be acquiring images when initialized"""
    assert not camera.is_acquiring

    camera._acquiring = True
    assert camera.is_acquiring


def test_mock_camera_grab_frame_errors() -> None:
    """Verifies the grabbing frames before the camera is connected produces a RuntimeError"""
    camera = MockCamera(name="Test Camera", camera_id=1)

    message = (
        f"The Mocked camera {camera._name} with id {camera._camera_id} is not 'connected' and cannot yield images."
        f"Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    """Verifies that timer class is used to force block in-place behaviour if the frame is not available to main a certain FPS rate"""
    assert camera._timer.elapsed >= camera._time_between_frames


def test_mock_camera_acquirenextframe():
    camera = MockCamera(name="Test Camera", camera_id=-1)

    camera.connect()

    """Verifies that initial frame index is 0"""
    assert camera._current_frame_index == 0


def test_OpenCV_repr():
    camera = OpenCVCamera(name="Test Camera", camera_id=0)

    camera.connect()
    camera._acquiring = True

    """Verifies that a string representation of the OpenCVCamera object is returned """
    representation_string = (
        f"OpenCVCamera(name={camera._name}, camera_id={camera._camera_id}, fps={camera.fps}, width={camera.width}, "
        f"height={camera.height}, connected={camera._camera is not None}, acquiring={camera._acquiring}, "
        f"backend = {camera.backend})"
    )

    assert repr(camera) == representation_string


def test_OpenCV_backendname():
    """Verifies that names of camera corresponding to their backend code are returned. Backend codes obtained by the VideoCapture get() method."""

    camera = OpenCVCamera(name="Test Camera", camera_id=-1)
    assert camera.backend == "Any"


def test_OpenCV_camera_grab_frame_errors() -> None:
    camera = OpenCVCamera(name="Test Camera", camera_id=-1)  # Uses invalid ID -1

    camera._backend = -10

    """Verifies that all OpenCV cameras connected have a valid backend code"""
    message = (
        f"Unknown backend code {camera._backend} encountered when retrieving the backend name used by the "
        f"OpenCV-managed {camera._name} camera with id {camera._camera_id}. Recognized backend codes are: "
        f"{(camera._backends.values())}"
    )
    with pytest.raises(ValueError, match=(error_format(message))):
        _ = camera.backend

    """Verifies that grabbing frames before the camera is connected produces a RuntimeError"""
    message = (
        f"The OpenCV-managed camera {camera._name} with id {camera._camera_id} is not 'connected', and cannot yield images."
        f"Call the connect() method of the class prior to calling the grab_frame() method."
    )

    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    camera.connect()

    """ Verifies that the camera should yield images upon connection"""
    message = (
        f"The OpenCV-managed camera {camera._name} with id {camera._camera_id} did not yield an image, "
        f"which is not expected. This may indicate initialization or connectivity issues."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 640, 480),
        (False, 30, 1280, 720),
    ],
)
def test_OpenCV_camera_grab_frame(color, fps, width, height) -> None:
    camera = OpenCVCamera(name="Test Camera", camera_id=0, color=color, fps=fps, width=width, height=height)

    camera.connect()

    """Verifies that the camera VideoCapture object is initialized and video acquisition parameters are set"""
    frame = camera.grab_frame()
    assert camera.name == "Test Camera"
    assert frame.shape[0] == height
    assert frame.shape[1] == width

    if color:
        assert frame.shape[2] > 1

    else:
        assert len(frame.shape) == 2


def test_grab_frame_pool():
    camera = MockCamera(name="Test Camera", camera_id=-1, color=False, width=2, height=3)

    camera.connect()

    """Verifies that the tuple that stores the frames are pooled to produce images during grab_frame() runtime"""
    frame_pool = camera.frame_pool

    for _ in range(11):
        frame = camera.grab_frame()

        for num, image in enumerate(frame_pool, start=1):
            if np.array_equal(image, frame):
                break
            elif num == len(frame_pool):
                raise Exception("No match")


# def test_disconnect():

#     cti_path = "" "Fill in with actual cti path"
#     camera = HarvestersCamera(name="Test camera", cti_path=cti_path)

#     # Act: Call disconnect
#     camera.disconnect()

#     # Assert: Verify that stop() was called
#     # assert camera.stop()


#     assert camera.num_holding_filled_buffers == "0"

#     assert camera.destroy()

#     assert camera._camera is None

@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
        (False, None, None, None)
    ],
)
def test_Harvester_init(fps, width, height) -> None:
    """Verifies Mock camera initialization under different conditions."""
    camera = MockCamera(name="Test Camera", camera_id=1, color=color, fps=fps, width=width, height=height)
    assert camera.width == width
    assert camera.height == height
    assert camera.fps == fps
    assert camera.name == "Test Camera"


def test_Harvester_disconnect():
    """Verifies that the Harvester camera is disconnected upon garbage collection"""

    cti_path = "" "Fill in with actual cti path"

    camera = HarvestersCamera(name="Test camera", cti_path=cti_path)

    camera.connect()

    camera.disconnect()

    assert not camera.is_connected
    assert camera._harvester.reset() is None


def test_Harvester_acquisition():
    cti_path = ""

    camera = HarvestersCamera(name="Test camera", cti_path=cti_path, camera_id=0)

    """Verifies that the program returns True if the camera is acquiring video frames"""
    assert not camera.is_acquiring

    camera.connect()
    camera.grab_frame()

    assert camera.is_acquiring


def test_Harvester_repr():
    cti_path = ""

    camera = HarvestersCamera(name="Test camera", cti_path=cti_path, camera_id=0)

    """Verifies that a string representation of the OpenCVCamera object is returned """
    representation_string = (
        f"HarvestersCamera(name={camera._name}, camera_id={camera._camera_id}, fps={camera.fps}, width={camera.width}, "
        f"height={camera.height}, connected={camera._camera is not None}, acquiring={camera.is_acquiring})"
    )

    assert repr(camera) == representation_string



def test_Harvesters_camera_grab_frame_errors() -> None:
    cti_path = ""

    camera = HarvestersCamera(name="Test camera", cti_path=cti_path, camera_id=0)

    data_format = [rgb_formats, rgba_formats, bgr_formats, bgra_formats]

    message = (
        f"The Harvesters-managed camera {camera._name} with id {camera._camera_id} is not connected and cannot "
        f"yield images. Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    """Verifies that the image has a supported color format"""

    camera.connect()

    """ Verifies that the camera should yield images upon connection"""
    message = (
        f"The Harvesters-managed camera {camera._name} with id {camera._camera_id} did not yield an image, "
        f"which is not expected. This may indicate initialization or connectivity issues."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    message = (
        f"The Harvesters-managed camera {camera._name} with id {camera._camera_id} yielded an image "
        f"with an unsupported data (color) format {data_format}. If possible, re-configure the "
        f"camera to use one of the supported formats: Monochrome, RGB, RGBA, BGR, BGRA. "
        f"Otherwise, you may need to implement a custom data reshaper algorithm."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()
