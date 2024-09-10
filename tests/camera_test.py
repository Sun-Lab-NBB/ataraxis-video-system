"""Contains tests for classes and methods stored inside the camera module."""

import re
import textwrap

import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
    ],
)
def test_openCV_camera_init(color, fps, width, height) -> None:
    """Verifies Mock camera initialization under different conditions."""
    camera = OpenCVCamera(name="Test Camera", camera_id=1, color=color, fps=fps, width=width, height=height)
    assert camera.width == width
    assert camera.height == height
    assert camera.fps == fps
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


""" Verifies that the camera should not be acquiring images when initialized"""


def test_mock_acquisition():
    camera = MockCamera()

    assert not camera.is_acquiring

    camera._acquiring = True
    assert camera.is_acquiring


"""Verifies that the tuple that stores the frames are pooled to produce images when grab_frame() is called"""
def test_mock_frame_pool():
    pass


def test_openCV_connect_disconnect():
    camera = OpenCVCamera(name="Test camera")

    """Verifies that the camera is not connected when initialized"""
    assert not camera.is_connected

    """Verifies that the camera should be connected and disconnected when the connect() and disconnect() modules are called accordingly"""
    camera.connect()
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected


def test_openCV_acquisition():
    camera = OpenCVCamera(name="Test camera")

    """Verifies that the camera should not be acquiring images when initialized"""
    assert not camera.is_acquiring

    camera._acquiring = True
    assert camera.is_acquiring


""" Verifies that grabbing frames before the camera is connected produces a RuntimeError"""
def test_mock_camera_grab_frame_errors() -> None:


    """Verifies the error-handling behavior of MockCamera grab_frame() method."""
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

    # frame = camera.grab_frame()
    # assert np.array_equal(frame, np.array([[0]]))
    # # assert camera._timer.reset is True
    # # assert camera._current_frame_index == 1


"""Verifies that a string representation of the OpenCVCamera object is returned """
def test_OpenCV_repr():
    camera = OpenCVCamera(name="Test Camera", camera_id=0)

    camera.connect()
    camera._acquiring = True

    representation_string = (
        f"OpenCVCamera(name={camera._name}, camera_id={camera._camera_id}, fps={camera.fps}, width={camera.width}, "
        f"height={camera.height}, connected={camera._camera is not None}, acquiring={camera._acquiring}, "
        f"backend = {camera.backend})"
    )

    assert repr(camera) == representation_string


def test_OpenCV_camera_grab_frame_errors() -> None:
    camera = OpenCVCamera(name="Test Camera", camera_id=-1)  # Uses invalid ID -1

    camera._backend = -10

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

    """Verifies that BGR color scheme is converted to monochrome when necessary"""


def test_openCV_backendname():
    camera = OpenCVCamera(name="Test Camera", camera_id=-1)
    assert camera.backend == "Any"


def test_Harvesters_camera_grab_frame_errors() -> None:
    pass
