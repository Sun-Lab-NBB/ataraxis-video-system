"""Contains tests for classes and methods stored inside the camera module."""

import re
import textwrap

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


def test_mock_camera_attributes():
    pass


def test_connect_disconnect():
    camera = MockCamera()

    # Verifies that the camera is not connected when initialized
    assert not camera.is_connected

    # Verifies that the camera should be connected and disconnected when the connect() and disconnect() modules are called accordingly
    camera.connect()
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected

    # # Verifies that the camera should not be acquiring images when initialized
    # assert not self.camera.is_acquiring==("Camera should not be acquiring initially",)

    # camera._acquiring = True
    # assert self.camera.is_acquiring == ("Camera should be acquiring when _acquiring is set to True",)


def test_mock_camera_grab_frame_errors() -> None:
    """Verifies the error-handling behavior of MockCamera grab_frame() method."""
    camera = MockCamera(name="Test Camera", camera_id=1)

    # Verifies that grabbing frames before the camera is connected produces a RuntimeError.
    message = (
        f"The Mocked camera {camera._name} with id {camera._camera_id} is not 'connected' and cannot yield images."
        f"Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()



def test_OpenCV_camera_grab_frame_errors() -> None:
    camera = OpenCVCamera(name="Test Camera", camera_id=-1)  # Uses invalid ID -1

    backend_code = camera._backend

    # Verifies that the backend name of the camera has a recognized backend code

    # message = (
    #     f"Unknown backend code {backend_code} encountered when retrieving the backend name used by the "
    #     f"OpenCV-managed {camera._name} camera with id {camera._camera_id}. Recognized backend codes are: "
    #     f"{(camera._backends.values())}"
    # )
    # with pytest.raises(ValueError, match=(error_format(message))):
    #     _ = camera.grab_frame()

    # Verifies that grabbing frames before the camera is connected produces a RuntimeError.
    message = (
        f"The OpenCV-managed camera {camera._name} with id {camera._camera_id} is not 'connected', and cannot yield images."
        f"Call the connect() method of the class prior to calling the grab_frame() method."
    )

    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    camera.connect()

    # Verifies that the camera should yield images upon connection 
    message = (
        f"The OpenCV-managed camera {camera._name} with id {camera._camera_id} did not yield an image, "
        f"which is not expected. This may indicate initialization or connectivity issues."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()


def test_Harvesters_camera_grab_frame_errors() -> None:
    pass

    

