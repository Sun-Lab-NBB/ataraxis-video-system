"""Contains tests for classes and methods stored inside the camera module."""

import re
import textwrap

import pytest

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


@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
    ],
)
def test_mock_camera_init_errors(color, fps, width, height) -> None:
    """Verifies Mock camera initialization under different conditions."""
    camera = MockCamera(name="Test Camera", camera_id=1, color=color, fps=fps, width=width, height=height)
    assert camera.width == width
    assert camera.height == height
    assert camera.fps == fps
    assert camera.name == "Test Camera"
