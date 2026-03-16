"""Provides shared fixtures for all test modules."""

from contextlib import suppress
import subprocess

import pytest

from ataraxis_video_system import (
    CameraInterfaces,
    discover_camera_ids,
    check_ffmpeg_availability,
)
from ataraxis_video_system.camera import HarvestersCamera


def _restore_camera_state(saved_state: dict[str, int]) -> None:
    """Restores critical camera parameters in the correct GenICam order.

    Resets offsets to 0 first to maximize the allowed Width/Height range, then restores dimensions and frame rate.
    Silently skips if the camera is inaccessible.
    """
    with suppress(Exception):
        camera = HarvestersCamera(system_id=222, camera_index=0)
        camera.connect()
        node_map = camera.node_map

        # Resets offsets to 0 to ensure the saved Width/Height values fit within the allowed range.
        with suppress(Exception):
            node_map.OffsetX.value = 0
        with suppress(Exception):
            node_map.OffsetY.value = 0

        node_map.Width.value = saved_state["width"]
        node_map.Height.value = saved_state["height"]
        node_map.AcquisitionFrameRate.value = saved_state["frame_rate"]
        camera.disconnect()


@pytest.fixture(scope="session")
def has_opencv():
    """Checks for OpenCV camera availability in the test environment."""
    try:
        all_cameras = discover_camera_ids()
        return any(cam.interface == CameraInterfaces.OPENCV for cam in all_cameras)
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_harvesters():
    """Checks for Harvesters camera availability and saves camera state for restore at session end.

    Captures the camera's original Width, Height, and AcquisitionFrameRate from the discovery results (without an
    extra connection) so they can be restored after all tests complete. This prevents tests that modify camera
    dimensions from permanently altering the hardware configuration.
    """
    try:
        all_cameras = discover_camera_ids()
        harvesters_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]
        has = bool(harvesters_cameras)
    except Exception:
        has = False
        harvesters_cameras = []

    # Saves the original camera parameters from discovery results. No extra GenTL connection is needed since
    # discover_camera_ids() already reads Width, Height, and AcquisitionFrameRate from the node map.
    saved_state: dict[str, int] | None = None
    if has:
        camera_info = harvesters_cameras[0]
        saved_state = {
            "width": camera_info.frame_width,
            "height": camera_info.frame_height,
            "frame_rate": camera_info.acquisition_frame_rate,
        }

    yield has

    # Restores camera state at session end if it was successfully saved.
    if saved_state is not None:
        _restore_camera_state(saved_state=saved_state)


@pytest.fixture(scope="session")
def has_nvidia():
    """Checks for NVIDIA GPU availability in the test environment."""
    try:
        subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception:
        return False
    else:
        return True


@pytest.fixture(scope="session")
def has_ffmpeg():
    """Checks for FFMPEG availability in the test environment."""
    return check_ffmpeg_availability()
