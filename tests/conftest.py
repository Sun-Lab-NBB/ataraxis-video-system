"""Stores fixtures used by other tests. Defining these fixtures at the session scope allows running the associated
test once for each test session.
"""

from pathlib import Path
import subprocess

import pytest

# Global cache for hardware checks. This allows checking the hardware configuration once, before running any tests
_cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
_has_opencv = None
_has_harvesters = None
_has_nvidia = None


def _check_opencv():
    """Static check for OpenCV camera availability."""
    try:
        from ataraxis_video_system import VideoSystem  # Import here instead of at module level

        opencv_id = VideoSystem.get_opencv_ids()
        return len(opencv_id) > 0
    except:
        return False


def _check_harvesters():
    """Static check for Harvesters camera availability."""
    if not _cti_path.exists():
        return False

    try:
        from ataraxis_video_system import VideoSystem  # Import here instead of at module level

        harvesters_id = VideoSystem.get_harvesters_ids(_cti_path)
        return len(harvesters_id) > 0
    except:
        return False


def _check_nvidia():
    """Static check for NVIDIA GPU availability."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def pytest_configure():
    """Runs all hardware checks once before any tests start."""
    global _has_opencv, _has_harvesters, _has_nvidia

    _has_opencv = _check_opencv()
    _has_harvesters = _check_harvesters()
    _has_nvidia = _check_nvidia()


@pytest.fixture(scope="session")
def cti_path():
    """Provides the CTI file path."""
    return _cti_path


@pytest.fixture(scope="session")
def has_opencv():
    """Returns true if the host-system is equipped with an OpenCV-compatible camera."""
    return _has_opencv


@pytest.fixture(scope="session")
def has_harvesters():
    """Returns true if the host-system is equipped with a GeniCam-compatible camera (Harvesters camera)."""
    return _has_harvesters


@pytest.fixture(scope="session")
def has_nvidia():
    """Returns true if the host-system is equipped with an NVIDIA GPU."""
    return _has_nvidia
