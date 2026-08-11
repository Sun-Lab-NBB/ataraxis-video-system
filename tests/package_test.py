"""Contains tests for the import-time runtime configuration provided by the __init__.py module of the library."""

import os
import importlib

import pytest

import ataraxis_video_system


@pytest.mark.xdist_group(name="package_import")
def test_import_forces_the_xcb_qt_platform_on_a_wayland_session(monkeypatch) -> None:
    """Verifies that importing the library on a Wayland session redirects Qt onto the X11 compatibility layer."""
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-probe")

    # Binds the variable through monkeypatch before the import writes to it, so the teardown restores the value the
    # process held before this test. The import mutates the live process environment, which every later test shares.
    monkeypatch.setenv("QT_QPA_PLATFORM", "operator-choice")

    importlib.reload(ataraxis_video_system)

    # The Qt build bundled with OpenCV ships no Wayland plugin, so the live frame-display window only opens when the
    # library redirects Qt onto the X11 compatibility layer.
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


@pytest.mark.xdist_group(name="package_import")
def test_import_preserves_the_qt_platform_outside_a_wayland_session(monkeypatch) -> None:
    """Verifies that importing the library outside a Wayland session leaves the operator's Qt platform untouched."""
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "operator-choice")

    importlib.reload(ataraxis_video_system)

    assert os.environ["QT_QPA_PLATFORM"] == "operator-choice"
