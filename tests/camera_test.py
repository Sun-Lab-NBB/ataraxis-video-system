"""Contains tests for classes and methods provided by the camera.py module."""

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system.camera import (
    MockCamera,
    OpenCVCamera,
    HarvestersCamera,
    discover_camera_ids,
    CameraInformation,
    CameraInterfaces,
)


@pytest.fixture(scope="session")
def has_opencv():
    """Checks for OpenCV camera availability in the test environment."""
    try:
        all_cameras = discover_camera_ids()
        opencv_ids = [cam for cam in all_cameras if cam.interface == CameraInterfaces.OPENCV]
        if len(opencv_ids) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_harvesters():
    """Checks for Harvesters camera availability in the test environment."""
    try:
        # Attempts to discover Harvesters cameras using the internally stored CTI path
        all_cameras = discover_camera_ids()
        harvesters_ids = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]
        if len(harvesters_ids) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.mark.parametrize(
    "color, frame_rate, frame_width, frame_height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
    ],
)
def test_mock_camera_init(color, frame_rate, frame_width, frame_height) -> None:
    """Verifies the functioning of the MockCamera __init__() method."""
    camera = MockCamera(
        system_id=222, color=color, frame_rate=frame_rate, frame_width=frame_width, frame_height=frame_height
    )
    assert camera.frame_width == frame_width
    assert camera.frame_height == frame_height
    assert camera.frame_rate == frame_rate
    assert camera._system_id == 222
    assert not camera.is_acquiring
    assert not camera.is_connected


def test_mock_camera_connect_disconnect():
    """Verifies the functioning of the MockCamera connect() and disconnect() methods."""
    # Setup
    camera = MockCamera(system_id=222)  # Uses default parameters

    # Verifies camera connection
    camera.connect()
    assert camera.is_connected

    # Verifies camera disconnection
    camera.disconnect()
    assert not camera.is_connected


def test_mock_camera_grab_frame():
    """Verifies the functioning of the MockCamera grab_frame() method."""
    # Setup
    camera = MockCamera(system_id=222, color=False, frame_width=2, frame_height=3)
    camera.connect()

    # Accesses the frame pool generated at class initialization. All 'grabbed' frames are sampled from the frame pool.
    frame_pool = camera.frame_pool

    # Acquires 11 frames. Note, the code below will STOP working unless the tested number of frames is below 20.
    for num in range(11):
        frame = camera.grab_frame()  # Grabs the frame from the pre-created frame-pool

        # Currently, the frame pool consists of 10 images. To optimize grabbed image verification, ensures that 'num' is
        # always within the range of the frame pool and follows the behavior of the grabber that treats the pool as a
        # circular buffer. So, when it reaches '10' (maximum index is 9), it is reset to 0.
        if num == 10:
            num -= 10

        # Verifies that the grabbed frame matches expectation
        assert np.array_equal(frame_pool[num], frame)


def test_mock_camera_grab_frame_errors() -> None:
    """Verifies the error handling of the MockCamera grab_frame() method."""
    # Setup
    camera = MockCamera(system_id=222)

    # Verifies that the camera cannot yield images if it is not connected.
    message = (
        f"The MockCamera instance for the VideoSystem with id {camera._system_id} is not currently simulating "
        f"connection to the camera hardware, and cannot simulate image acquisition. Call the connect() method "
        f"prior to calling the grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_init_repr() -> None:
    """Verifies the functioning of the OpenCVCamera __init__() and __repr__() methods."""
    # Setup
    camera = OpenCVCamera(system_id=222, camera_index=0, color=True, frame_rate=100, frame_width=500, frame_height=500)

    # Verifies initial camera parameters
    assert camera.frame_rate == 100
    assert camera.frame_width == 500
    assert camera.frame_height == 500
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera._system_id == 222

    # Verifies the __repr__() method
    representation_string = (
        f"OpenCVCamera(system_id={camera._system_id}, camera_index={camera._camera_index}, "
        f"frame_rate={camera.frame_rate} frames / second, frame_width={camera.frame_width} pixels, "
        f"frame_height={camera.frame_height} pixels, connected={camera._camera is not None}, "
        f"acquiring={camera._acquiring})"
    )
    assert repr(camera) == representation_string


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color",
    [
        True,
        False,
    ],
)
@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_connect_disconnect(has_opencv, color) -> None:
    """Verifies the functioning of the OpenCVCamera connect() and disconnect() methods."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Setup
    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        color=color,
    )

    # Tests connect method. Note, this may change the frame_rate, frame_width and frame_height class properties, as the
    # camera may not support the requested parameters and instead set them to the nearest supported values or to default
    # values. The specific behavior depends on each camera. Since this code is tested across many different cameras, and
    # it is hard to predict which cameras will support which settings, formal verification of parameter assignment is
    # not performed.
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # Tests disconnect method
    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color",
    [
        True,
        False,
    ],
)
@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_grab_frame(has_opencv, color) -> None:
    """Verifies the functioning of the OpenCVCamera grab_frame() method."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Setup
    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        color=color,
    )
    camera.connect()

    # Tests grab_frame() method.
    assert not camera.is_acquiring
    frame = camera.grab_frame()
    assert camera.is_acquiring  # Ensures calling grab_frame() switches the camera into acquisition mode

    # Ensures that acquiring colored frames correctly returns a multidimensional numpy array
    if color:
        assert frame.shape[2] > 1
    else:
        # For monochrome frames, ensures that the returned frame array does not contain color dimensions.
        assert len(frame.shape) == 2

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_grab_frame_errors() -> None:
    """Verifies the error handling of the OpenCVCamera grab_frame() method."""
    # Setup
    camera = OpenCVCamera(system_id=222, camera_index=333)  # Uses invalid index 333

    # Verifies that calling grab_frame() correctly raises a ConnectionError when the camera is not connected
    message = (
        f"The OpenCVCamera instance for the VideoSystem with id {camera._system_id} is not connected to the "
        f"camera hardware, and cannot acquire images. Call the connect() method prior to calling the "
        f"grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()

    # Verifies that connecting to an invalid camera ID correctly raises a BrokenPipeError when grab_frame() is called
    # for that camera
    camera.connect()
    message = (
        f"The OpenCVCamera instance for the VideoSystem with id {camera._system_id} has failed to grab a frame "
        f"image from the camera hardware, which is not expected. This indicates initialization or connectivity "
        f"issues."
    )
    with pytest.raises(BrokenPipeError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_init_repr(has_harvesters) -> None:
    """Verifies the functioning of the HarvestersCamera __init__() and __repr__() methods."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup - Note that the CTI path is automatically resolved internally by the HarvestersCamera class
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=60, frame_width=1000, frame_height=1000)

    # Verifies initial camera parameters
    assert camera.frame_rate == 60
    assert camera.frame_width == 1000
    assert camera.frame_height == 1000
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera._system_id == 222

    # Verifies the __repr__() method
    representation_string = (
        f"HarvestersCamera(system_id={camera._system_id}, camera_index={camera._camera_index}, "
        f"frame_rate={camera.frame_rate} frames / second, frame_width={camera.frame_width} pixels, "
        f"frame_height={camera.frame_height} pixels, connected={camera._camera is not None}, "
        f"acquiring={camera.is_acquiring})"
    )
    assert repr(camera) == representation_string


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_connect_disconnect(has_harvesters) -> None:
    """Verifies the functioning of the HarvestersCamera connect() and disconnect() methods."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=60, frame_width=1000, frame_height=1000)

    # Tests connect method. Unlike OpenCV camera, if Harvesters camera is unable to set the parameters to the
    # requested values, it may raise an error depending on the camera model.
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # Tests disconnect method
    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.xdist_group(name="group2")
@pytest.mark.parametrize(
    "frame_rate, frame_width, frame_height",
    [(30, 600, 400), (60, 1200, 1200), (None, None, None)],
)
def test_harvesters_camera_grab_frame(has_harvesters, frame_rate, frame_width, frame_height) -> None:
    """Verifies the functioning of the HarvestersCamera grab_frame() method."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup - The library internally manages the CTI path
    camera = HarvestersCamera(
        system_id=222, camera_index=0, frame_rate=frame_rate, frame_width=frame_width, frame_height=frame_height
    )
    camera.connect()

    # Tests grab_frame() method.
    assert not camera.is_acquiring
    frame = camera.grab_frame()
    assert camera.is_acquiring  # Ensures calling grab_frame() switches the camera into acquisition mode

    # Verifies the dimensions of the grabbed frame
    if frame_height is not None and frame_width is not None:
        assert frame.shape[0] == frame_height
        assert frame.shape[1] == frame_width

    # Does not check the color handling, as it is expected that the camera itself is configured to properly handle
    # monochrome / color conversions on-hardware. Also, because Harvesters cameras used in testing may not be
    # compatible with color imaging.

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_grab_frame_errors(has_harvesters) -> None:
    """Verifies the error handling of the HarvestersCamera grab_frame() method."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup - Uses the internally stored CTI path
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=60, frame_width=1000, frame_height=1000)

    # Verifies that calling grab_frame() correctly raises a ConnectionError when the camera is not connected
    message = (
        f"The HarvestersCamera instance for the VideoSystem with id {camera._system_id} is not connected to the "
        f"camera hardware and cannot acquire images. Call the connect() method prior to calling the "
        f"grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()

    # Other GrabFrame errors cannot be readily reproduced under a test environment and are likely not possible to
    # encounter under most real-world conditions.
