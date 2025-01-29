"""Contains tests for classes and methods provided by the camera.py module."""

from pathlib import Path

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_video_system import VideoSystem
from ataraxis_video_system.camera import MockCamera, OpenCVCamera, HarvestersCamera


@pytest.fixture(scope="session")
def cti_path():
    _cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    return _cti_path


@pytest.fixture(scope="session")
def has_opencv():
    """Static check for OpenCV camera availability."""
    try:
        opencv_id = VideoSystem.get_opencv_ids()
        if len(opencv_id) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.fixture(scope="session")
def has_harvesters(cti_path):
    """Static check for Harvesters camera availability."""
    if not cti_path.exists():
        return False

    try:
        harvesters_id = VideoSystem.get_harvesters_ids(cti_path)
        if len(harvesters_id) > 0:
            return True
        else:
            return False
    except:
        return False


@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 600, 400),
        (False, 60, 1200, 1200),
        (False, 10, 3000, 3000),
    ],
)
def test_mock_camera_init(color, fps, width, height) -> None:
    """Verifies the functioning of the MockCamera __init__() method."""
    camera = MockCamera(camera_id=np.uint8(222), camera_index=1, color=color, fps=fps, width=width, height=height)
    assert camera.width == width
    assert camera.height == height
    assert camera.fps == fps
    assert camera.camera_id == np.uint8(222)
    assert not camera.is_acquiring
    assert not camera.is_connected


def test_mock_camera_connect_disconnect():
    """Verifies the functioning of the MockCamera connect() and disconnect() methods."""
    # Setup
    camera = MockCamera()  # Uses default parameters

    # Verifies camera connection
    camera.connect()
    assert camera.is_connected

    # Verifies camera disconnection
    camera.disconnect()
    assert not camera.is_connected


def test_mock_camera_grab_frame():
    """Verifies the functioning of the MockCamera grab_frame() method."""
    # Setup
    camera = MockCamera(camera_id=np.uint8(222), camera_index=-1, color=False, width=2, height=3)
    camera.connect()

    # Accesses the frame pool generated at class initialization. All 'grabbed' frames are sampled from the frame pool.
    frame_pool = camera.frame_pool

    # Acquires 11 frames. Note, the code below will STOP working unless the tested number of frames is below 20.
    for num in range(11):
        frame = camera.grab_frame()  # Grabs the frame from the precreated frame-pool

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
    camera = MockCamera(camera_id=np.uint8(222), camera_index=1)

    # Verifies that the camera cannot yield images if it is not connected.
    message = (
        f"The Mocked camera with id {camera._id} is not 'connected' and cannot yield images."
        f"Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_init_repr() -> None:
    """Verifies the functioning of the OpenCVCamera __init__() and __repr__() methods."""
    # Setup
    camera = OpenCVCamera(camera_id=np.uint8(222), camera_index=0, color=True, fps=100, width=500, height=500)

    # Verifies initial camera parameters
    assert camera.fps == 100
    assert camera.width == 500
    assert camera.height == 500
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera.camera_id == np.uint8(222)
    assert camera.backend == "Any"

    # Verifies the __repr__() method
    representation_string = (
        f"OpenCVCamera(camera_id={camera._id}, camera_index={camera._camera_index}, fps={camera.fps}, "
        f"width={camera.width}, height={camera.height}, connected={camera.is_connected}, "
        f"acquiring={camera.is_acquiring}, backend = {camera.backend})"
    )
    assert repr(camera) == representation_string


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 640, 480),
        (False, 15, 176, 144),
    ],
)
def test_opencv_camera_connect_disconnect(has_opencv, color, fps, width, height) -> None:
    """Verifies the functioning of the OpenCVCamera connect() and disconnect() methods."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Setup
    camera = OpenCVCamera(camera_id=np.uint8(222), camera_index=0, color=color, fps=fps, width=width, height=height)

    # Tests connect method. Note, this may change the fps, width and height class properties, as the camera may not
    # support the requested parameters and instead set them to the nearest supported values or to default values. The
    # specific behavior depends on each camera. Since this code is tested across many different cameras, and it is hard
    # to predict which cameras will support which settings, we do not formally verify whether parameter assignment has
    # worked.
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # Tests disconnect method
    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "color, fps, width, height",
    [
        (True, 30, 640, 480),
        (False, 30, 1280, 720),
    ],
)
def test_opencv_camera_grab_frame(color, fps, width, height, has_opencv) -> None:
    """Verifies the functioning of the OpenCVCamera grab_frame() method."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Setup
    camera = OpenCVCamera(camera_id=np.uint8(222), camera_index=0, color=color, fps=fps, width=width, height=height)
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
    camera = OpenCVCamera(camera_id=np.uint8(222), camera_index=-1)  # Uses invalid ID -1
    camera._backend = -10  # Also, invalidates the backend code to trigger some errors below.

    # Verifies that retrieving an invalid backend code correctly raises a ValueError
    message = (
        f"Unknown backend code {camera._backend} encountered when retrieving the backend name used by the "
        f"OpenCV-managed camera with id {camera._id}. Recognized backend codes are: "
        f"{(camera._backends.values())}"
    )
    with pytest.raises(ValueError, match=(error_format(message))):
        _ = camera.backend

    # Verifies that calling grab_frame() correctly raises a RuntimeError when the camera is not connected
    message = (
        f"The OpenCV-managed camera with id {camera._id} is not connected, and "
        f"cannot yield images. Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    # Verifies that connecting to an invalid camera ID correctly raises a ValueError when grab_frame() is called for
    # that camera
    camera.connect()
    message = (
        f"The OpenCV-managed camera with id {camera._id} did not yield an image, "
        f"which is not expected. This may indicate initialization or connectivity issues."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_init_repr(has_harvesters, cti_path) -> None:
    """Verifies the functioning of the HarvestersCamera __init__() and __repr__() methods."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup
    camera = HarvestersCamera(
        camera_id=np.uint8(222), camera_index=0, fps=60, width=1000, height=1000, cti_path=cti_path
    )

    # Verifies initial camera parameters
    assert camera.fps == 60
    assert camera.width == 1000
    assert camera.height == 1000
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera.camera_id == np.uint8(222)

    # Verifies the __repr__() method
    representation_string = (
        f"HarvestersCamera(camera_id={camera._id}, camera_index={camera._camera_index}, fps={camera.fps}, "
        f"width={camera.width}, height={camera.height}, connected={camera.is_connected}, "
        f"acquiring={camera.is_acquiring})"
    )
    assert repr(camera) == representation_string


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_connect_disconnect(has_harvesters, cti_path) -> None:
    """Verifies the functioning of the HarvestersCamera connect() and disconnect() methods."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup
    camera = HarvestersCamera(
        camera_id=np.uint8(222), camera_index=0, fps=60, width=1000, height=1000, cti_path=cti_path
    )

    # Tests connect method. Unlike OpenCV camera, if Harvesters camera is unable to set the parameters to the
    # requested values, it will raise an error.
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # Tests disconnect method
    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.xdist_group(name="group2")
@pytest.mark.parametrize(
    "fps, width, height",
    [(30, 600, 400), (60, 1200, 1200), (None, None, None)],
)
def test_harvesters_camera_grab_frame(fps, width, height, has_harvesters, cti_path) -> None:
    """Verifies the functioning of the OpenCVCamera grab_frame() method."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup
    camera = HarvestersCamera(
        camera_id=np.uint8(222), camera_index=0, fps=fps, width=width, height=height, cti_path=cti_path
    )
    camera.connect()

    # Tests grab_frame() method.
    assert not camera.is_acquiring
    frame = camera.grab_frame()
    assert camera.is_acquiring  # Ensures calling grab_frame() switches the camera into acquisition mode

    # Verifies the dimensions of the grabbed frame
    if height is not None and width is not None:
        assert frame.shape[0] == height
        assert frame.shape[1] == width

    # Does not check the color handling, as it is expected that the camera itself is configured to properly handle
    # monochrome / color conversions on-hardware. Also, because we do not have a Harvesters camera that is compatible
    # with color imaging.

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_grab_frame_errors(has_harvesters, cti_path) -> None:
    """Verifies the error handling of the HarvestersCamera grab_frame() method."""
    # Skips the test if Harvesters-compatible hardware is not available.
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GeniCam camera).")

    # Setup
    camera = HarvestersCamera(
        camera_id=np.uint8(222), camera_index=0, fps=60, width=1000, height=1000, cti_path=cti_path
    )

    # Verifies that calling grab_frame() correctly raises a RuntimeError when the camera is not connected
    message = (
        f"The Harvesters-managed camera with id {camera._id} is not connected and cannot "
        f"yield images. Call the connect() method of the class prior to calling the grab_frame() method."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _ = camera.grab_frame()

    # Other GrabFrame errors cannot be readily reproduced under a test environment and are likely not possible to
    # encounter under most real-world conditions.
