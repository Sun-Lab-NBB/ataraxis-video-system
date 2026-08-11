"""Contains tests for classes and methods provided by the camera.py module."""

import cv2
import numpy as np
import pytest
from tests.fake_opencv import FakeVideoCapture, build_capture_factory
from ataraxis_base_utilities import error_format

from ataraxis_video_system import CameraInterfaces, InputPixelFormats
from ataraxis_video_system.video import camera as camera_module
from ataraxis_video_system.video.camera import (
    GENICAM_UNAVAILABLE_REASON,
    MockCamera,
    OpenCVCamera,
    HarvestersCamera,
    check_cti_file,
    _get_opencv_ids,
    _get_harvesters_ids,
    discover_camera_ids,
    genicam_runtime_available,
)

_SIMULATED_DEVICE_COUNT: int = 4
"""Stores the number of devices the bundled GenTL Producer simulator exposes."""

_SIMULATED_MONOCHROME_MODEL: str = "TLSimuMono"
"""Stores the model name the bundled GenTL Producer simulator reports for its monochrome devices."""

_SIMULATED_MONOCHROME_SERIAL: str = "SN_InterfaceA_0"
"""Stores the serial number the bundled GenTL Producer simulator reports for the first monochrome device."""

_SIMULATED_COLOR_INDEX: int = 1
"""Stores the discovery index of the first color device exposed by the bundled GenTL Producer simulator."""


@pytest.mark.parametrize(
    ("color", "frame_rate", "frame_width", "frame_height"),
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


def test_mock_camera_connect_disconnect() -> None:
    """Verifies the functioning of the MockCamera connect() and disconnect() methods."""
    camera = MockCamera(system_id=222)  # Uses default parameters

    # Verifies camera connection
    camera.connect()
    assert camera.is_connected

    # Verifies camera disconnection
    camera.disconnect()
    assert not camera.is_connected


def test_mock_camera_grab_frame() -> None:
    """Verifies the functioning of the MockCamera grab_frame() method."""
    camera = MockCamera(system_id=222, color=False, frame_width=2, frame_height=3)
    camera.connect()

    # Accesses the frame pool generated at class initialization. All 'grabbed' frames are sampled from the frame pool.
    frame_pool = camera.frame_pool

    # Acquires 11 frames.
    for frame_number in range(11):
        frame = camera.grab_frame()

        # Currently, the frame pool consists of 10 images. To optimize grabbed image verification, ensures that the
        # index is always within the range of the frame pool and follows the behavior of the grabber that treats the
        # pool as a circular buffer. So, when it reaches '10' (maximum index is 9), it wraps to 0.
        pool_index = frame_number % 10

        # Verifies that the grabbed frame matches expectation
        assert np.array_equal(frame_pool[pool_index], frame)


def test_mock_camera_grab_frame_errors() -> None:
    """Verifies the error handling of the MockCamera grab_frame() method."""
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
    # Setup - uses parameters that are NOT applied to hardware (no connect() call).
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
def test_opencv_camera_connect_disconnect(has_opencv, color) -> None:
    """Verifies the functioning of the OpenCVCamera connect() and disconnect() methods."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

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
def test_opencv_camera_grab_frame(has_opencv, color) -> None:
    """Verifies the functioning of the OpenCVCamera grab_frame() method."""
    # Skips the test if OpenCV-compatible hardware is not available.
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        color=color,
    )
    camera.connect()

    # Tests grab_frame() method.
    assert not camera.is_acquiring
    frame = camera.grab_frame()
    # Ensures calling grab_frame() switches the camera into acquisition mode.
    assert camera.is_acquiring

    # Ensures that acquiring colored frames correctly returns a multidimensional numpy array
    if color:
        assert frame.shape[2] > 1
    else:
        # For monochrome frames, ensures that the returned frame array does not contain color dimensions.
        assert len(frame.shape) == 2

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_connect_with_params(has_opencv) -> None:
    """Verifies that OpenCVCamera connect() applies explicit frame_rate, frame_width, and frame_height parameters."""
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    # Discovers the default camera properties first.
    camera_defaults = OpenCVCamera(system_id=222, camera_index=0)
    camera_defaults.connect()
    default_rate = camera_defaults.frame_rate
    default_width = camera_defaults.frame_width
    default_height = camera_defaults.frame_height
    camera_defaults.disconnect()

    # Reconnects with the same parameters explicitly to exercise the parameter-setting branches.
    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        frame_rate=default_rate,
        frame_width=default_width,
        frame_height=default_height,
    )
    camera.connect()

    # Verifies that the camera accepted the explicitly provided parameters.
    assert camera.frame_rate > 0
    assert camera.frame_width > 0
    assert camera.frame_height > 0

    camera.disconnect()


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_pixel_color_format() -> None:
    """Verifies the pixel_color_format property of OpenCVCamera for both color and monochrome modes."""
    # Tests color mode. No connect() needed — property reads an init-time attribute.
    camera_color = OpenCVCamera(system_id=222, camera_index=0, color=True)
    assert camera_color.pixel_color_format == InputPixelFormats.BGR

    # Tests monochrome mode.
    camera_mono = OpenCVCamera(system_id=222, camera_index=0, color=False)
    assert camera_mono.pixel_color_format == InputPixelFormats.MONOCHROME


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_grab_frame_errors() -> None:
    """Verifies the error handling of the OpenCVCamera grab_frame() method."""
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


@pytest.mark.parametrize(
    ("property_id", "requested", "reported", "argument"),
    [
        (cv2.CAP_PROP_FPS, 15, 30.0, "frame_rate"),
        (cv2.CAP_PROP_FRAME_WIDTH, 320, 640.0, "frame_width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, 240, 480.0, "frame_height"),
    ],
)
def test_opencv_camera_connect_rejects_a_substituted_parameter(
    monkeypatch, property_id, requested, reported, argument
) -> None:
    """Verifies that connect() fails when the camera substitutes its own value for a requested parameter."""
    capture = FakeVideoCapture(properties={property_id: reported}, accepts_writes=False)
    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: capture}))

    camera = OpenCVCamera(system_id=222, camera_index=0, **{argument: requested})

    # The library performs no software decimation or scaling, so a camera acquiring at anything other than the
    # requested parameters would leave the recording stamped with values it does not hold.
    with pytest.raises(ValueError, match="Unable to configure the OpenCVCamera interface"):
        camera.connect()

    # The write was attempted before the readback rejected the substituted value, which is what separates a camera
    # that refuses the parameter from one that was never asked for it.
    assert (property_id, float(requested)) in capture.set_calls


def test_opencv_camera_connect_accepts_a_rounded_frame_rate(monkeypatch) -> None:
    """Verifies that a camera reporting the requested rate as a near-integer float is accepted rather than rejected."""
    # A 30 fps camera commonly reports 29.97, which truncation would read as 29 and reject.
    capture = FakeVideoCapture(properties={cv2.CAP_PROP_FPS: 29.97}, accepts_writes=False)
    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: capture}))

    camera = OpenCVCamera(system_id=222, camera_index=0, frame_rate=30)
    camera.connect()

    assert camera.frame_rate == 30


def test_opencv_camera_grab_frame_converts_to_monochrome(monkeypatch) -> None:
    """Verifies that a monochrome camera reduces the three-channel frame OpenCV returns to a single channel."""
    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: FakeVideoCapture()}))

    camera = OpenCVCamera(system_id=222, camera_index=0, color=False)
    camera.connect()

    frame = camera.grab_frame()

    assert frame.ndim == 2
    assert frame.shape == (480, 640)
    assert camera.pixel_color_format == InputPixelFormats.MONOCHROME


def test_get_opencv_ids_reports_every_working_index(monkeypatch) -> None:
    """Verifies that OpenCV discovery reports one entry per working index and stops after five idle indices."""
    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: FakeVideoCapture()}))

    cameras = _get_opencv_ids()

    assert len(cameras) == 1
    assert cameras[0].camera_index == 0
    assert cameras[0].interface == CameraInterfaces.OPENCV
    assert cameras[0].frame_width == 640
    assert cameras[0].frame_height == 480
    assert cameras[0].acquisition_frame_rate == 30


def test_get_opencv_ids_collapses_duplicate_device_nodes(monkeypatch) -> None:
    """Verifies that two consecutive indices reporting identical properties collapse into one physical camera."""
    # V4L2 routinely exposes one USB camera as two consecutive device nodes, only one of which can stream at a time.
    first = FakeVideoCapture()
    duplicate = FakeVideoCapture(readable=False)

    monkeypatch.setattr(cv2, "VideoCapture", build_capture_factory(captures={0: first, 1: duplicate}))

    # The duplicate answers reads while it is probed on its own, and refuses them while the first node is held open,
    # since one physical camera streams to one capture at a time.
    duplicate.readable = True
    original_read = duplicate.read

    def _read_unless_sibling_is_held():
        """Refuses the read while the sibling node is streaming, the way a duplicate device node does."""
        if first.released:
            return original_read()
        return False, None

    monkeypatch.setattr(duplicate, "read", _read_unless_sibling_is_held)

    cameras = _get_opencv_ids()

    assert [camera.camera_index for camera in cameras] == [0]


def test_harvesters_camera_init_repr() -> None:
    """Verifies the functioning of the HarvestersCamera __init__() and __repr__() methods."""
    # Construction resolves no GenTL Producer, so this test needs neither hardware nor the simulator.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10, frame_width=200, frame_height=200)

    # Verifies initial camera parameters
    assert camera.frame_rate == 10
    assert camera.frame_width == 200
    assert camera.frame_height == 200
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


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_disconnect() -> None:
    """Verifies the functioning of the HarvestersCamera connect() and disconnect() methods."""
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_width=200, frame_height=200)

    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # The simulator reports the identity of the device the instance connected to.
    assert camera.model == _SIMULATED_MONOCHROME_MODEL
    assert camera.serial_number == _SIMULATED_MONOCHROME_SERIAL

    # Tests disconnect method
    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.usefixtures("gentl_simulator")
def test_get_harvesters_ids() -> None:
    """Verifies that Harvesters discovery reports every device the configured GenTL Producer exposes."""
    cameras = _get_harvesters_ids()

    assert len(cameras) == _SIMULATED_DEVICE_COUNT
    assert all(camera.interface == CameraInterfaces.HARVESTERS for camera in cameras)
    assert [camera.camera_index for camera in cameras] == list(range(_SIMULATED_DEVICE_COUNT))
    assert cameras[0].model == _SIMULATED_MONOCHROME_MODEL
    assert cameras[0].serial_number == _SIMULATED_MONOCHROME_SERIAL
    assert all(camera.frame_width > 0 for camera in cameras)
    assert all(camera.frame_height > 0 for camera in cameras)

    # The simulated devices omit the optional AcquisitionFrameRate feature, so discovery reports no rate for them.
    assert all(camera.acquisition_frame_rate == 0 for camera in cameras)


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_missing_frame_rate_node() -> None:
    """Verifies that connecting to a camera without an AcquisitionFrameRate node retains the requested rate."""
    # The simulated devices do not implement the optional AcquisitionFrameRate feature.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10)
    camera.connect()
    try:
        assert camera.frame_rate == 10
    finally:
        camera.disconnect()

    # Without a requested rate there is no value to fall back to, so the camera reports no rate at all.
    default_camera = HarvestersCamera(system_id=222, camera_index=0)
    default_camera.connect()
    try:
        assert default_camera.frame_rate == 0
    finally:
        default_camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_rounds_the_frame_rate_node(monkeypatch) -> None:
    """Verifies that a camera implementing AcquisitionFrameRate reports the node's value rounded, not truncated."""

    class _FrameRateNode:
        """Stands in for the optional AcquisitionFrameRate node the bundled simulator does not implement."""

        def __init__(self) -> None:
            self._value = 30.0

        @property
        def value(self) -> float:
            """Returns the rate the node currently holds."""
            return self._value

        @value.setter
        def value(self, new_value: float) -> None:
            """Clamps the written rate the way a real float node does, landing just below the requested value."""
            self._value = float(new_value) - 0.000076

    node = _FrameRateNode()

    def _resolve_node(node_map):
        """Returns the stand-in node in place of the feature the simulated devices omit."""
        assert node_map is not None
        return node

    monkeypatch.setattr(camera_module, "_get_frame_rate_node", _resolve_node)

    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=30)
    camera.connect()
    try:
        # Truncating the clamped readback would report 29, which is the value discovery already avoids by rounding,
        # and would leave the two entry points describing one device with two different rates.
        assert camera.frame_rate == 30
    finally:
        camera.disconnect()


@pytest.mark.usefixtures("gentl_simulator")
@pytest.mark.parametrize(
    ("frame_width", "frame_height"),
    [(440, 440), (200, 200), (None, None)],
)
def test_harvesters_camera_grab_frame(frame_width, frame_height) -> None:
    """Verifies the functioning of the HarvestersCamera grab_frame() method."""
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_width=frame_width, frame_height=frame_height)
    camera.connect()

    # Tests grab_frame() method.
    assert not camera.is_acquiring
    frame = camera.grab_frame()
    # Ensures calling grab_frame() switches the camera into acquisition mode.
    assert camera.is_acquiring

    # Verifies the dimensions of the grabbed frame
    if frame_height is not None and frame_width is not None:
        assert frame.shape[0] == frame_height
        assert frame.shape[1] == frame_width

    # The monochrome simulated device yields single-channel frames.
    assert frame.ndim == 2

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_grab_frame_color() -> None:
    """Verifies that HarvestersCamera converts frames from a color device into three-channel BGR arrays."""
    camera = HarvestersCamera(system_id=222, camera_index=_SIMULATED_COLOR_INDEX, frame_width=200, frame_height=200)
    camera.connect()
    try:
        frame = camera.grab_frame()
        assert frame.shape == (200, 200, 3)
        assert camera.pixel_color_format == InputPixelFormats.BGR
    finally:
        camera.disconnect()


def test_harvesters_camera_grab_frame_errors() -> None:
    """Verifies the error handling of the HarvestersCamera grab_frame() method."""
    # The guard under test runs before any GenTL Producer is resolved, so no camera source is required.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10, frame_width=200, frame_height=200)

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


@pytest.mark.usefixtures("gentl_simulator")
@pytest.mark.parametrize(
    ("camera_index", "expected_format"),
    [(0, InputPixelFormats.MONOCHROME), (_SIMULATED_COLOR_INDEX, InputPixelFormats.BGR)],
)
def test_harvesters_camera_pixel_color_format(camera_index, expected_format) -> None:
    """Verifies that pixel_color_format reflects the data format of the connected device."""
    camera = HarvestersCamera(system_id=222, camera_index=camera_index)
    camera.connect()
    try:
        # Grabs a frame to trigger pixel format detection.
        camera.grab_frame()
        assert camera.pixel_color_format == expected_format
    finally:
        camera.disconnect()


@pytest.mark.xdist_group(name="group2")
def test_harvesters_camera_hardware_acquisition(has_harvesters) -> None:
    """Verifies frame rate negotiation and acquisition against real GenICam hardware."""
    if not has_harvesters:
        pytest.skip("Skipping this test as it requires a Harvesters-compatible camera (GenICam camera).")

    # Frame rate negotiation depends on the optional AcquisitionFrameRate feature, which the bundled GenTL Producer
    # simulator does not implement, so it is only reachable through real hardware.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10)
    camera.connect()
    try:
        assert camera.frame_rate > 0
        assert camera.frame_width > 0
        assert camera.frame_height > 0

        frame = camera.grab_frame()
        assert camera.is_acquiring
        assert frame.shape[0] == camera.frame_height
        assert frame.shape[1] == camera.frame_width
    finally:
        camera.disconnect()


def test_harvesters_camera_pixel_color_format_both_branches() -> None:
    """Verifies the pixel_color_format property for both _color=True and _color=False states."""
    # Tests both branches of pixel_color_format by directly setting the _color attribute. No hardware
    # interaction is needed since the property simply reads the attribute value.
    camera = HarvestersCamera(system_id=222, camera_index=0)

    camera._color = True
    assert camera.pixel_color_format == InputPixelFormats.BGR

    camera._color = False
    assert camera.pixel_color_format == InputPixelFormats.MONOCHROME


def test_genicam_runtime_available_tracks_the_imported_runtime(monkeypatch) -> None:
    """Verifies that genicam_runtime_available() reports whether the GenICam runtime imported."""
    # Patches both states rather than reading the host's own, so the assertions hold on macOS too, where the library
    # installs no runtime and the guarded import falls back to None.
    monkeypatch.setattr(camera_module, "Harvester", object())
    assert genicam_runtime_available()

    monkeypatch.setattr(camera_module, "Harvester", None)
    assert not genicam_runtime_available()


def test_harvesters_camera_connect_requires_the_genicam_runtime(monkeypatch) -> None:
    """Verifies that connecting to a GenICam camera aborts on a platform that does not support the interface."""
    monkeypatch.setattr(camera_module, "Harvester", None)
    camera = HarvestersCamera(system_id=222, camera_index=3)

    # Reuses the module's own explanation, which is resolved from the host platform and therefore differs per platform.
    message = f"Unable to connect to the GenICam camera at index 3. {GENICAM_UNAVAILABLE_REASON}"
    with pytest.raises(NotImplementedError, match=error_format(message)):
        camera.connect()


def test_discover_camera_ids_skips_genicam_without_the_runtime(monkeypatch) -> None:
    """Verifies that camera discovery reports OpenCV cameras alone where the GenICam interface is unsupported."""

    def _forbidden_discovery():
        message = "Harvesters discovery ran without the GenICam runtime."
        raise AssertionError(message)

    def _no_opencv_cameras():
        return ()

    monkeypatch.setattr(camera_module, "Harvester", None)
    monkeypatch.setattr(camera_module, "_get_harvesters_ids", _forbidden_discovery)
    monkeypatch.setattr(camera_module, "_get_opencv_ids", _no_opencv_cameras)

    assert discover_camera_ids() == ()


def test_check_cti_file_reports_an_unsupported_platform(monkeypatch) -> None:
    """Verifies that check_cti_file() reports an unusable configuration where the GenICam interface is unsupported."""
    monkeypatch.setattr(camera_module, "Harvester", None)

    assert check_cti_file() is None
