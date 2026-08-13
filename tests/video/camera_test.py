"""Contains tests for classes and methods provided by the camera.py module."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.fake_opencv import FakeVideoCapture, build_capture_factory
from tests.fake_harvesters import FakeImageAcquirer, build_frame_buffer
from ataraxis_base_utilities import error_format

from ataraxis_video_system import CameraInterfaces, InputPixelFormats
from ataraxis_video_system.video import camera as camera_module
from ataraxis_video_system.video.camera import (
    _MAXIMUM_NON_WORKING_IDS,
    GENICAM_UNAVAILABLE_REASON,
    MockCamera,
    OpenCVCamera,
    HarvestersCamera,
    add_cti_file,
    _get_cti_path,
    check_cti_file,
    _get_opencv_ids,
    _get_harvesters_ids,
    discover_camera_ids,
    harvester_connection,
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
def test_mock_camera_init_stores_the_requested_acquisition_parameters(
    color, frame_rate, frame_width, frame_height
) -> None:
    """Verifies that constructing a MockCamera stores the requested geometry, rate, and idle acquisition state."""
    camera = MockCamera(
        system_id=222, color=color, frame_rate=frame_rate, frame_width=frame_width, frame_height=frame_height
    )
    assert camera.frame_width == frame_width
    assert camera.frame_height == frame_height
    assert camera.frame_rate == frame_rate
    assert camera._system_id == 222
    assert not camera.is_acquiring

    # The representation reports the acquisition state alongside the geometry, which is what distinguishes a mock that
    # is standing by from one a VideoSystem has already connected and started.
    assert repr(camera) == (
        f"MockCamera(system_id=222, frame_rate={frame_rate} frames / second, frame_width={frame_width} pixels, "
        f"frame_height={frame_height} pixels, connected=False, acquiring=False)"
    )
    assert not camera.is_connected


def test_mock_camera_connect_and_disconnect_toggle_the_connection_state() -> None:
    """Verifies that connecting and disconnecting a MockCamera toggles the reported connection state."""
    camera = MockCamera(system_id=222)  # Uses default parameters.

    camera.connect()
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected


def test_mock_camera_grab_frame_cycles_through_the_frame_pool() -> None:
    """Verifies that grabbing frames cycles through the pre-generated frame pool as a circular buffer."""
    camera = MockCamera(system_id=222, color=False, frame_width=2, frame_height=3)
    camera.connect()

    # Accesses the frame pool generated at class initialization. All 'grabbed' frames are sampled from the frame pool.
    frame_pool = camera.frame_pool

    for frame_number in range(11):
        frame = camera.grab_frame()

        # Currently, the frame pool consists of 10 images. To optimize grabbed image verification, ensures that the
        # index is always within the range of the frame pool and follows the behavior of the grabber that treats the
        # pool as a circular buffer. So, when it reaches '10' (maximum index is 9), it wraps to 0.
        pool_index = frame_number % 10

        assert np.array_equal(frame_pool[pool_index], frame)


def test_mock_camera_grab_frame_rejects_a_disconnected_interface() -> None:
    """Verifies that grabbing a frame before the simulated connection starts is rejected."""
    camera = MockCamera(system_id=222)

    message = (
        f"Unable to simulate a frame acquisition using the MockCamera interface for the VideoSystem with id "
        f"{camera._system_id}. The interface must be simulating a connection to the camera hardware, but the "
        f"connection simulation has not been started. Call the connect() method prior to calling the "
        f"grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_init_stores_parameters_and_renders_its_repr() -> None:
    """Verifies that constructing an OpenCVCamera stores the requested parameters and renders them in its repr."""
    # Setup - uses parameters that are NOT applied to hardware (no connect() call).
    camera = OpenCVCamera(system_id=222, camera_index=0, color=True, frame_rate=100, frame_width=500, frame_height=500)

    assert camera.frame_rate == 100
    assert camera.frame_width == 500
    assert camera.frame_height == 500
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera._system_id == 222

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
def test_opencv_camera_connect_and_disconnect_toggle_the_connection_state(has_opencv, color) -> None:
    """Verifies that connecting and disconnecting an OpenCVCamera toggles the reported connection state."""
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        color=color,
    )

    # This instance requests none of the three acquisition parameters, so connect() adopts whatever frame rate, width,
    # and height the connected camera reports. A camera that substitutes its own value for an explicitly requested one
    # is rejected with ValueError instead. Since this code is tested across many different cameras whose default values
    # are hard to predict, formal verification of the adopted values is not performed.
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

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
def test_opencv_camera_grab_frame_returns_the_requested_channel_count(has_opencv, color) -> None:
    """Verifies that a grabbed frame carries the channel count the requested color mode implies."""
    if not has_opencv:
        pytest.skip("Skipping this test as it requires an OpenCV-compatible camera.")

    camera = OpenCVCamera(
        system_id=222,
        camera_index=0,
        color=color,
    )
    camera.connect()

    assert not camera.is_acquiring
    frame = camera.grab_frame()
    assert camera.is_acquiring

    # Ensures that acquiring colored frames correctly returns a multidimensional numpy array.
    if color:
        assert frame.shape[2] > 1
    else:
        # For monochrome frames, ensures that the returned frame array does not contain color dimensions.
        assert len(frame.shape) == 2

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_connect_applies_the_requested_parameters(has_opencv) -> None:
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

    assert camera.frame_rate > 0
    assert camera.frame_width > 0
    assert camera.frame_height > 0

    camera.disconnect()


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_pixel_color_format_reports_the_requested_mode() -> None:
    """Verifies the pixel_color_format property of OpenCVCamera for both color and monochrome modes."""
    # The property reads an init-time attribute, so no connect() call is needed.
    camera_color = OpenCVCamera(system_id=222, camera_index=0, color=True)
    assert camera_color.pixel_color_format == InputPixelFormats.BGR

    camera_mono = OpenCVCamera(system_id=222, camera_index=0, color=False)
    assert camera_mono.pixel_color_format == InputPixelFormats.MONOCHROME


@pytest.mark.xdist_group(name="group1")
def test_opencv_camera_grab_frame_rejects_a_disconnected_or_silent_camera() -> None:
    """Verifies that grabbing a frame is rejected while disconnected and when the camera returns none."""
    camera = OpenCVCamera(system_id=222, camera_index=333)  # Uses invalid index 333.

    message = (
        f"Unable to acquire a frame from the OpenCVCamera interface for the VideoSystem with id "
        f"{camera._system_id}. The interface must be connected to the camera hardware, but it is currently "
        f"disconnected. Call the connect() method prior to calling the grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()

    camera.connect()
    message = (
        f"Unable to acquire a frame from the OpenCVCamera interface for the VideoSystem with id "
        f"{camera._system_id}. The camera hardware must return a frame image for each acquisition request, but "
        f"it returned none. This indicates an initialization or a connectivity issue."
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
    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: capture}))

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
    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: capture}))

    camera = OpenCVCamera(system_id=222, camera_index=0, frame_rate=30)
    camera.connect()

    assert camera.frame_rate == 30


def test_opencv_camera_grab_frame_converts_to_monochrome(monkeypatch) -> None:
    """Verifies that a monochrome camera reduces the three-channel frame OpenCV returns to a single channel."""
    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: FakeVideoCapture()}))

    camera = OpenCVCamera(system_id=222, camera_index=0, color=False)
    camera.connect()

    frame = camera.grab_frame()

    assert frame.ndim == 2
    assert frame.shape == (480, 640)
    assert camera.pixel_color_format == InputPixelFormats.MONOCHROME


def test_get_opencv_ids_reports_every_working_index(monkeypatch) -> None:
    """Verifies that OpenCV discovery reports one entry per working index, carrying the camera's own properties."""
    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: FakeVideoCapture()}))

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

    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: first, 1: duplicate}))

    # The duplicate answers reads while it is probed on its own, and refuses them while the first node is held open,
    # since one physical camera streams to one capture at a time.
    duplicate.readable = True
    original_read = duplicate.read

    def _read_unless_sibling_is_held():
        """Refuses the read while the sibling node is streaming, the way a duplicate device node does."""
        if first.released:
            return original_read()
        return False, None

    monkeypatch.setattr(target=duplicate, name="read", value=_read_unless_sibling_is_held)

    cameras = _get_opencv_ids()

    assert [camera.camera_index for camera in cameras] == [0]


def test_get_opencv_ids_skips_an_index_whose_driver_raises(monkeypatch) -> None:
    """Verifies that an index whose driver raises is skipped rather than aborting the whole discovery sweep."""
    factory = build_capture_factory(captures={1: FakeVideoCapture()})

    def _open_capture(index, apiPreference=None):  # noqa: N803 - mirrors the cv2.VideoCapture argument name.
        """Refuses the first index the way a driver that cannot open its device node does."""
        if index == 0:
            message = "V4L2: failed to open /dev/video0"
            raise RuntimeError(message)
        return factory(index, apiPreference)

    monkeypatch.setattr(target=cv2, name="VideoCapture", value=_open_capture)

    cameras = _get_opencv_ids()

    # One unusable device node costs the caller that node alone, leaving the healthy camera behind it discoverable.
    assert [camera.camera_index for camera in cameras] == [1]


def test_get_opencv_ids_stops_after_five_raising_indices(monkeypatch) -> None:
    """Verifies that an index whose driver raises counts as non-working, ending the sweep after five of them."""
    probed: list[int] = []

    def _open_capture(index, apiPreference=None):  # noqa: N803 - mirrors the cv2.VideoCapture argument name.
        """Refuses every index the way a host whose capture stack is broken does."""
        probed.append(index)
        message = "V4L2: failed to open the capture device"
        raise RuntimeError(message)

    monkeypatch.setattr(target=cv2, name="VideoCapture", value=_open_capture)

    assert _get_opencv_ids() == ()

    # A raise that did not count toward the non-working tally would leave discovery probing all 100 indices, which on
    # a host with a broken capture stack stalls every command that lists cameras.
    assert probed == list(range(_MAXIMUM_NON_WORKING_IDS))


def test_harvesters_camera_init_stores_parameters_and_renders_its_repr() -> None:
    """Verifies that constructing a HarvestersCamera stores the requested parameters and renders them in its repr."""
    # Construction resolves no GenTL Producer, so this test needs neither hardware nor the simulator.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10, frame_width=200, frame_height=200)

    assert camera.frame_rate == 10
    assert camera.frame_width == 200
    assert camera.frame_height == 200
    assert not camera.is_connected
    assert not camera.is_acquiring
    assert camera._system_id == 222

    representation_string = (
        f"HarvestersCamera(system_id={camera._system_id}, camera_index={camera._camera_index}, "
        f"frame_rate={camera.frame_rate} frames / second, frame_width={camera.frame_width} pixels, "
        f"frame_height={camera.frame_height} pixels, connected={camera._camera is not None}, "
        f"acquiring={camera.is_acquiring})"
    )
    assert repr(camera) == representation_string


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_and_disconnect_toggle_the_connection_state() -> None:
    """Verifies that connecting and disconnecting a HarvestersCamera toggles the reported connection state."""
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_width=200, frame_height=200)

    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert not camera.is_acquiring

    # The simulator reports the identity of the device the instance connected to.
    assert camera.model == _SIMULATED_MONOCHROME_MODEL
    assert camera.serial_number == _SIMULATED_MONOCHROME_SERIAL

    camera.disconnect()
    assert not camera.is_connected


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_is_idempotent() -> None:
    """Verifies that reconnecting an already connected camera opens no second GenTL handle."""
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_width=200, frame_height=200)
    camera.connect()
    try:
        acquirer = camera._camera
        harvester = camera._harvester

        camera.connect()

        # Replacing either handle would orphan a live GenTL resource that only garbage collection releases, and a
        # camera that grants exclusive access would refuse to hand out the second one at all.
        assert camera._camera is acquirer
        assert camera._harvester is harvester

        # The identity the first connection resolved survives the re-entry.
        assert camera.model == _SIMULATED_MONOCHROME_MODEL
    finally:
        camera.disconnect()

    assert not camera.is_connected


@pytest.mark.usefixtures("gentl_simulator")
def test_get_harvesters_ids_reports_every_exposed_device() -> None:
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
def test_get_harvesters_ids_skips_an_unqueryable_device(monkeypatch) -> None:
    """Verifies that a device that cannot be queried costs the caller that device alone."""
    original_create = camera_module.Harvester.create
    attempted: list[int] = []

    def _create(self, search_key):
        """Refuses the second device the way a camera already opened by another process does."""
        attempted.append(search_key)
        if search_key == 1:
            message = "device is in use"
            raise RuntimeError(message)
        return original_create(self, search_key=search_key)

    monkeypatch.setattr(target=camera_module.Harvester, name="create", value=_create)

    cameras = _get_harvesters_ids()

    # A busy or half-initialized camera must not blank the listing every command that selects a camera reads.
    assert [camera.camera_index for camera in cameras] == [0, 2, 3]
    assert len(cameras) == _SIMULATED_DEVICE_COUNT - 1
    assert cameras[0].serial_number == _SIMULATED_MONOCHROME_SERIAL

    # Discovery moved past the refusal instead of ending at it, which a returned tuple of the right length alone
    # would not distinguish.
    assert attempted == list(range(_SIMULATED_DEVICE_COUNT))


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_connect_retains_the_rate_without_a_rate_node() -> None:
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
        """Stands in for the optional AcquisitionFrameRate node the bundled simulator does not implement.

        Attributes:
            _value: Stores the frame rate the node currently holds, clamped the way a real float node clamps it.
        """

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

    monkeypatch.setattr(target=camera_module, name="_get_frame_rate_node", value=_resolve_node)

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
def test_harvesters_camera_grab_frame_returns_frames_at_the_requested_size(frame_width, frame_height) -> None:
    """Verifies that a grabbed frame carries the requested geometry and the device's single channel."""
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_width=frame_width, frame_height=frame_height)
    camera.connect()

    assert not camera.is_acquiring
    frame = camera.grab_frame()
    assert camera.is_acquiring

    if frame_height is not None and frame_width is not None:
        assert frame.shape[0] == frame_height
        assert frame.shape[1] == frame_width

    # The monochrome simulated device yields single-channel frames.
    assert frame.ndim == 2

    # Deletes the class to test the functioning of the __del__() method.
    del camera


@pytest.mark.usefixtures("gentl_simulator")
def test_harvesters_camera_grab_frame_converts_a_color_device_to_bgr() -> None:
    """Verifies that HarvestersCamera converts frames from a color device into three-channel BGR arrays."""
    camera = HarvestersCamera(system_id=222, camera_index=_SIMULATED_COLOR_INDEX, frame_width=200, frame_height=200)
    camera.connect()
    try:
        frame = camera.grab_frame()
        assert frame.shape == (200, 200, 3)
        assert camera.pixel_color_format == InputPixelFormats.BGR
    finally:
        camera.disconnect()


def test_harvesters_camera_grab_frame_rejects_a_disconnected_interface() -> None:
    """Verifies that grabbing a frame before the interface connects to the camera is rejected."""
    # The guard under test runs before any GenTL Producer is resolved, so no camera source is required.
    camera = HarvestersCamera(system_id=222, camera_index=0, frame_rate=10, frame_width=200, frame_height=200)

    message = (
        f"Unable to acquire a frame from the HarvestersCamera interface for the VideoSystem with id "
        f"{camera._system_id}. The interface must be connected to the camera hardware, but it is currently "
        f"disconnected. Call the connect() method prior to calling the grab_frame() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = camera.grab_frame()


def test_harvesters_camera_grab_frame_reports_a_failed_fetch() -> None:
    """Verifies that a fetch answering with no buffer raises the documented acquisition failure."""
    camera = HarvestersCamera(system_id=222, camera_index=0)

    # The runtime discards a buffer whose metadata it cannot parse and answers the fetch with nothing, which the
    # bundled Producer simulator never does.
    camera._camera = FakeImageAcquirer(buffers=[None])

    message = (
        f"Unable to acquire a frame from the HarvestersCamera interface for the VideoSystem with id "
        f"{camera._system_id}. The camera hardware must return a frame image for each acquisition request, but "
        f"it returned none. This indicates an initialization or a connectivity issue."
    )
    # Entering the missing buffer's context directly would report an opaque TypeError, which is the whole diagnostic
    # the spawned producer process forwards to the user in place of this message.
    with pytest.raises(BrokenPipeError, match=error_format(message)):
        _ = camera.grab_frame()


@pytest.mark.parametrize(
    ("data_format", "expected_pixel"),
    [
        ("RGB8", [30, 20, 10]),
        ("BGR8", [10, 20, 30]),
    ],
)
def test_harvesters_camera_grab_frame_orders_color_channels(data_format, expected_pixel) -> None:
    """Verifies that color frames are returned in the BGR channel order whichever order the camera streams."""
    # The format tables the branch under test consults are empty where the GenICam runtime is absent, which leaves
    # every format unsupported.
    if not genicam_runtime_available():
        pytest.skip("Skipping this test as this platform does not support the GenICam camera interface.")

    camera = HarvestersCamera(system_id=222, camera_index=0)
    streamed_frame = np.tile(np.array([10, 20, 30], dtype=np.uint8), (2, 3, 1))
    camera._camera = FakeImageAcquirer(buffers=[build_frame_buffer(frame=streamed_frame, data_format=data_format)])

    frame = camera.grab_frame()

    assert frame.shape == (2, 3, 3)

    # A swap that is dropped or applied twice leaves the recording color-inverted, which the frame dimensions the
    # simulated color device pins do not reveal.
    assert frame[0, 0].tolist() == expected_pixel

    # The saver hands FFMPEG the format reported here, so it has to describe the frames the method returns rather
    # than the ones the camera streamed.
    assert camera.pixel_color_format == InputPixelFormats.BGR


@pytest.mark.parametrize("data_format", ["BayerRG8", "RGBa8"])
def test_harvesters_camera_grab_frame_rejects_an_unsupported_format(data_format) -> None:
    """Verifies that a frame in an unsupported format is rejected by name with its buffer returned to the camera."""
    camera = HarvestersCamera(system_id=222, camera_index=0)
    buffer = build_frame_buffer(frame=np.zeros((2, 3, 3), dtype=np.uint8), data_format=data_format)
    camera._camera = FakeImageAcquirer(buffers=[buffer])

    # Naming the offending format is the entire diagnostic a user pointing the library at a Bayer-output or
    # four-channel camera has to work from.
    with pytest.raises(ValueError, match=data_format):
        _ = camera.grab_frame()

    # A camera streaming an unsupported format raises once per frame, so a buffer that is not re-queued here would
    # exhaust the acquirer's buffer pool instead of failing cleanly.
    assert buffer.exited


@pytest.mark.usefixtures("gentl_simulator")
@pytest.mark.parametrize(
    ("camera_index", "expected_format"),
    [(0, InputPixelFormats.MONOCHROME), (_SIMULATED_COLOR_INDEX, InputPixelFormats.BGR)],
)
def test_harvesters_camera_pixel_color_format_reflects_the_connected_device(camera_index, expected_format) -> None:
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
def test_harvesters_camera_negotiates_the_rate_and_acquires_from_hardware(has_harvesters) -> None:
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


def test_harvesters_camera_pixel_color_format_covers_both_color_states() -> None:
    """Verifies the pixel_color_format property for both _color=True and _color=False states."""
    # Tests both branches of pixel_color_format by directly setting the _color attribute. No hardware
    # interaction is needed since the property simply reads the attribute value.
    camera = HarvestersCamera(system_id=222, camera_index=0)

    camera._color = True
    assert camera.pixel_color_format == InputPixelFormats.BGR

    camera._color = False
    assert camera.pixel_color_format == InputPixelFormats.MONOCHROME


@pytest.mark.usefixtures("gentl_simulator")
def test_harvester_connection_yields_a_connected_camera() -> None:
    """Verifies that the managed connection exposes a connected camera and releases it when the block ends."""
    with harvester_connection(camera_index=0) as camera:
        assert camera.is_connected
        assert camera.model == _SIMULATED_MONOCHROME_MODEL

        # Exposing the node map is the reason the helper exists, since every GenICam configuration entry point reads
        # and writes the camera through it.
        assert camera.node_map.Width.value > 0

    # The GenTL handle is released for other processes rather than held until the interpreter exits.
    assert not camera.is_connected
    assert camera._harvester is None


@pytest.mark.usefixtures("gentl_simulator")
def test_harvester_connection_releases_the_camera_on_error() -> None:
    """Verifies that an error raised inside the managed block reaches the caller with the camera released."""
    message = "The configuration command failed while the camera was connected."

    # Swallowing the error would leave a failed configuration command reporting success, and releasing the camera only
    # on the success path would hold the GenTL handle for the rest of the runtime.
    with pytest.raises(RuntimeError, match=message), harvester_connection(camera_index=0) as camera:
        raise RuntimeError(message)

    assert not camera.is_connected
    assert camera._harvester is None


def test_genicam_runtime_available_tracks_the_imported_runtime(monkeypatch) -> None:
    """Verifies that genicam_runtime_available() reports whether the GenICam runtime imported."""
    # Patches both states rather than reading the host's own, so the assertions hold on macOS too, where the library
    # installs no runtime and the guarded import falls back to None.
    monkeypatch.setattr(target=camera_module, name="Harvester", value=object())
    assert genicam_runtime_available()

    monkeypatch.setattr(target=camera_module, name="Harvester", value=None)
    assert not genicam_runtime_available()


def test_harvesters_camera_connect_requires_the_genicam_runtime(monkeypatch) -> None:
    """Verifies that connecting to a GenICam camera aborts on a platform that does not support the interface."""
    monkeypatch.setattr(target=camera_module, name="Harvester", value=None)
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

    monkeypatch.setattr(target=camera_module, name="Harvester", value=None)
    monkeypatch.setattr(target=camera_module, name="_get_harvesters_ids", value=_forbidden_discovery)
    monkeypatch.setattr(target=camera_module, name="_get_opencv_ids", value=_no_opencv_cameras)

    assert discover_camera_ids() == ()


@pytest.mark.usefixtures("persisted_cti_directory")
def test_discover_camera_ids_skips_genicam_without_a_configured_producer(monkeypatch) -> None:
    """Verifies that camera discovery reports OpenCV cameras alone where no GenTL Producer has been configured."""
    # A non-None sentinel keeps the runtime gate open on every platform, including macOS. It is never instantiated,
    # since resolving the Producer path raises before Harvesters discovery constructs one.
    monkeypatch.setattr(target=camera_module, name="Harvester", value=object)
    monkeypatch.setattr(target=cv2, name="VideoCapture", value=build_capture_factory(captures={0: FakeVideoCapture()}))

    cameras = discover_camera_ids()

    # An unconfigured Producer costs the caller the GenICam half of discovery alone, which is what keeps every command
    # that lists cameras usable on a machine that has a webcam and no vendor SDK.
    assert [camera.camera_index for camera in cameras] == [0]
    assert all(camera.interface == CameraInterfaces.OPENCV for camera in cameras)


def test_check_cti_file_reports_an_unsupported_platform(monkeypatch) -> None:
    """Verifies that check_cti_file() reports an unusable configuration where the GenICam interface is unsupported."""
    monkeypatch.setattr(target=camera_module, name="Harvester", value=None)

    assert check_cti_file() is None


@pytest.mark.usefixtures("persisted_cti_directory")
def test_check_cti_file_reports_no_configured_producer(monkeypatch) -> None:
    """Verifies that a machine with no configured Producer reports one as absent instead of raising."""
    # A non-None sentinel keeps the runtime gate open on every platform, including macOS. It is never instantiated,
    # since the absent path file answers before any Producer is loaded.
    monkeypatch.setattr(target=camera_module, name="Harvester", value=object)

    assert check_cti_file() is None


def test_check_cti_file_prefers_the_runtime_override(persisted_cti_directory, gentl_simulator) -> None:
    """Verifies that the runtime override outranks the persisted path when the configured Producer is reported."""
    (persisted_cti_directory / "cti_path.txt").write_text(str(persisted_cti_directory / "decoy.cti"))

    # The override redirects every process of a runtime, so the reported Producer has to be the one a camera
    # connection opened from the same runtime would load.
    assert check_cti_file() == gentl_simulator.resolve()


def test_check_cti_file_rejects_a_stale_persisted_producer(persisted_cti_directory, monkeypatch) -> None:
    """Verifies that a persisted Producer the runtime can no longer load is reported as absent."""
    (persisted_cti_directory / "cti_path.txt").write_text(str(persisted_cti_directory / "uninstalled.cti"))

    class _UninstalledProducer:
        """Stands in for the runtime refusing a Producer whose vendor SDK is no longer installed."""

        def add_file(self, file_path: str, **_checks: bool) -> None:
            """Refuses the Producer the way the runtime refuses one it cannot load."""
            message = f"{file_path}: cannot open shared object file"
            raise OSError(message)

    monkeypatch.setattr(target=camera_module, name="Harvester", value=_UninstalledProducer)

    # Uninstalling the vendor SDK leaves the status query answering 'not configured' rather than propagating the
    # loader's failure to the caller.
    assert check_cti_file() is None


def test_add_cti_file_requires_the_genicam_runtime(monkeypatch) -> None:
    """Verifies that configuring a Producer aborts on a platform that does not support the GenICam interface."""
    monkeypatch.setattr(target=camera_module, name="Harvester", value=None)

    # Reuses the module's own explanation, which is resolved from the host platform and therefore differs per platform.
    message = f"Unable to configure the GenTL Producer interface (.cti) file. {GENICAM_UNAVAILABLE_REASON}"
    with pytest.raises(NotImplementedError, match=error_format(message)):
        add_cti_file(cti_path=Path("TLSimu.cti"))


def test_add_cti_file_persists_a_resolved_producer_path(
    persisted_cti_directory, simulator_cti_path, monkeypatch
) -> None:
    """Verifies that a configured Producer is persisted as the absolute path every later runtime resolves."""
    if not genicam_runtime_available():
        pytest.skip("Skipping this test as this platform does not support the GenICam camera interface.")

    if simulator_cti_path is None:
        pytest.skip("Skipping this test as no GenTL Producer simulator is bundled for this platform.")

    # A relative argument resolves against the directory the configuring command happened to run from, so the command
    # is run from that directory rather than the one the test session started in.
    monkeypatch.chdir(path=simulator_cti_path.parent)

    add_cti_file(cti_path=Path(simulator_cti_path.name))

    # Persisting the argument as given would leave every later runtime started from anywhere else unable to find the
    # Producer this command validated.
    resolved_path = simulator_cti_path.resolve()
    assert (persisted_cti_directory / "cti_path.txt").read_text() == str(resolved_path)

    # The three functions that share the path file agree on its format, so what one runtime configures is what the
    # next one reports and connects through.
    assert check_cti_file() == resolved_path
    assert _get_cti_path() == resolved_path


def test_add_cti_file_rejects_a_missing_producer(persisted_cti_directory, tmp_path) -> None:
    """Verifies that configuring an absent Producer leaves the previously configured Producer in place."""
    if not genicam_runtime_available():
        pytest.skip("Skipping this test as this platform does not support the GenICam camera interface.")

    path_file = persisted_cti_directory / "cti_path.txt"
    configured_path = tmp_path / "configured.cti"
    path_file.write_text(str(configured_path))

    # The runtime reports an absent Producer with a bare FileNotFoundError, which carries no message to match against.
    with pytest.raises(FileNotFoundError):
        add_cti_file(cti_path=tmp_path / "absent.cti")

    # A mistyped path must not cost the user the working configuration they already had.
    assert path_file.read_text() == str(configured_path)


def test_add_cti_file_rejects_a_file_that_is_not_a_producer(persisted_cti_directory, tmp_path) -> None:
    """Verifies that configuring a file the runtime cannot load persists nothing."""
    if not genicam_runtime_available():
        pytest.skip("Skipping this test as this platform does not support the GenICam camera interface.")

    fake_producer = tmp_path / "fake.cti"
    fake_producer.write_text("not a shared library")

    # The dynamic loader phrases its refusal differently on each platform, so only the failure itself is pinned.
    with pytest.raises(OSError, match=r".*") as loader_error:
        add_cti_file(cti_path=fake_producer)

    # A file that exists but cannot be loaded is refused by the validity check rather than accepted as present, which
    # is what separates a configured Producer from a merely present file.
    assert not isinstance(loader_error.value, FileNotFoundError)
    assert not (persisted_cti_directory / "cti_path.txt").exists()


@pytest.mark.usefixtures("persisted_cti_directory")
def test_get_cti_path_reports_an_unconfigured_producer() -> None:
    """Verifies that resolving the Producer path on an unconfigured machine names the command that configures one."""
    # Every runtime resolves the Producer through this function, including each spawned producer process, so this
    # message is the only guidance a user whose recording refuses to start receives.
    message = (
        "Unable to resolve the path to the GenTL Producer interface (.cti) file to use for the harvesters camera "
        "interface, as the .cti file has not been set. Set the .cti file path by calling the 'axvs cti set' CLI "
        "command."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        _get_cti_path()


def test_get_cti_path_reads_the_persisted_producer(persisted_cti_directory) -> None:
    """Verifies that the persisted Producer path is read back whole, without validation."""
    configured_path = persisted_cti_directory / "vendor" / "Producer.cti"

    # The trailing newline reproduces the file a text editor leaves behind, which must not become part of the path.
    (persisted_cti_directory / "cti_path.txt").write_text(f"{configured_path}\n")

    # Resolution performs no validation of its own, since callers validate the path through the Producer load that
    # follows it.
    assert _get_cti_path() == configured_path
