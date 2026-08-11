"""Provides a synthetic cv2.VideoCapture stand-in shared by the camera and video system test modules.

The OpenCV acquisition paths negotiate frame rate, width, and height against whatever the driver reports back, and a
real webcam is free to ignore any of the three. No camera is attached to a continuous integration host, and the ones
attached to a developer's host cannot be made to refuse a specific parameter on demand, so those negotiations are
exercised against this stand-in instead.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

_DEFAULT_PROPERTIES: dict[int, float] = {
    cv2.CAP_PROP_FPS: 30.0,
    cv2.CAP_PROP_FRAME_WIDTH: 640.0,
    cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
}
"""The property values a capture reports when a test declares no override of its own."""


class FakeVideoCapture:
    """Emulates a cv2.VideoCapture bound to one camera index, without touching any hardware.

    Args:
        properties: The property values this capture reports, keyed by the cv2 property identifier.
        accepts_writes: Determines whether a set() call updates the reported property value. A capture that refuses
            them reproduces the common driver that silently keeps its own acquisition parameters.
        opened: Determines whether isOpened() reports the capture as usable.
        readable: Determines whether read() answers with a frame.
        color: Determines whether read() answers with a three-channel frame or a single-channel one.
    """

    def __init__(
        self,
        properties: dict[int, float] | None = None,
        *,
        accepts_writes: bool = True,
        opened: bool = True,
        readable: bool = True,
        color: bool = True,
    ) -> None:
        self.properties = dict(_DEFAULT_PROPERTIES if properties is None else properties)
        self.accepts_writes = accepts_writes
        self.opened = opened
        self.readable = readable
        self.color = color
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2.VideoCapture method name this class stands in for.
        """Returns whether the capture reports itself as usable."""
        return self.opened

    def get(self, propId: int) -> float:  # noqa: N803 - mirrors the cv2.VideoCapture argument name.
        """Returns the value this capture reports for the requested property."""
        return self.properties.get(propId, 0.0)

    def set(self, propId: int, value: float) -> bool:  # noqa: N803 - mirrors the cv2.VideoCapture argument name.
        """Records the requested property write and applies it only when this capture accepts writes."""
        self.set_calls.append((propId, value))
        if self.accepts_writes:
            self.properties[propId] = value
        return self.accepts_writes

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Returns one synthetic frame sized to the properties this capture currently reports."""
        if not self.readable:
            return False, None

        height = int(self.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
        shape = (height, width, 3) if self.color else (height, width)
        return True, np.zeros(shape, dtype=np.uint8)

    def release(self) -> None:
        """Marks the capture as released, which is what the interfaces call during their own teardown."""
        self.released = True


def build_capture_factory(captures: dict[int, FakeVideoCapture], default: FakeVideoCapture | None = None):
    """Builds the cv2.VideoCapture replacement that hands each camera index its declared stand-in.

    Args:
        captures: The stand-in each camera index resolves to, keyed by that index.
        default: The stand-in every index absent from the mapping resolves to. Leaving this unset resolves those
            indices to an unopened capture, which is what an index holding no camera reports.

    Returns:
        The callable that replaces cv2.VideoCapture for the duration of a test.
    """

    def _open_capture(
        index: int,
        apiPreference: int | None = None,  # noqa: N803 - mirrors the cv2.VideoCapture argument name.
    ) -> FakeVideoCapture:
        """Returns the stand-in bound to the requested camera index, marked as open again."""
        capture = captures.get(index, default)
        if capture is None:
            return FakeVideoCapture(opened=False, readable=False)

        # Each call reopens the device, so a stand-in handed out twice reports itself as held rather than as the
        # released one its previous probe left behind. Duplicate device-node detection reads exactly that state.
        capture.released = False
        return capture

    return _open_capture
