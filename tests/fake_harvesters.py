"""Provides synthetic Harvesters acquisition stand-ins shared by the camera test module.

The GenICam frame grab path reshapes whatever payload the camera streams and re-orders its color channels. The bundled
GenTL Producer simulator streams Mono8 and RGB8 payloads alone: it never fails a fetch, never delivers a payload that
is already in the BGR channel order, and never delivers one in a format the library does not support. Those three
acquisition outcomes are reproduced with these stand-ins, which supply the payload while every reshaping, re-ordering,
and rejection decision under test remains the library's own.
"""

from typing import Self

import numpy as np
from numpy.typing import NDArray


class _FakeComponent:
    """Emulates one component of a Harvesters buffer payload, which carries a single acquired frame.

    Args:
        data: The one-dimensional frame data, laid out the way the camera streams it.
        width: The width of the frame the data encodes, in pixels.
        height: The height of the frame the data encodes, in pixels.
        data_format: The GenICam pixel format name this component reports for the data.
        num_components_per_pixel: The number of color channels the data holds for each pixel.
    """

    def __init__(
        self,
        data: NDArray[np.uint8],
        width: int,
        height: int,
        data_format: str,
        num_components_per_pixel: float,
    ) -> None:
        self.data = data
        self.width = width
        self.height = height
        self.data_format = data_format
        self.num_components_per_pixel = num_components_per_pixel


class _FakePayload:
    """Emulates the payload of a Harvesters buffer, which exposes the acquired frame as its first component.

    Args:
        component: The component carrying the acquired frame.
    """

    def __init__(self, component: _FakeComponent) -> None:
        self.components = (component,)


class FakeBuffer:
    """Emulates a Harvesters buffer, which is re-queued with the camera when its context is exited.

    Args:
        payload: The payload this buffer carries.
    """

    def __init__(self, payload: _FakePayload) -> None:
        self.payload = payload
        self.entered = False
        self.exited = False

    def __enter__(self) -> Self:
        """Marks the buffer as held by the caller and returns it."""
        self.entered = True
        return self

    def __exit__(self, exception_type: object, exception_value: object, traceback: object) -> bool:
        """Marks the buffer as re-queued with the camera, which is what a real buffer does when its context exits."""
        self.exited = True
        return False


class FakeImageAcquirer:
    """Emulates a Harvesters ImageAcquirer bound to one camera, without loading any GenTL Producer.

    Args:
        buffers: The buffers this acquirer hands out, one per fetch() call, in order. A None entry reproduces the fetch
            that discards a buffer whose metadata the runtime cannot parse and answers with nothing.
    """

    def __init__(self, buffers: list[FakeBuffer | None]) -> None:
        self.buffers = list(buffers)
        self.start_calls = 0
        self._acquiring = False

    def is_acquiring(self) -> bool:
        """Returns whether the acquirer has been started."""
        return self._acquiring

    def start(self) -> None:
        """Starts the simulated acquisition, the way the frame grab path starts a real one on its first call."""
        self._acquiring = True
        self.start_calls += 1

    def fetch(self) -> FakeBuffer | None:
        """Returns the next declared buffer, answering with None where the declaration stands for a discarded one."""
        return self.buffers.pop(0)


def build_frame_buffer(frame: NDArray[np.uint8], data_format: str) -> FakeBuffer:
    """Builds the buffer a camera streaming the provided frame in the provided format hands to the frame grab path.

    Args:
        frame: The frame the buffer carries, shaped (height, width) for monochrome data or (height, width, channels)
            for color data, with the channels in the order the camera streams them rather than the order the library
            returns them in.
        data_format: The GenICam pixel format name the buffer reports for the frame.

    Returns:
        The buffer instance carrying the flattened frame data.
    """
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    component = _FakeComponent(
        data=frame.reshape(-1),
        width=frame.shape[1],
        height=frame.shape[0],
        data_format=data_format,
        num_components_per_pixel=float(channels),
    )
    return FakeBuffer(payload=_FakePayload(component=component))
