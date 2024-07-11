import numpy as np
from _typeshed import Incomplete
from typing import Any

class Camera:
    """A wrapper clase for an opencv VideoCapture object.

    Attributes:
        _vid: opencv video capture object.
    """
    num_connected: int
    video_capture: Incomplete
    _connected: bool
    def __init__(self) -> None: ...
    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
    def disconnect(self) -> None:
        """Disconnects from camera."""
    @property
    def is_connected(self) -> bool:
        """Whether or not the camera is connected."""
    def grab_frame(self) -> np.typing.NDArray[Any]:
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected.

        """
