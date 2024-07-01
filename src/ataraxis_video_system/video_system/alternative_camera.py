from typing import Any

import cv2
import numpy as np


class Camera:
    """A wrapper clase for an opencv VideoCapture object.

    Attributes:
        _vid: opencv video capture object.
    """

    num_connected = 0
    video_capture = None

    def __init__(self) -> None:
        self._connected = False

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()

    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
        if not self._connected:
            if Camera.num_connected == 0:
                Camera.video_capture = cv2.VideoCapture(0)
            self._connected = True
            Camera.num_connected += 1

    def disconnect(self) -> None:
        """Disconnects from camera."""
        if self._connected:
            if Camera.num_connected == 1:
                if Camera.video_capture is not None:
                    Camera.video_capture.release()
            self._connected = False
            Camera.num_connected -= 1

    @property
    def is_connected(self) -> bool:
        """Whether or not the camera is connected."""
        return self._connected

    def grab_frame(self) -> np.typing.NDArray[Any]:
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected.

        """
        if self._connected and Camera.video_capture is not None:
            ret, frame = Camera.video_capture.read()
            # if not ret:
            #     raise Exception("camera did not yield an image")
            return frame
        else:
            raise Exception("camera not connected")
