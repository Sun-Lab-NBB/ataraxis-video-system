import os
import glob

import cv2
import numpy as np
import ffmpeg  # type: ignore
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import SharedMemoryArray

from ataraxis_video_system.video_system.vsc import MPQueue

from .vsc import Camera, VideoSystem


# Since the video system uses multiprocessing, it should always be called within '__main__' scope
def interactive_run() -> None:
    vs = VideoSystem(
        save_directory="imgs", camera=Camera(), save_format="tiff"
    )  # Create the system using the built-in camera class
    vs.start(
        listen_for_keypress=True, tiff_compression_level=9
    )  # Start the system, activates camera and begins taking and saving images
    input()

    # time.sleep(5)
    # vs.stop()  # End the system, discarding any unsaved images
