# First, import the VideoSystem and Camera classes
import time

# from video_system.video_system import Camera, VideoSystem
from .vsc import Camera, VideoSystem


# Since the video system uses multiprocessing, it should always be called within '__main__' scope
def interactive_run() -> None:
    print("today1")
    vs = VideoSystem("imgs", Camera())  # Create the system using the built-in camera class
    vs.start(
        save_format="mp4", listen_for_keypress=True
    )  # Start the system, activates camera and begins taking and saving images
    input()
    # time.sleep(5)
    # vs.stop_image_collection()  # Stop the camera from taking any more pictures but continue image saving
    # time.sleep(5)
    # vs.stop()  # End the system, discarding any unsaved images
