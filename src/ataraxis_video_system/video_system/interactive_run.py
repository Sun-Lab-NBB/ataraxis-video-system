"""This module contains the interactive_run script that allows..."""

from ataraxis_base_utilities import console

from .vsc import Camera, VideoSystem


# Since the video system uses multiprocessing, it should always be called within '__main__' scope
def interactive_run() -> None:
    console.enable()
    vs = VideoSystem(
        save_directory="imgs", camera=Camera(), save_format="png"
    )  # Create the system using the built-in camera class
    vs.start(listen_for_keypress=True)  # Start the system, activates camera and begins taking and saving images
    input()

    # time.sleep(5)
    # vs.stop()  # End the system, discarding any unsaved images
