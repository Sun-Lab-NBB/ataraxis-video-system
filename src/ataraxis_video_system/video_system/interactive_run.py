"""This module contains the interactive_run script that allows..."""

from ataraxis_base_utilities import console

from .vsc import Camera, VideoSystem


# Since the video system uses multiprocessing, it should always be called within '__main__' scope
def interactive_run() -> None:
    console.enable()
    vs = VideoSystem("img_directory", Camera())
    vs.start(listen_for_keypress=True)
    input()

