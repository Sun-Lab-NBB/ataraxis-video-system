# First, import the Videosystem and Camera classes
from video_system.ataraxis_video_system import Videosystem, Camera
import time

# Since the video system uses multiprocessing, it should always be called within '__main__' scope
if __name__ == "__main__":
    vs = Videosystem("imgs", Camera()) # Create the system using the built in camera class
    vs.start() # Start the system, activates camera and begins takign and saving images
    time.sleep(5)
    vs.stop_image_collection() # Stop the camera from taking any more pictures but continue image saving
    time.sleep(5)
    vs.stop() # End the system, discarding any unsaved images


# if __name__ == "__main__":
#     vs = Videosystem("imgs", Camera())
#     vs.start(True)
#     input()  # Need this line because subprocesses are daemon, as soon as you leave the __main__ scope the processes stop
