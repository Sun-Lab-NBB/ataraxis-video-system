from ataraxis_video_system import VideoSystem
from camera import Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("imgs", Camera())
    vs.start()
    time.sleep(5)
    vs.stop()
