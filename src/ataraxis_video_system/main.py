from ataraxis_video_system import VideoSystem
from camera import Camera
import time
from pynput import keyboard
import threading

if __name__ == "__main__":
    vs = VideoSystem('imgs', Camera())
    vs.start(True)

    time.sleep(7)
    print('yes')

    vs.stop()












