from ataraxis_video_system import VideoSystem
from camera import Camera

if __name__ == '__main__':
    vs = VideoSystem("imgs", Camera())
    vs.start()


