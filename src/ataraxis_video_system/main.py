from ataraxis_video_system import VideoSystem
from camera import Camera

if __name__ == '__main__':
    vs = VideoSystem("imgs", Camera())
    # VideoSystem._delete_files_in_directory('foo')
    vs.start()


