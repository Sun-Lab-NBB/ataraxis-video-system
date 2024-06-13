from ataraxis_video_system.ataraxis_video_system import Camera, Videosystem

if __name__ == "__main__":
    vs = VideoSystem("imgs", Camera())
    vs.start(True)
    input()  # Need this line because subprocesses are daemon, as soon as you leave the __main__ scope the processes stop
