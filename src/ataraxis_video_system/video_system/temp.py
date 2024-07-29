from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("imgs", Camera())
    vs.start()
    time.sleep(5)
    vs.stop()
    vs.save_imgs_as_vid()
