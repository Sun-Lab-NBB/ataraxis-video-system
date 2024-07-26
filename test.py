import sys

from src.ataraxis_video_system.video_system.camera import OpenCVCamera
import time as tm
import cv2

cam = OpenCVCamera(0, 10, 100, 100)
cam.connect()
print(cam.fps)
print(cam.backend)
print(cam.width)
print(cam.height)

frames = 0
start = tm.time()
while tm.time() - start < 1:
    frame = cam.grab_frame()
    print(frame.size)
    frames += 1
cam.disconnect()

print(frames)
