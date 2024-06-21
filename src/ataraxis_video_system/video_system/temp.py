from typing import Literal, get_args

T = Literal['png', 'jpg', 'mp4']
s: T = 'jpg'

print(s in get_args(T))




# import cv2

# vid = cv2.VideoCapture(0)

# ret, frame = vid.read()
# print(ret)

# print(vid.get(cv2.CAP_PROP_VIDEO_TOTAL_CHANNELS))

# vid.release()

# print(vid.get(cv2.CAP_PROP_VIDEO_TOTAL_CHANNELS))
