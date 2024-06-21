import logging
import sys
import time

import cv2
import ffmpeg
import numpy

logger = logging.getLogger("Writer")
logger.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(module)s %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)


videoCapture = cv2.VideoCapture("src\\ffmpeg_backend\\earth_rotation.avi")

process = (
    ffmpeg.input(
        "pipe:",
        framerate="{}".format(videoCapture.get(cv2.CAP_PROP_FPS)),
        format="rawvideo",
        pix_fmt="bgr24",
        s="{}x{}".format(
            int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ),
    )
    .output("src\\ffmpeg_backend\\earth_rotation.mp4", vcodec="h264", pix_fmt="nv21", **{"b:v": 2000000})
    .overwrite_output()
    .run_async(pipe_stdin=True)
)
lastFrame = False
frames = 0
start = time.time()
while not lastFrame:
    ret, image = videoCapture.read()
    if ret:
        process.stdin.write(image.astype(numpy.uint8).tobytes())
        frames += 1
    else:
        lastFrame = True
elapsed = time.time() - start
logger.info("%d frames" % frames)
logger.info("%4.1f FPS, elapsed time: %4.2f seconds" % (frames / elapsed, elapsed))
del videoCapture
