import logging
import sys
import time

import cv2
import numpy

import ffmpeg

# logger = logging.getLogger("Writer")
# logger.setLevel("INFO")
# formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(module)s %(message)s")
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(formatter)
# logger.addHandler(handler)


videoCapture = cv2.VideoCapture(0)

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
    .output("src\\ffmpeg_backend\\webcam.mp4", vcodec="h264", pix_fmt="nv21", **{"b:v": 2000000})
    .overwrite_output()
    .run_async(pipe_stdin=True)
)


# ffmpeg_backend -codecs

# nvenc

# h264
# DEV.LS h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (decoders: h264 libopenh264 h264_cuvid) (encoders: libx264 libx264rgb libopenh264 h264_mf h264_nvenc)

# h265

# DEV.L. hevc                 H.265 / HEVC (High Efficiency Video Coding) (decoders: hevc hevc_cuvid) (encoders: libx265 hevc_mf hevc_nvenc)

# What works on my laptop:
# h264
# h264_mf
# hevc

vid_len = 6
frames = 0
start = time.time()

while time.time() - start <= vid_len:
    ret, image = videoCapture.read()
    if ret:
        process.stdin.write(image.astype(numpy.uint8).tobytes())
        frames += 1
# elapsed = time.time() - start
# logger.info("%d frames" % frames)
# logger.info("%4.1f FPS, elapsed time: %4.2f seconds" % (frames / elapsed, elapsed))
del videoCapture

process.stdin.close()
process.wait()
