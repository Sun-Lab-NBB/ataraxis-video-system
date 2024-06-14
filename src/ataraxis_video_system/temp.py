import ffmpeg

(ffmpeg.input("/path/to/jpegs/*.jpg", pattern_type="glob", framerate=25).output("movie.mp4").run())


import os
import subprocess

command = "ffmpeg -framerate 30 -i image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4"
subprocess.call(command, shell=True)
