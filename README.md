# AtaraxisVideoSystem

A Python library that interfaces with a wide range of cameras to flexibly record visual stream data as images or videos.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-video-system)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-video-system)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-video-system)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-video-system)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-video-system)
___

## Detailed Description

This library provides an interface for efficiently acquiring and saving visual data from cameras in real time. To 
achieve this, the library internally binds OpenCV and GeniCam backends to grab frames from a wide range of consumer, 
industrial, and scientific cameras using USB and Gigabit interfaces. To save the acquired frames, the library uses 
FFMPEG to recruit CPUs or GPUs and supports H264 and H265 codecs. The library abstracts setup, acquisition, and cleanup 
procedures via a simple API exposed by the VideoSystem interface class, while allowing for extensive configuration of 
all managed elements. To optimize runtime efficiency, the library uses multithreading and multiprocessing, where 
appropriate.
___

## Features

- Supports Windows, Linux, and macOS.
- Uses OpenCV or GeniCam (Harvesters) to interface with a wide range of consumer, industrial and scientific cameras.
- Uses FFMPEG to efficiently encode acquired data as videos or images in real time.
- Highly flexible and customizable, can be extensively fine-tuned for quality or throughput.
- Pure-python API.
- GPL 3 License.
___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

- [FFMPEG](https://www.ffmpeg.org/download.html). Make sure that the installed FFMPEG is available in your system’s 
  path, and Python has permissions to call FFMPEG. We recommend using the latest stable release of FFMPEG, although the
  minimal requirement is support for H254 and H265 codecs.
- [Harvesters Dependencies](https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst) if you intend to use 
  Harvesters camera backend. Primarily, this only includes the ***GenTL Producer CTI***, as all other dependencies are 
  installed automatically. If your camera vendor supplies a .cti file for that camera, use that file. Otherwise, we 
  recommend using the [MvImpactAcquire](https://assets-2.balluff.com/mvIMPACT_Acquire/). This producer requires 
  licensing for versions at or above **3.0.0**, but lower versions seem to function well with all tested hardware. This 
  library has been tested using version **2.9.2**.

For users, all other library dependencies are installed automatically by all supported installation methods 
(see [Installation](#installation) section).

For developers, see the [Developers](#developers) section for information on installing additional development 
dependencies.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-video-system/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-video-system```.
___

## Usage

### Quickstart
This is a minimal example of how to use this library. It is also available as a [script](examples/quickstart.py).
This example is intentionally kept minimal, consult 
[API documentation](https://ataraxis-video-system-api-docs.netlify.app/) for more details.
```
from ataraxis_video_system import VideoSystem, InputPixelFormats
from ataraxis_data_structures import DataLogger
from pathlib import Path
import numpy as np
from ataraxis_time import PrecisionTimer

# The directory where to output the recorded frames and the acquisition timestamps
output_directory = Path("/home/cyberaxolotl/Desktop/video_test")

# The DataLogger is used to save frame acquisition timestamps to disk. During runtime, it logs frame timestamps as
# uncompressed NumPy arrays (.npy) and after runtime it can compress log entries into one .npz archive.
logger = DataLogger(output_directory=output_directory, instance_name="webcam", exist_ok=True)

# DataLogger uses a parallel process to write log entries to disk. It has to be started before it can save any log
# entries.
logger.start()

# The VideoSystem minimally requires an ID and a DataLogger instance. The ID is critical, as it is used to identify the
# log entries generated by the VideoSystem. For VideoSystems that will be saving frames, output_directory is also
# required
vs = VideoSystem(system_id=np.uint8(101), data_logger=logger, output_directory=output_directory)

# By default, all added cameras are interfaced with using OpenCV backend. Check the API documentation to learn about
# all available camera configuration options and supported backends. This camera is configured to save frames and to
# display the live video feed to the user at 25 fps. The display framerate can be the same or lower as the acquisition
# framerate.
vs.add_camera(
    save_frames=True,
    acquisition_frame_rate=30,
    display_frames=True,
    display_frame_rate=15,
    color=True,  # Most WebCameras default to using colored output
)

# To save the frames acquired by the system, we need to add a saver. Here, we demonstrate adding a video saver, but you
# can also use the add_image_saver() method to output frames as images. The default video saver uses CPU and H265 codec
# to encode frames as an MP4 video.
vs.add_video_saver(input_pixel_format=InputPixelFormats.BGR)  # Colored frames will be acquired with BGR/A format

# Calling this method arms the video system and starts frame acquisition. However, the frames are not initially saved to
# disk.
vs.start()

timer = PrecisionTimer("s")
timer.delay_noblock(delay=2)  # During this delay, camera frames are displayed to the user, but are not saved

# Begins saving frames to disk as an MP4 video file
vs.start_frame_saving()
timer.delay_noblock(delay=5)  # Records frames for 5 seconds, generating ~150 frames
vs.stop_frame_saving()

# Frame acquisition can be started and stopped as needed, although all frames will be written to the same output video
# file. If you intend to cycle frame acquisition, it may be better to use an image saver backend.

# Stops the VideoSystem runtime and releases all resources
vs.stop()

# Stops the DataLogger and compresses acquired logs into a single .npz archive. This step is required for being able to
# parse the data with the VideoSystem API
logger.stop()
logger.compress_logs(remove_sources=True)

# Extracts the list of frame timestamps from the compressed log generated above. The extraction function automatically
# uses VideoSystem ID, DataLogger name, and the output_directory to resolve the archive path.
timestamps = vs.extract_logged_data()  # Returns a list of timestamps, each is given in microseconds since epoch onset

# Computes and prints the actual framerate of the camera based on saved frames.
timestamps = np.array(timestamps, dtype=np.uint64)
time_diffs = np.diff(timestamps)
fps = 1 / (np.mean(time_diffs) / 1e6)
print(fps)

# You can also check the output directory for the created video.
```

___

## API Documentation

See the [API documentation](https://ataraxis-video-system-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library.

### Installing the library

The easiest way to ensure you have most recent development dependencies and library source files is to install the 
python environment for your OS (see below). All environments used during development are exported as .yml files and as 
spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three explicitly 
supported OS families: Windows 11, OSx Darwin, and GNU Linux.

**Note!** Since the OSx environment was built for the Darwin platform (Apple Silicon), it may not work on Intel-based 
Apple devices.

1. If you do not already have it installed, install [tox](https://tox.wiki/en/latest/user_guide.html) into the active
   python environment. The rest of this installation guide relies on the interaction of local tox installation with the
   configuration files included in with this library.
2. Download this repository to your local machine using your preferred method, such as git-cloning. If necessary, unpack
   and move the project directory to the appropriate location on your system.
3. ```cd``` to the root directory of the project using your command line interface of choice. Make sure it contains
   the `tox.ini` and `pyproject.toml` files.
4. Run ```tox -e import``` to automatically import the os-specific development environment included with the source 
   distribution. Alternatively, you can use ```tox -e create``` to create the environment from scratch and automatically
   install the necessary dependencies using pyproject.toml file. 
5. If either step 4 command fails, use ```tox -e provision``` to fix a partially installed environment.

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project will likely work on any 
common OS, but may require additional configurations steps.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. These 
   versions will be installed in-addition to the main Python version installed in the development environment.
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime for this task, use ```tox --parallel```.

For more information, you can also see the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation#Usage) documentation.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.

___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-video-system/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Jacob Groner ([Jgroner11](https://github.com/Jgroner11))
- Natalie Yeung

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other projects used in our development automation pipelines [see pyproject.toml](pyproject.toml).

---
