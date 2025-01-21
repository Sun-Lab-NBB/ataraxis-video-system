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
industrial and scientific cameras using USB and Gigabit interfaces. To save the acquired frames, the library uses FFMPEG
CPUs and / or GPUs and supports H264 and H265 codecs. The library abstracts all setup, acquisition, and cleanup 
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
This is a minimal example of how to use this library:

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera())
    vs.start()
    time.sleep(5)
    vs.stop()
```

In this example, we create a video system object, specifying the location to save images and passing a default Camera 
object. We run the system for 5 seconds before stopping. The video system should always be run within the "\_\_main__" 
scope to ensure it is only run when the file is executed as a script.

### Interactive Mode

Here is an example of how to run the system in interactive mode:

```
from ataraxis_video_system import VideoSystem, Camera
from ataraxis_base_utilities import console

if __name__ == "__main__":
    console.enable()
    vs = VideoSystem("img_directory", Camera())
    vs.start(listen_for_keypress=True)
    input()
```
In this example, we enable the console to allow the system to display messages. We then create the VideoSystem object with the listen_for_keypress flag set to True. When started in this way, pressing the "q" key will end image taking and pressing the "w" will end image saving, even if all the images taken have not yet been saved. We use the input() command to prevent the program from terminating and garbage_collecting the video system, which would immediately stop image production and saving.

### Preventing Image Loss

In many use cases, it may be beneficial to stop taking images but keep the video system alive to make sure all images are saved before termination. To do this, you can use the stop_image_production function.

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera())
    vs.start()
    time.sleep(5)
    vs.stop_image_production()
    time.sleep(2)
    vs.stop()
```
In this example, we run the system for 5 seconds before stopping image production; however, we keep the system alive for 2 more seconds before calling stop. This makes sure that all images from the five second session are properly saved before the system is shut down. Note that when running this system with data-intensive cameras for long durations, it may take a long time for the system to save all the images.

### Leveraging Multiprocessing

The VideoSystem uses multiprocessing to speed up image saving and multithreading to maximize process efficiency. You can specify the number of threads and processes used by the video system via the num_processes and num_threads parameters.

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera())
    vs.start(num_processes=5, num_threads=4)
    time.sleep(5)
    vs.stop()
```
This parameter setting creates 5 image saving processes which each have 4 image saving threads. The optimal number of processes and threads per process depend on the computer that the system is running on.

### Different Image Filetypes

Here is an example of saving images in different formats, the defualt is png:

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera(), save_format="png")
    vs.start(save_format="png")
    time.sleep(5)
    vs.stop()
```
The save_format argument of the VideoSystem class can be specified in the creation or starting of the VideoSystem. It only needs to be specified in one of these places. Once specified, all future calls to the start method will also use the specified format.

When the save_format is set to "tiff" for image saving as .tiff files, the compression level can be specified using the tiff_compression_level argument:

```
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera())
    vs.start(save_format="tiff", tiff_compression_level=5)
    time.sleep(5)
    vs.stop()
```

The tiff_compression_level can be set from 0 to 9. The lower the compression level, the faster images will be saved (at the cost of more memory being used per image). Note that tiff saving is lossless at any compression level.

When the save_format is set to "jpg" for image saving as .jpg files, the image quality can be specified using the jpeg_quality argument:

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("img_directory", Camera())
    vs.start(save_format="jpg", jpeg_quality=80)
    time.sleep(5)
    vs.stop()

```
The quality can be set from 0 to 100. The lower the quality, the less memory will be used to save each image.

### Video Saving

If save_format is set to "mp4", the system will save the images to a single mp4 video file.

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("save_directory", Camera())
    config = {"codec": "hevc", "preset": "slow"}
    vs.start(save_format="mp4", mp4_config=config)
    time.sleep(5)
    vs.stop()
```

Video saving is accomplished using an ffmpeg backend. To use this backend, an ffmpeg encodecer must be used. The VideoSystem defaults to an h264 encoder but a different codec can be specified by passing a disctionary of arguments into the mp4_config parameter. Here, codec is set to hevc, an h265 cpu encoder. Any other ffmpeg parameters can also be passed into the video saving by putting them into the config dictionary. The dictionary key should be a string with the parameter name and the dictionary value should be a valid value for that parameter. A full list of ffmpeg codecs and their parameters can be found here: https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb

### Image Video Conversion

Alternatively to directly saving images as an mp4 video file, the data can first be saved as images and then converted to video offline via the save_images_as_vid method.

```
from ataraxis_video_system import VideoSystem, Camera
import time

if __name__ == "__main__":
    vs = VideoSystem("save_directory", Camera())
    vs.start()
    time.sleep(5)
    vs.stop()
    vs.save_imgs_as_vid()
```
If you want to convert a segment of images to video and no longer have access to the original video system object, you can use the static version of the method, called imgs_to_vid, but you need to specify the fps, image filetype, and directories.

```
from ataraxis_video_system import VideoSystem

if __name__ == "__main__":
    VideoSystem.imgs_to_vid(fps=30, img_directory = "img_directory", img_filetype = "png", vid_directory = None)
```
When vid_directory is set to None, the video will be saved in the same folder as the images.

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
