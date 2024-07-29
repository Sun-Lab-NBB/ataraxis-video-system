# AtaraxisVideoSystem

A library that combines OpenCV, GenTL, and FFMPEG to interface with and flexibly record and manipulate the visual data
from a wide range of cameras.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-video-system)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-video-system)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-video-system)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-video-system)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-video-system)
___

## Detailed Description

This library provides a system for efficiently saving image and video data from cameras in real time. This includes the ability to receive data from different camera types and input formats. The system can be configured to save data in different formats including JPEG, PNG, and TIFF. The system can dynamically balance lossless vs lossy compression in order maximize data precision while keeping up with high speed cameras. To accomplish this, the library leverages multiprocessing and can use both CPU and GPU cores.
___

## Features

- Supports Windows, Linux, and OSx.
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

For users, all library dependencies are installed automatically when using the appropriate installation specification
for all supported installation methods (see [Installation](#Installation) section). For developers, see the
[Developers](#Developers) section for information on installing additional development dependencies.
___

## Installation

### Source

**_Note. Building from source may require additional build-components. It is highly advised to use the option to install from PIP or CONDA instead._**

1. Download this repository to your local machine using your preferred method, such as git-cloning. Optionally, use one
   of the stable releases that include precompiled binary wheels in addition to source code.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Run ```python -m pip install .``` to install the project.

### PIP

Use the following command to install the library using PIP:  
```pip install ataraxis-video-system```

### Conda / Mamba

Use the following command to install the library using Conda or Mamba:  
```conda install ataraxis-video-system```
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

See the [API documentation](link) for the
detailed description of the methods and their arguments, exposed through the Videosystem python class.
___

## Developers

This section provides additional installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library. Additionally, it contains instructions for recreating the conda environments
that were used during development from the included .yml files.

### Installing the library

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library. For some
   systems, you may need to use a slightly modified version of this command: ```python -m pip install .[dev]```.
   Alternatively, see the [environments](#environments) section for details on how to create a development environment
   with all necessary dependencies, using a .yml or requirements.txt file.

**Note:** When using tox automation, having a local version of the library may interfere with tox methods that attempt
to build a library using an isolated environment. It is advised to remove the library from your test environment, or
disconnect from the environment, prior to running any tox tasks.

### Additional Dependencies

In addition to installing the python packages, separately install the following dependencies:

- An appropriate build tools or Docker, if you intend to build binary wheels via
  [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/) (See the link for information on which dependencies to
  install).
- [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. Currently,
  this library supports 3.10, 3.11 and 3.12. The easiest way to get tox to work as intended is to have separate
  python distributions, but using [pyenv](https://github.com/pyenv/pyenv) is a good alternative too.

### Development Automation

To help developers, this project comes with a set of fully configured 'tox'-based pipelines for verifying and building
the project. Each of the tox commands builds the project in an isolated environment before carrying out its task.

Below is a list of all available commands and their purpose:

- ```tox -e lint``` Checks and, where safe, fixes code formatting, style, and type-hinting.
- ```tox -e test``` Builds the projects and executes the tests stored in the /tests directory using pytest-coverage
  module.
- ```tox -e docs``` Uses Sphinx to generate API documentation from Python Google-style docstrings. If Doxygen-generated
  .xml files for the C++ extension are available, uses Breathe plugin to convert them to Sphinx-compatible format and
  add
  them to the final API .html file.
- ```tox --parallel``` Carries out all commands listed above in-parallel (where possible). Remove the '--parallel'
  argument to run the commands sequentially. Note, this command will build and test the library for all supported python
  versions.
- ```tox -e build``` Builds the binary wheels for the library for all architectures supported by the host machine.

### Environments

In addition to tox-based automation, all environments used during development are exported as .yml
files and as spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three
supported OS families: Windows 11, OSx 14.5 and Ubuntu Cinnamon 24.04 LTS.

To install the development environment for your OS:

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` into the [envs](envs) folder.
3. Run ```conda env create -f ENVNAME.yml``` or ```mamba env create -f ENVNAME.yml```. Replace 'ENVNAME.yml' with the
   name of the environment you want to install (hpt_dev_osx for OSx, hpt_dev_win64 for Windows and hpt_dev_lin64 for
   Linux). Note, the OSx environment was built against M1 (Apple Silicon) platform and may not work on Intel-based Apple
   devices.

___

## Authors

- Jacob Groner ([Jgroner11](https://github.com/Jgroner11))
- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun Lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.