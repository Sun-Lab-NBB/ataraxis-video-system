# ataraxis-video-system

A Python library that interfaces with a wide range of cameras to flexibly record visual stream data as video files.

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

This library abstracts all necessary steps for acquiring and saving video data. During each runtime, it interfaces with 
one or more cameras to grab the raw frames and encodes them as video files stored in the non-volatile memory. The 
library is specifically designed for working with multiple cameras at the same time and supports fine-tuning the 
acquisition and saving parameters to precisely balance the resultant video quality and real-time throughput for 
a wide range of applications.

___

## Features

- Supports Windows, Linux, and macOS.
- Uses OpenCV or GeniCam (Harvesters) to interface with a wide range of consumer, industrial, and scientific cameras.
- Uses FFMPEG to efficiently encode acquired data as videos in real time using CPU or GPU.
- Highly customizable and can be extensively fine-tuned for quality or throughput.
- Includes an MCP server for AI agent integration (compatible with Claude Desktop and other MCP clients).
- GPL 3 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [MCP Server](#mcp-server-agentic-integration)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)

___

## Dependencies

- [FFMPEG](https://www.ffmpeg.org/download.html) version **n8.0**. The installed FFMPEG must be available on the 
  system’s path and callable from Python processes.
- A [GenTL Producer](https://www.emva.org/wp-content/uploads/GenICam_GenTL_1_6.pdf) interface compatible with the 
  [Harvesters](https://github.com/genicam/harvesters/blob/master/docs/INSTALL.rst#installing-a-gentl-producer) library 
  if the target camera requires the 'harvesters' camera interface. It is recommended to use the CTI interface supplied 
  by the camera’s vendor, if possible, as this typically ensures that the camera performs as advertised. If the 
  camera-specific CTI file is not available, it is possible to instead use a general interface, such as 
  [MvImpactAcquire](https://assets-2.balluff.com/mvIMPACT_Acquire/). This library has been tested using MvImpactAcquire 
  version **2.9.2**.

For users, all other library dependencies are installed automatically by all supported installation methods 
(see the [Installation](#installation) section).

***Note!*** Developers should see the [Developers](#developers) section for information on installing additional 
development dependencies.

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the 
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-video-system/tags) that include precompiled binary and 
   source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### pip

Use the following command to install the library using pip: ```pip install ataraxis-video-system```

___

## Usage

### OS Support Status

While this library works on all major operating systems, it is largely up to the maintainers of the low-level library 
components (OpenCV, Harvesters, FFMPEG) to ensure that the operation is smooth on each supported OS. That, in turn, is 
not always possible for many nuanced reasons. This section summarizes the current state of the library for the three 
explicitly supported operating systems: macOS, Windows, and Linux.

#### Linux
This library was primarily written on and for Linux systems. It is extensively tested on Linux and performs well under
all test conditions. It is very likely that Linux users will not experience any issues specific to this library.

#### Windows
The library is mostly stable on Windows systems, but requires additional setup to ensure smooth operation. First, the 
FFMPEG **has** to be updated to the latest stable version, as older versions may have a drastically reduced encoding 
speed even with hardware acceleration. Additionally, some of the advanced OpenCV’s features, such as the MSMF HW 
transformations, have to be disabled to support smooth runtimes on the Windows platform. Typically, the information of 
which features to disable is readily available from the OpenCV’s Windows community.

#### macOS
macOS mostly works as expected except for live frame displaying, which does not work for modern macOS devices. The issue
is due to the OS restriction on drawing certain GUI elements outside the main thread of the application. The restriction
interferes with the library, as it displays the acquired frames from the same process that interfaces with the camera 
to minimize the visual lag between grabbing and displaying the frame. This is a persistent issue that is unlikely to 
be fixed any time soon.

### Quickstart
This is a minimal example of how to use this library. It is also available as a [script](examples/quickstart.py).
This example is intentionally kept minimal, consult the
[API documentation](https://ataraxis-video-system-api-docs.netlify.app/) for all available VideoSystem configuration 
parameters.

Most library functionality is accessible through the **VideoSystem** class:
```
from pathlib import Path

import numpy as np
import tempfile
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import DataLogger, assemble_log_archives
from ataraxis_base_utilities import console, LogLevel

from ataraxis_video_system import VideoSystem, VideoEncoders, CameraInterfaces, extract_logged_camera_timestamps

# Since the VideoSystem and DataLogger classes use multiprocessing under-the-hood, the runtime must be protected by the
# __main__ guard.
if __name__ == "__main__":

    # Enables the console module to communicate the example's runtime progress via the terminal.
    console.enable()

    # Specifies the directory where to save the acquired video frames and timestamps.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes
    output_directory = Path(tempdir.name)

    # The DataLogger is used to save frame acquisition timestamps to disk as uncompressed .npy files.
    logger = DataLogger(output_directory=output_directory, instance_name="webcam")

    # The DataLogger has to be started before it can save any log entries.
    logger.start()

    # The VideoSystem minimally requires an ID and a DataLogger instance. The ID is critical, as it is used to identify
    # the log entries generated by the VideoSystem. For VideoSystems that will be saving frames, output_directory is
    # also required
    vs = VideoSystem(
        system_id=np.uint8(101),
        data_logger=logger,
        output_directory=output_directory,
        camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
        display_frame_rate=15,  # Displays the acquired data at a rete of 15 frames per second
        color=False,  # Acquires images in MONOCHROME mode
        video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
        quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
    )

    # Calling this method arms the video system and starts frame acquisition. However, the frames are not initially
    # saved to disk.
    vs.start()
    console.echo(f"VideoSystem: Started", level=LogLevel.SUCCESS)

    console.echo(f"Acquiring frames without saving...")
    timer = PrecisionTimer("s")
    timer.delay(delay=5, block=False)  # During this delay, camera frames are displayed to the user but are not saved

    # Begins saving frames to disk as an MP4 video file
    console.echo(f"Saving the acquired frames to disk...")
    vs.start_frame_saving()
    timer.delay(delay=5, block=False)  # Records frames for 5 seconds, generating ~150 frames
    vs.stop_frame_saving()

    # Frame acquisition can be started and stopped as needed, although all frames are written to the same output
    # video file.

    # Stops the VideoSystem runtime and releases all resources
    vs.stop()
    console.echo(f"VideoSystem: Stopped", level=LogLevel.SUCCESS)

    # Stops the DataLogger and assembles all logged data into a single .npz archive file. This step is required to be
    # able to extract the timestamps for further analysis.
    logger.stop()
    console.echo(f"Assembling the frame timestamp log archive...")
    assemble_log_archives(remove_sources=True, log_directory=logger.output_directory, verbose=True)

    # Extracts the list of frame timestamps from the assembled log archive generated above. This returns a list of
    # timestamps. Each is given in microseconds elapsed since the UTC epoch onset.
    console.echo(f"Extracting frame acquisition timestamps from the assembled log archive...")
    timestamps = extract_logged_camera_timestamps(log_path=logger.output_directory.joinpath(f"101_log.npz"))

    # Computes and prints the frame rate of the camera based on the extracted frame timestamp data.
    timestamp_array = np.array(timestamps, dtype=np.uint64)
    time_diffs = np.diff(timestamp_array)
    fps = 1 / (np.mean(time_diffs) / 1e6)
    console.echo(
        message=(
            f"According to the extracted timestamps, the interfaced camera had an acquisition frame rate of "
            f"approximately {fps:.2f} frames / second."
        ),
        level=LogLevel.SUCCESS,
    )

    # Cleans up the temporary directory before shutting the runtime down.
    tempdir.cleanup()
```

### Data Logging
This library relies on the [DataLogger](https://github.com/Sun-Lab-NBB/ataraxis-data-structures#datalogger) class to 
save frame acquisition timestamps to disk during runtime. Each **saved** frame’s acquisition timestamp is serialized 
and saved as an uncompressed **.npy** file.

The same DataLogger instance as used by the VideoSystem instances may be shared by multiple other Ataraxis assets that 
generate log entries, such as 
[MicroControllerInterface](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface) instances. To support using 
the same logger instance for multiple concurrently active sources, **each source has to use a unique identifier value
(system id) when sending data to the logger instance**.

#### Log Format
Each frame’s acquisition timestamp is logged as a one-dimensional numpy uint8 array, saved as an .npy file. Inside the 
array, the data is organized in the following order:
1. The uint8 id of the data source (video system instance). The ID occupies the first byte of each log entry.
2. The uint64 timestamp that specifies the number of microseconds elapsed since the acquisition of the **onset** 
   timestamp (see below). The timestamp occupies **8** bytes following the ID byte. This value communicates when each 
   saved camera frame has been acquired.

**Note!** Timestamps are generated at frame acquisition but are only submitted to the logger when the corresponding 
frame is saved to disk. Therefore, the timestamps always match the order that the saved frames appear in the video file.

#### Onset Timestamp
Each VideoSystem generates an `onset` timestamp as part of its `start()` method runtime. This log entry uses a modified 
data order and stores the current UTC time, accurate to microseconds, as the total number of microseconds elapsed since
the UTC epoch onset. All further log entries for the same source use the timestamp section of their payloads to 
communicate the number of microseconds elapsed since the onset timestamp acquisition. 

The onset log entry uses the following data organization order:
1. The uint8 id of the data source (video system instance).
2. The uint64 value **0** that occupies 8 bytes following the source id. A 'timestamp' value of 0 universally indicates 
   that the log entry stores the onset timestamp.
3. The uint64 value that stores the number of microseconds elapsed since the UTC epoch onset. This value specifies the 
   current time when the onset timestamp was generated.

#### Working with VideoSystem Logs
See the [quickstart](#quickstart) example above for a demonstration on how to assemble and parse the frame acquisition
log archives generated by the VideoSystem instance at runtime. 

**Note!** The parsed frame acquisition timestamps are returned as a tuple of values that match the order in which the 
frames were saved to disk as an .mp4 file. Each timestamp is given as the number of microseconds elapsed since the UTC 
epoch onset.

### CLI

This library exposes the `axvs` Command-Line Interface (CLI) as part of its installation into a Python environment. To
see the list of available CLI commands, call the `axvs --help` command from the environment that has the library
installed or see the API documentation below.

### MCP Server (Agentic Integration)

This library includes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables AI agents
to programmatically interact with camera discovery, configuration, and video recording functionality. The MCP server
exposes the following tools:

**Camera Discovery and Configuration:**
- **list_cameras**: Discovers all cameras compatible with OpenCV and Harvesters interfaces.
- **get_cti_status**: Checks whether the library is configured with a valid GenTL Producer (.cti) file.
- **set_cti_file**: Configures the library to use a specified CTI file for GeniCam camera support.

**Runtime Requirements:**
- **check_runtime_requirements**: Checks FFMPEG and GPU availability for video encoding.

**Video Session Management:**
- **start_video_session**: Starts a video capture session with specified camera and encoding parameters.
- **stop_video_session**: Stops the active video capture session and releases resources.
- **start_frame_saving**: Begins saving captured frames to a video file.
- **stop_frame_saving**: Stops saving frames while keeping the session active.
- **get_session_status**: Returns the current status of the video session.

To start the MCP server, use the `axvs mcp` command. For integration with Claude Desktop, add the following to the
Claude Desktop configuration file (`~/.config/claude/claude_desktop_config.json` on Linux,
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or
`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "ataraxis-video-system": {
      "command": "axvs",
      "args": ["mcp"]
    }
  }
}
```

### Using GeniCam Compatible Cameras
This library supports all cameras compatible with the [GeniCam](https://www.emva.org/standards-technology/genicam/) 
standard, which includes most GigE+ scientific and machine vision cameras. 

**Note!** Before using the library with a GeniCam camera, it must be provided with the path to the .cti GenTL Producer
Interface file. Without an interface, the library is not able to interface with the GeniCam cameras. Use the 
`axvs cti` CLI command to configure the library to use the .cti file provided by the camera vendor (preferred) or a 
general .cti file, such as [mvImpactAcquire](#dependencies). This command only needs to be called once, as the library 
remembers and reuses the provided .cti file for all future runtimes.

___

## API Documentation

See the [API documentation](https://ataraxis-video-system-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.

**Note!** The API documentation also includes the details about the `axvs` CLI interface exposed by this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for project developers.

### Installing the Project

***Note!*** This installation method requires **mamba version 2.3.2 or above**. Currently, all Sun lab automation 
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the 
   ```mamba install tox uv tox-uv``` command.
5. Use the ```tox -e create``` command to create the project-specific development environment followed by 
   ```tox -e install``` command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project. 
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like 
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check the [tox.ini file](tox.ini) for details about the 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All pull requests for this project have to successfully complete the ```tox``` task before being merged. 
To expedite the task’s runtime, use the ```tox --parallel``` command to run some tasks in-parallel.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself may experience runtime failures. In 
most cases, this is related to their caching behavior. If an unintelligible error is encountered with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a CLI command typically solves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-video-system/tags) for the available project 
releases.

___

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
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

___
