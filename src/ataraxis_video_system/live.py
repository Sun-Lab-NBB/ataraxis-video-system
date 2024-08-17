"""This module contains a CLI script that can be used to instantiate a VideoSystem from command-line interface and
manually control its runtime.

Primarily, this is helpful for manual and semi-automated evaluation of class performance that benefits from the ability
to manually control the system. This is also helpful for mapping available camera IDs to physical cameras, as this CLI
command can be configured to display visual stream from a camera without saving frames.

For production applications, it is highly recommended to use the VideoSystem API, instead of the CLI command available
through this module. Primarily, this is because the API allows fine-tuning the behavior of the Saver class, which is
not possible through this CLI script. Almost every use-case will benefit from using an optimally tuned Saver class.
"""

from typing import Any
from pathlib import Path

import click
from ataraxis_base_utilities import LogLevel, console

from .saver import ImageSaver, VideoSaver, ImageFormats, CPUEncoderPresets, GPUEncoderPresets, InputPixelFormats
from .camera import CameraBackends
from .video_system import VideoSystem


def _validate_positive_int(_ctx: Any, _param: Any, value: Any) -> int | None:
    """Ensures that the provided integer value is positive."""
    if value is not None and value <= 0:
        raise click.BadParameter("Must be a positive integer.")

    if value is not None:
        return int(value)

    else:
        return None


def _validate_positive_float(_ctx: Any, _param: Any, value: Any) -> float | None:
    """Ensures that the provided float value is positive."""
    if value is not None and value <= 0:
        raise click.BadParameter("Must be a positive number (integer or float).")

    if value is not None:
        return float(value)

    else:
        return None


@click.command()
@click.option(
    "-c",
    "--camera-backend",
    type=click.Choice(["mock", "harvesters", "opencv"]),
    default="mock",
    show_default=True,
    help="The backend to use for the Camera class. For Gen-TL compatible cameras, it is recommended to use "
    "'harvesters' backend. When using 'harvesters' backend, make sure to provide the path to the .cti file "
    "through the -cti option.",
)
@click.option(
    "-i",
    "--camera_id",
    type=int,
    default=0,
    show_default=True,
    help="The list-id of the camera to use, for 'opencv' and 'harvesters' backends. This option allows selecting the "
    "desired camera, if multiple are available on the host-system.",
)
@click.option(
    "-s",
    "--saver-backend",
    type=click.Choice(["image", "video_cpu", "video_gpu"]),
    default="video_cpu",
    show_default=True,
    help="The backend to use for the Saver class. If the host-system has an NVIDIA GPU, it is recommended to use the"
    "video_gpu backend which efficiently encodes acquired camera frames as a video file. Image backend will save"
    "camera frames as individual images, using 1 CPU core and multiple threads.",
)
@click.option(
    "-o",
    "--output-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="The path to the output directory where to save the acquired frames.",
)
@click.option(
    "-d",
    "--display-frames",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display acquired frames in real time.",
)
@click.option(
    "-m",
    "--monochrome",
    is_flag=True,
    default=False,
    show_default=True,
    help="Determines whether the camera records frames in monochrome (grayscale) or colored spectrum.",
)
@click.option(
    "-cti",
    "--cti_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="The path to the .cti file, which is sued by the 'harvesters' camera backend. This is required for "
    "'harvesters' backend to work as expected.",
)
@click.option(
    "-w",
    "--width",
    type=int,
    default=600,
    show_default=True,
    callback=_validate_positive_int,
    help="The width of the camera frames to acquire, in pixels. Must be a positive integer.",
)
@click.option(
    "-h",
    "--height",
    type=int,
    default=400,
    show_default=True,
    callback=_validate_positive_int,
    help="The height of the camera frames to acquire, in pixels. Must be a positive integer.",
)
@click.option(
    "-f",
    "--fps",
    type=float,
    default=30.0,
    show_default=True,
    callback=_validate_positive_float,
    help="The frames per second (FPS) to use for the camera. Must be a positive number.",
)
def live_run(
    camera_backend: str,
    camera_id: int,
    saver_backend: str,
    output_directory: str,
    display_frames: bool,
    monochrome: bool,
    cti_path: str,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Instantiates Camera, Saver and VideoSystem classes using the input parameters and runs the VideoSystem with
    manual control from the user.

    This method uses input() command to allow interfacing with the running system through a terminal window. The user
    has to use the terminal to quite the runtime and enable / disable saving acquired frames to non-volatile memory.

    Notes:
        This CLI script does not support the full range of features offered through the library. Specifically, it does
        not allow fine-tuning Saver class parameters and supports limited Camera and VideoSystem class behavior
        configuration. Use the API for production applications.
    """

    console.enable()  # Enables console output

    # Instantiates the requested camera
    camera_name = "Live Camera"
    if camera_backend == "mock":
        camera = VideoSystem.create_camera(
            camera_name=camera_name,
            camera_backend=CameraBackends.MOCK,
            camera_id=camera_id,
            frame_width=width,
            frame_height=height,
            frames_per_second=fps,
            color=monochrome,
        )
    elif camera_backend == "harvesters":
        camera = VideoSystem.create_camera(
            camera_name=camera_name,
            camera_backend=CameraBackends.HARVESTERS,
            camera_id=camera_id,
            frame_width=width,
            frame_height=height,
            frames_per_second=fps,
            cti_path=Path(cti_path),
        )
    else:
        camera = VideoSystem.create_camera(
            camera_name=camera_name,
            camera_backend=CameraBackends.OPENCV,
            camera_id=camera_id,
            frame_width=width,
            frame_height=height,
            frames_per_second=fps,
            color=monochrome,
        )

    # Instantiates the requested saver
    if monochrome:
        pixel_color = InputPixelFormats.MONOCHROME
    else:
        pixel_color = InputPixelFormats.BGR

    saver: ImageSaver | VideoSaver
    if saver_backend == "image":
        saver = VideoSystem.create_image_saver(
            output_directory=Path(output_directory),
            image_format=ImageFormats.PNG,
            png_compression=1,
            thread_count=10,
        )
    elif saver_backend == "video_cpu":
        saver = VideoSystem.create_video_saver(
            output_directory=Path(output_directory),
            hardware_encoding=False,
            preset=CPUEncoderPresets.FAST,
            input_pixel_format=pixel_color,
        )
    else:
        saver = VideoSystem.create_video_saver(
            output_directory=Path(output_directory),
            hardware_encoding=True,
            preset=GPUEncoderPresets.FAST,
            input_pixel_format=pixel_color,
        )

    # Initializes the system
    video_system = VideoSystem(
        camera=camera,
        saver=saver,
        shutdown_timeout=60,  # Intentionally kept low, unlike the API default
        system_name="Live",
        display_frames=display_frames,
    )

    # Starts the system by spawning child processes
    video_system.start()
    message = f"VideoSystem {video_system.name}: initialized and started (spawned child processes)."
    console.echo(message=message, level=LogLevel.INFO)

    # Ensures that manual control instruction is only shown once
    once: bool = True
    # Ues terminal input to control the video system
    while video_system.is_running:
        if once:
            message = (
                f"VideoSystem {video_system.name} manual control: activated. Enter 'q' to terminate system runtime."
                f"Enter 'w' to start saving camera frames. Enter 's' to stop saving camera frames. After termination, "
                f"the system may stay alive for up to 60 seconds to finish saving buffered frames."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
            once = False

        key = input("\nEnter command key:")
        if key.lower()[0] == "q":
            message = f"Terminating VideoSystem {video_system.name}..."
            console.echo(message)
            video_system.stop()
        elif key.lower()[0] == "w":
            message = f"Starting VideoSystem {video_system.name} camera frames saving..."
            console.echo(message)
            video_system.start_frame_saving()
        elif key.lower()[0] == "s":
            message = f"Stopping VideoSystem {video_system.name} camera frames saving..."
            console.echo(message)
            video_system.stop_frame_saving()
        else:
            message = (
                f"Unknown input key {key.lower()[0]} encountered while interacting with VideoSystem "
                f"{video_system.name}. Use 'q' to terminate the system, 'w' to start saving frames, and 's' to stop "
                f"saving frames."
            )
            console.echo(message, level=LogLevel.WARNING)

    message = (
        f"VideoSystem {video_system.name}: terminated. Saved frames (if any) are available from the "
        f"{output_directory} directory."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


if __name__ == "__main__":
    live_run()
