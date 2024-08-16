"""This module contains the interactive_run script that allows..."""

import click
from ataraxis_base_utilities import console
from pathlib import Path

from .video_system import VideoSystem
from .camera import MockCamera, HarvestersCamera, OpenCVCamera
from .saver import ImageSaver, VideoSaver, CPUEncoderPresets, GPUEncoderPresets


@click.command()
@click.option(
    "-cb",
    "--camera-backend",
    type=click.Choice(["mock", "harvesters", "opencv"]),
    default="mock",
    help="The Camera class backend to use.",
)
@click.option(
    "-sb",
    "--saver-backend",
    type=click.Choice(["image", "video_cpu", "video_gpu"]),
    default="video_cpu",
    help="The Saver class backend to use.",
)
@click.option(
    "-od",
    "--output-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="The path to the output directory where acquired frames will be saved as images or videos.",
)
@click.option(
    "-df",
    "--display-frames",
    is_flag=True,
    default=True,
    help="Determines whether to display acquired frames in real time.",
)
@click.option(
    "-cti",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the .cti file. This is mandatory for Harvesters cameras",
)
def interactive_run(camera_backend, saver_backend, output_directory, display_frames, cti):
    """Run the interactive video system."""
    console.enable()  # Enables console output

    # Instantiates the requested camera
    if camera_backend == "mock":
        camera = MockCamera(name='Interactive Camera')
    elif camera_backend == "harvesters":
        camera = HarvestersCamera(name='Interactive Camera', cti_path=Path(cti))
    else:
        camera = OpenCVCamera(name='Interactive Camera')

    # Instantiates the requested saver
    if saver_backend == "image":
        saver = ImageSaver(
            output_directory=Path(output_directory),
        )
    elif saver_backend == "video_cpu":
        saver = VideoSaver(
            output_directory=Path(output_directory),
            hardware_encoding=False,
            preset=CPUEncoderPresets.FAST,
        )
    else:
        saver = VideoSaver(
            output_directory=Path(output_directory),
            hardware_encoding=True,
            preset=GPUEncoderPresets.FAST,
        )

    video_system = VideoSystem(
        camera=camera,
        saver=saver,
        shutdown_timeout=60,
        system_name="interactive_video_system",
        display_frames=display_frames,
    )
    video_system.start()
    console.enable() # Enables console output

    # Ues terminal input to control the video system
    while video_system.is_running:
        key = input("Enter command key:")
        if key.lower()[0] == 'q':
            message = f"Terminating VideoSystem {video_system.name}"""
            console.echo(message)
            video_system.stop()
        elif key.lower()[0] == 'w':
            message = f"Starting VideoSystem {video_system.name} camera frames saving"""
            console.echo(message)
            video_system.start_frame_saving()
        elif key.lower()[0] == 's':
            message = f"Stopping VideoSystem {video_system.name} camera frames saving"""
            console.echo(message)
            video_system.stop_frame_saving()
        else:
            message = (
                f"Unknown input key {key.lower()[0]} encountered while interacting with VideoSystem "
                f"{video_system.name}. Use 'q' to terminate the system, 'w' to start saving frames, and 's' to stop "
                f"saving frames."
            )
            console.echo(message)


if __name__ == "__main__":
    interactive_run()
