from pathlib import Path

import click
import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from .saver import InputPixelFormats
from .camera import CameraInterfaces, get_opencv_ids, get_harvesters_ids
from .video_system import VideoSystem

console.enable()  # Enables console output

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = dict(max_content_width=120)  # or any width you want


@click.group("axvs", context_settings=CONTEXT_SETTINGS)
def axvs_cli() -> None:
    """This Command-Line Interface (CLI) functions as the entry-point for interfacing with all interactive components
    of the ataraxis-video-system (AXVS) library.
    """


@click.option(
    "-cti",
    "--cti-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="The path to the .cti file that stores the GenTL producer interface. This is required to discover "
    "'harvesters' camera ids.",
)
def add_cti() -> None:
    pass


@axvs_cli.command("id")
@click.option(
    "-i",
    "--interface",
    type=click.Choice(["opencv", "harvesters"]),
    default="opencv",
    required=True,
    help="The camera interface for which to discover and list compatible camera IDs. Note, the 'harvesters' interface "
    "also requires the path to the .cti file to be provided through -cti argument.",
)
@click.option(
    "-cti",
    "--cti-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="The path to the .cti file that stores the GenTL producer interface. This is required to discover "
    "'harvesters' camera ids.",
)
def list_ids(interface: str, cti_path: str) -> None:
    """Lists ids for the cameras available through the selected interface. Subsequently, the IDs from this list can be
    used when instantiating Camera class through API or as ain input to 'live-run' CLI script.

    This method is primarily intended to be used on systems where the exact camera layout is not known. This is
    especially true for the OpenCV id-discovery, which does not provide enough information to identify cameras.
    """
    # Depending on the interface, calls the appropriate ID-discovery command and lists discovered IDs.
    if interface == "opencv":
        opencv_ids = get_opencv_ids()
        console.echo("Available OpenCV camera IDs:")
        for num, id_string in enumerate(opencv_ids, start=1):
            console.echo(f"{num}: {id_string}")

    elif interface == "harvesters":  # pragma: no cover
        harvester_ids = get_harvesters_ids(Path(cti_path))
        console.echo("Available Harvesters camera IDs:")
        for num, id_string in enumerate(harvester_ids, start=1):
            console.echo(f"{num}: {id_string}")


@click.command("run")
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
    "--camera_index",
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
    default=None,
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
    help="The width of the camera frames to acquire, in pixels. Must be a positive integer.",
)
@click.option(
    "-h",
    "--height",
    type=int,
    default=400,
    show_default=True,
    help="The height of the camera frames to acquire, in pixels. Must be a positive integer.",
)
@click.option(
    "-f",
    "--fps",
    type=float,
    default=30.0,
    show_default=True,
    help="The frames per second (FPS) to use for the camera. Must be a positive number.",
)
def live_run(
    camera_backend: str,
    camera_index: int,
    saver_backend: str,
    output_directory: str,
    display_frames: bool,
    monochrome: bool,
    cti_path: str | None,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Instantiates Camera, Saver, and VideoSystem classes using the input parameters and runs the VideoSystem with
    manual control from the user.

    This method uses input() command to allow interfacing with the running system through a terminal window. The user
    has to use the terminal to quite the runtime and enable / disable saving acquired frames to non-volatile memory.

    Notes:
        This CLI script does not support the full range of features offered through the library. Specifically, it does
        not allow fine-tuning Saver class parameters and supports limited Camera and VideoSystem class behavior
        configuration. Use the API for production applications.
    """
    console.enable()  # Enables console output

    # Initializes and starts the DataLogger instance
    logger = DataLogger(output_directory=Path(output_directory))
    logger.start()

    # Initializes the system
    video_system = VideoSystem(
        system_id=np.uint8(111),
        data_logger=logger,
        output_directory=Path(output_directory),
        cti_path=Path(cti_path) if cti_path is not None else None,
    )

    # Adds the requested camera to the VideoSystem
    if camera_backend == "mock":
        video_system._add_camera(
            save_frames=True,
            display_frames=display_frames,
            camera_backend=CameraInterfaces.MOCK,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            acquisition_frame_rate=fps,
            color=not monochrome,
        )
    elif camera_backend == "harvesters":  # pragma: no cover
        video_system._add_camera(
            save_frames=True,
            display_frames=display_frames,
            camera_backend=CameraInterfaces.HARVESTERS,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            acquisition_frame_rate=fps,
        )
    else:  # pragma: no cover
        video_system._add_camera(
            save_frames=True,
            display_frames=display_frames,
            camera_backend=CameraInterfaces.OPENCV,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            acquisition_frame_rate=fps,
            color=not monochrome,
        )

    # Instantiates the requested saver
    if monochrome:
        pixel_color = InputPixelFormats.MONOCHROME  # pragma: no cover
    else:
        pixel_color = InputPixelFormats.BGR

    saver: ImageSaver | VideoSaver
    if saver_backend == "image":
        video_system.add_image_saver(
            image_format=ImageFormats.PNG,
            png_compression=1,
            thread_count=10,
        )
    else:  # pragma: no cover
        video_system.add_saver(
            hardware_encoding=False if saver_backend == "video_cpu" else True,
            preset=CPUEncoderPresets.FAST if saver_backend == "video_cpu" else GPUEncoderPresets.FAST,
            input_pixel_format=pixel_color,
        )

    # Starts the system by spawning child processes
    video_system.start()
    message = "Live VideoSystem: initialized and started (spawned child processes)."
    console.echo(message=message, level=LogLevel.INFO)

    # Ensures that manual control instruction is only shown once
    once: bool = True
    # Ues terminal input to control the video system
    while video_system.started:
        if once:
            message = (
                "Live VideoSystem manual control: activated. Enter 'q' to terminate system runtime."
                "Enter 'w' to start saving camera frames. Enter 's' to stop saving camera frames. After termination, "
                "the system may stay alive for up to 60 seconds to finish saving buffered frames."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
            once = False

        key = input("\nEnter command key:")
        if key.lower()[0] == "q":
            message = "Terminating Live VideoSystem..."
            console.echo(message)
            video_system.stop()
            logger.stop()
        elif key.lower()[0] == "w":  # pragma: no cover
            message = "Starting Live VideoSystem camera frames saving..."
            console.echo(message)
            video_system.start_frame_saving()
        elif key.lower()[0] == "s":  # pragma: no cover
            message = "Stopping Live VideoSystem camera frames saving..."
            console.echo(message)
            video_system.stop_frame_saving()
        else:  # pragma: no cover
            message = (
                f"Unknown input key {key.lower()[0]} encountered while interacting with Live VideoSystem. Use 'q' to "
                f"terminate the system, 'w' to start saving frames, and 's' to stop saving frames."
            )
            console.echo(message, level=LogLevel.WARNING)

    message = (
        f"Live VideoSystem: terminated. Saved frames (if any) are available from the {output_directory} directory."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)
