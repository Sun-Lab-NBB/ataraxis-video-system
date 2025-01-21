"""This module contains a helper script that lists the IDs of cameras accessible through OpenCV or Harvesters backend."""

from pathlib import Path

import click
from ataraxis_base_utilities import console

from .video_system import VideoSystem


@click.command()
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["opencv", "harvesters"]),
    default="opencv",
    required=True,
    help="The camera backend to list IDs for. Note, 'harvesters' backend also requires the path to the .cti file to be"
    "provided through -cti option.",
)
@click.option(
    "-cti",
    "--cti-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="The path to the .cti file. This is required to list 'harvesters' camera ids.",
)
def list_ids(backend: str, cti_path: str) -> None:
    """Lists ids for the cameras available through the selected interface. Subsequently, the IDs from this list can be
    used when instantiating Camera class through API or as ain input to 'live-run' CLI script.

    This method is primarily intended to be used on systems where the exact camera layout is not known. This is
    especially true for the OpenCV id-discovery, which does not provide enough information to identify cameras.
    """
    # Records teh current console status and, if necessary, ensured console is enabled before running this command.
    is_enabled = True
    if not console.enabled:
        is_enabled = False
        console.enable()  # Enables console output

    # Depending on the backend, calls the appropriate ID-discovery command and lists discovered IDs.
    if backend == "opencv":
        opencv_ids = VideoSystem.get_opencv_ids()
        console.echo("Available OpenCV camera IDs:")
        for num, id_string in enumerate(opencv_ids, start=1):
            console.echo(f"{num}: {id_string}")

    elif backend == "harvesters":  # pragma: no cover
        harvester_ids = VideoSystem.get_harvesters_ids(Path(cti_path))
        console.echo("Available Harvesters camera IDs:")
        for num, id_string in enumerate(harvester_ids, start=1):
            console.echo(f"{num}: {id_string}")

    # If the console was enabled by this runtime, ensures it is disabled before finishing the runtime.
    if not is_enabled:
        console.disable()


if __name__ == "__main__":  # pragma: no cover
    list_ids()
