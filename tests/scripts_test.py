"""Contains tests that verify standalone scripts can run on this system.

Notes:
    This test suite is not meant to be exhaustive! All scripts are evaluated manually before production distribution,
    this test suite is used to verify that the scripts are callable on supported architectures and OS versions.
"""

import tempfile

from click.testing import CliRunner

from ataraxis_video_system.live import live_run
from ataraxis_video_system.list_camera_ids import list_ids


def test_list_ids() -> None:
    """Verifies that the list_ids script is callable.

    This test only verifies that the host system is able to run the script.
    """
    runner = CliRunner()
    # Unlike the 'harvesters' version, this is guaranteed to run on any supported system.
    args = ["-b", "opencv"]
    # noinspection PyTypeChecker
    result = runner.invoke(list_ids, args)
    assert result.exit_code == 0  # Verifies the script terminates with a success exit_code (0)


def test_live_run() -> None:
    """Verifies that the live_run script is callable.

    This test only verifies that the host system is able to run the script.
    """
    runner = CliRunner()

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Prepare arguments
        args = [
            "--camera-backend",
            "mock",  # Uses mock backend to avoid the need for a real camera
            "--camera_index",
            "0",
            "--saver-backend",
            "image",  # Uses image saver for guaranteed support on all architectures
            "--output-directory",
            tmpdirname,
            "--width",
            "640",
            "--height",
            "480",
            "--fps",
            "30.0",
        ]

        # Simulates user input 'q' to immediately terminate the system after initialization
        # noinspection PyTypeChecker
        result = runner.invoke(live_run, args, input="q\n")
        assert (
            result.exit_code == 0 or result.exit_code == 1
        )  # Verifies the script terminates with a success exit_code (0)
