"""Provides shared fixtures for all test modules."""

import sys
import json
from pathlib import Path
import platform
import warnings
from contextlib import suppress
import subprocess
from dataclasses import asdict
from collections.abc import Generator

import pytest
import filelock
import platformdirs

from ataraxis_video_system import (
    CameraInterfaces,
    CameraInformation,
    GenicamConfiguration,
    discover_camera_ids,
    check_ffmpeg_availability,
)
from ataraxis_video_system.video.camera import (
    _CTI_PATH_VARIABLE,
    HarvestersCamera,
    genicam_runtime_available,
)

_SIMULATOR_ROOT: Path = Path(__file__).parent / "gentl_simulator"
"""Stores the path to the directory holding the vendored TLSimu GenTL Producer simulator binaries."""


@pytest.fixture(scope="session")
def simulator_cti_path() -> Path | None:
    """Provides the path to the bundled GenTL Producer simulator, or None when the host has no bundled build."""
    return _resolve_simulator_cti()


@pytest.fixture
def gentl_simulator(simulator_cti_path: Path | None, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the library at the bundled GenTL Producer simulator for the duration of a single test.

    The override is applied through the environment so that it also reaches the processes VideoSystem spawns, and it
    leaves the CTI path persisted for the user's real hardware untouched.
    """
    # The simulator is driven through the GenICam camera interface, so a bundled Producer is of no use on a platform
    # that installs no runtime to load it with. Every Mac bundles a Producer while only some install a runtime, so the
    # two conditions are checked separately.
    if not genicam_runtime_available():
        pytest.skip("Skipping this test as this platform does not support the GenICam camera interface.")

    if simulator_cti_path is None:
        pytest.skip("Skipping this test as no GenTL Producer simulator is bundled for this platform.")

    monkeypatch.setenv(name=_CTI_PATH_VARIABLE, value=str(simulator_cti_path))
    return simulator_cti_path


@pytest.fixture
def persisted_cti_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects the persisted GenTL Producer path file into a temporary directory for a single test.

    The runtime override is removed alongside the redirection, so that the path resolution under test reaches the
    persisted file rather than the environment. Both together keep a test that configures a Producer away from the one
    the developer's own machine has configured.
    """
    monkeypatch.delenv(name=_CTI_PATH_VARIABLE, raising=False)
    directory = tmp_path / "user_data"
    directory.mkdir()
    monkeypatch.setattr(target=platformdirs, name="user_data_dir", value=lambda **_kwargs: str(directory))
    return directory


@pytest.fixture(scope="session")
def has_opencv(_all_cameras: tuple[CameraInformation, ...]) -> bool:
    """Checks for OpenCV camera availability using cached discovery results."""
    return any(camera.interface == CameraInterfaces.OPENCV for camera in _all_cameras)


@pytest.fixture(scope="session")
def has_harvesters(_all_cameras: tuple[CameraInformation, ...]) -> Generator[bool, None, None]:
    """Checks for Harvesters camera availability and sandboxes the camera configuration for the whole session.

    Captures every writable GenICam node outside the default blacklist before any test runs and writes it back once
    every test has finished, so that tests which reconfigure the camera leave no trace on the hardware.
    """
    harvesters_cameras = [camera for camera in _all_cameras if camera.interface == CameraInterfaces.HARVESTERS]
    has = bool(harvesters_cameras)

    # Captures the writable configuration outside the default blacklist, so that every node a test writes through the
    # configuration API can be restored.
    saved_configuration: GenicamConfiguration | None = None
    if has:
        camera = HarvestersCamera(system_id=222, camera_index=0)
        try:
            camera.connect()
            saved_configuration = camera.get_configuration()
        except Exception as error:
            warnings.warn(
                message=f"Failed to capture the GenICam camera configuration before the test session: {error!r}",
                stacklevel=2,
            )
        finally:
            with suppress(Exception):
                camera.disconnect()

    yield has

    if saved_configuration is not None:
        _restore_camera_configuration(saved_configuration=saved_configuration)


@pytest.fixture(scope="session")
def has_nvidia() -> bool:
    """Checks for NVIDIA GPU availability in the test environment."""
    try:
        subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception:
        return False
    else:
        return True


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    """Checks for FFMPEG availability in the test environment."""
    return check_ffmpeg_availability()


def _resolve_simulator_cti() -> Path | None:
    """Resolves the path to the bundled GenTL Producer simulator built for the host platform.

    Returns:
        The path to the platform-appropriate TLSimu.cti file, or None if no build is bundled for the host.
    """
    machine = platform.machine().lower()

    # The macOS binary is universal, so it covers both Intel and Apple Silicon hosts.
    if sys.platform == "darwin":
        directory = _SIMULATOR_ROOT / "macos_universal2"
    elif machine not in {"x86_64", "amd64"}:
        return None
    elif sys.platform == "win32":
        directory = _SIMULATOR_ROOT / "windows_amd64"
    elif sys.platform.startswith("linux"):
        directory = _SIMULATOR_ROOT / "linux_x86_64"
    else:
        return None

    cti_path = directory / "TLSimu.cti"
    return cti_path if cti_path.exists() else None


def _restore_camera_configuration(saved_configuration: GenicamConfiguration) -> None:
    """Writes the saved GenICam configuration back onto the connected camera.

    Notes:
        Identity validation runs in strict mode so that a configuration captured from one camera is never written onto
        a different camera that happens to occupy the same index at teardown.
    """
    camera = HarvestersCamera(system_id=222, camera_index=0)
    try:
        camera.connect()
        camera.apply_configuration(config=saved_configuration, strict_identity=True)
    except Exception as error:
        warnings.warn(
            message=f"Failed to restore the GenICam camera configuration after the test session: {error!r}",
            stacklevel=2,
        )
    finally:
        with suppress(Exception):
            camera.disconnect()


@pytest.fixture(scope="session")
def _all_cameras(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[CameraInformation, ...]:
    """Discovers all cameras once per xdist worker cluster using file-based locking.

    Serializes camera discovery across pytest-xdist workers so that only the first worker probes hardware. Subsequent
    workers read cached results from a shared JSON file, preventing concurrent exclusive-access conflicts on USB camera
    devices. Discovery runs with the simulator override removed so that simulated devices are never mistaken for
    hardware.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(name=_CTI_PATH_VARIABLE, raising=False)

        # When not running under xdist (worker_id == "master"), discovers cameras directly without locking.
        if worker_id == "master":
            try:
                return discover_camera_ids()
            except Exception:
                return ()

        # Resolves the shared temp directory that all xdist workers can access. The parent of each worker's basetemp
        # is shared across the entire test session.
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        cache_file = root_tmp_dir / "camera_discovery.json"
        lock_file = root_tmp_dir / "camera_discovery.lock"

        with filelock.FileLock(lock_file=str(lock_file), timeout=120):
            if cache_file.exists():
                # Reads cached discovery results written by the first worker.
                data = json.loads(cache_file.read_text())
                return tuple(CameraInformation(**entry) for entry in data)

            # Runs the hardware discovery on the first worker to acquire the lock, as the probe tolerates one owner at
            # a time.
            try:
                all_cameras = discover_camera_ids()
            except Exception:
                all_cameras = ()

            # Caches discovery results as JSON for other workers.
            cache_file.write_text(json.dumps([asdict(camera) for camera in all_cameras]))

        return all_cameras
