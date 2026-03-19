from enum import StrEnum
from typing import Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Generator

import cv2
import numpy as np
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from genicam.genapi import NodeMap as NodeMap
from harvesters.core import (
    Harvester,
    ImageAcquirer as ImageAcquirer,
)

from .saver import InputPixelFormats as InputPixelFormats
from .configuration import (
    DEFAULT_BLACKLISTED_NODES as DEFAULT_BLACKLISTED_NODES,
    GenicamNodeInfo as GenicamNodeInfo,
    GenicamConfiguration as GenicamConfiguration,
    read_genicam_node as read_genicam_node,
    write_genicam_node as write_genicam_node,
    format_genicam_node as format_genicam_node,
    enumerate_genicam_nodes as enumerate_genicam_nodes,
    apply_genicam_configuration as apply_genicam_configuration,
)

_MONOCHROME_FORMATS: set[Any]
_COLOR_FORMATS: set[Any]
_ALL_RGB_FORMATS: set[Any]
_FRAME_POOL_SIZE: int
_MAXIMUM_NON_WORKING_IDS: int

class CameraInterfaces(StrEnum):
    HARVESTERS = "harvesters"
    OPENCV = "opencv"
    MOCK = "mock"

@dataclass(frozen=True, slots=True)
class CameraInformation:
    camera_index: int
    interface: CameraInterfaces | str
    frame_width: int
    frame_height: int
    acquisition_frame_rate: int
    serial_number: str | None = ...
    model: str | None = ...

def discover_camera_ids() -> tuple[CameraInformation, ...]: ...
def add_cti_file(cti_path: Path) -> None: ...
def check_cti_file() -> Path | None: ...

class OpenCVCamera:
    _system_id: int
    _color: bool
    _camera_index: int
    _frame_rate: int
    _frame_width: int
    _frame_height: int
    _camera: cv2.VideoCapture | None
    _acquiring: bool
    def __init__(
        self,
        system_id: int,
        camera_index: int = 0,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        *,
        color: bool = True,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_acquiring(self) -> bool: ...
    @property
    def frame_rate(self) -> int: ...
    @property
    def frame_width(self) -> int: ...
    @property
    def frame_height(self) -> int: ...
    @property
    def pixel_color_format(self) -> InputPixelFormats: ...
    def grab_frame(self) -> NDArray[np.floating[Any] | np.integer[Any]]: ...

class HarvestersCamera:
    _system_id: int
    _camera_index: int
    _frame_rate: int
    _frame_width: int
    _frame_height: int
    _harvester: Harvester | None
    _camera: ImageAcquirer | None
    _color: bool
    _model: str
    _serial_number: str
    def __init__(
        self,
        system_id: int,
        camera_index: int = 0,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_acquiring(self) -> bool: ...
    @property
    def frame_rate(self) -> int: ...
    @property
    def frame_width(self) -> int: ...
    @property
    def frame_height(self) -> int: ...
    @property
    def pixel_color_format(self) -> InputPixelFormats: ...
    @property
    def model(self) -> str: ...
    @property
    def serial_number(self) -> str: ...
    @property
    def node_map(self) -> NodeMap: ...
    def get_node_info(self, name: str) -> GenicamNodeInfo: ...
    def get_node_description(self, name: str) -> str: ...
    def set_node_value(self, name: str, value: str) -> None: ...
    def get_configuration(self, blacklisted_nodes: frozenset[str] = ...) -> GenicamConfiguration: ...
    def apply_configuration(
        self, config: GenicamConfiguration, *, strict_identity: bool = False, blacklisted_nodes: frozenset[str] = ...
    ) -> None: ...
    def grab_frame(self) -> NDArray[np.integer[Any]]: ...

class MockCamera:
    _system_id: int
    _color: bool
    _frame_rate: int
    _frame_width: int
    _frame_height: int
    _camera: bool
    _acquiring: bool
    _frames: tuple[NDArray[np.uint8], ...]
    _current_frame_index: int
    _timer: PrecisionTimer | None
    _time_between_frames: float
    def __init__(
        self,
        system_id: int,
        frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        *,
        color: bool = True,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_acquiring(self) -> bool: ...
    @property
    def frame_rate(self) -> int: ...
    @property
    def frame_width(self) -> int: ...
    @property
    def frame_height(self) -> int: ...
    @property
    def frame_pool(self) -> tuple[NDArray[np.uint8], ...]: ...
    @property
    def pixel_color_format(self) -> InputPixelFormats: ...
    def grab_frame(self) -> NDArray[np.uint8]: ...

def _get_opencv_ids() -> tuple[CameraInformation, ...]: ...
def _get_harvesters_ids() -> tuple[CameraInformation, ...]: ...
def _get_cti_path() -> Path: ...
@contextmanager
def _suppress_output() -> Generator[None, None, None]: ...
