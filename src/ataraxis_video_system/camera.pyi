from enum import StrEnum
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from harvesters.core import (
    Harvester,
    ImageAcquirer as ImageAcquirer,
)

from .saver import InputPixelFormats as InputPixelFormats

_mono_formats: Incomplete
_color_formats: Incomplete
_all_rgb_formats: Incomplete
_FRAME_POOL_SIZE: int
_MAXIMUM_NON_WORKING_IDS: int

class CameraInterfaces(StrEnum):
    HARVESTERS = "harvesters"
    OPENCV = "opencv"
    MOCK = "mock"

@dataclass()
class CameraInformation:
    camera_index: int
    interface: CameraInterfaces | str
    frame_width: int
    frame_height: int
    acquisition_frame_rate: int
    serial_number: str | None = ...
    model: str | None = ...

def get_opencv_ids() -> tuple[CameraInformation, ...]: ...
def get_harvesters_ids() -> tuple[CameraInformation, ...]: ...
def add_cti_file(cti_path: Path) -> None: ...
def _get_cti_path() -> Path: ...

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
