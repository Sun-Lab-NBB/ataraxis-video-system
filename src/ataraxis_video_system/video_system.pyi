from queue import Queue
from pathlib import Path
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager

import numpy as np
from numpy.typing import NDArray as NDArray
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .saver import (
    VideoSaver as VideoSaver,
    VideoEncoders as VideoEncoders,
    OutputPixelFormats as OutputPixelFormats,
    EncoderSpeedPresets as EncoderSpeedPresets,
    check_gpu_availability as check_gpu_availability,
    check_ffmpeg_availability as check_ffmpeg_availability,
)
from .camera import (
    MockCamera as MockCamera,
    OpenCVCamera as OpenCVCamera,
    CameraInterfaces as CameraInterfaces,
    HarvestersCamera as HarvestersCamera,
)

_MAXIMUM_QUANTIZATION_VALUE: int
_PROCESS_INITIALIZATION_TIME: int
_PROCESS_SHUTDOWN_TIME: int
_FRAME_TIMESTAMP_LOG_MESSAGE_SIZE: int
_MINIMUM_LOG_SIZE_FOR_PARALLELIZATION: int

class VideoSystem:
    _started: bool
    _mp_manager: SyncManager
    _system_id: np.uint8
    _output_file: Path | None
    _camera: OpenCVCamera | HarvestersCamera | MockCamera
    _display_frame_rate: int
    _saver: VideoSaver | None
    _logger_queue: MPQueue
    _saver_queue: MPQueue
    _terminator_array: SharedMemoryArray | None
    _producer_process: Process | None
    _consumer_process: Process | None
    _watchdog_thread: Thread | None
    def __init__(
        self,
        system_id: np.uint8,
        data_logger: DataLogger,
        output_directory: Path | None,
        camera_interface: CameraInterfaces | str = ...,
        camera_index: int = 0,
        display_frame_rate: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame_rate: int | None = None,
        gpu: int = -1,
        video_encoder: VideoEncoders | str = ...,
        encoder_speed_preset: EncoderSpeedPresets | int = ...,
        output_pixel_format: OutputPixelFormats | str = ...,
        quantization_parameter: int = 15,
        *,
        color: bool | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @staticmethod
    def _frame_display_loop(display_queue: Queue, system_id: np.uint8) -> None: ...
    @staticmethod
    def _frame_production_loop(
        system_id: np.uint8,
        camera: OpenCVCamera | HarvestersCamera | MockCamera,
        display_frame_rate: int,
        saver_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
    ) -> None: ...
    @staticmethod
    def _frame_saving_loop(
        system_id: np.uint8,
        saver: VideoSaver,
        saver_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
    ) -> None: ...
    def _watchdog(self) -> None: ...
    def start_frame_saving(self) -> None: ...
    def stop_frame_saving(self) -> None: ...
    @property
    def video_file_path(self) -> Path | None: ...
    @property
    def started(self) -> bool: ...
    @property
    def system_id(self) -> np.uint8: ...

def _process_frame_message_batch(log_path: Path, file_names: list[str], onset_us: np.uint64) -> list[np.uint64]: ...
def extract_logged_camera_timestamps(log_path: Path, n_workers: int = -1) -> tuple[np.uint64, ...]: ...
