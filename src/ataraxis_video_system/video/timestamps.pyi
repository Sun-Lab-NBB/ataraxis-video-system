from enum import StrEnum
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext

import numpy as np
from numpy.typing import NDArray as NDArray

_WORKER_THREAD_CEILING: int
_MULTIPROCESSING_CONTEXT: SpawnContext

class ExtractedDataColumns(StrEnum):
    FRAME_TIME = "frame_time_us"

def extract_logged_camera_timestamps(
    log_path: Path, workers: int = -1, *, display_progress: bool = True, executor: ProcessPoolExecutor | None = None
) -> NDArray[np.uint64]: ...
def _process_frame_message_batch(log_path: Path, keys: list[str], onset_us: np.uint64) -> NDArray[np.uint64]: ...
