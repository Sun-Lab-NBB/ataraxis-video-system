import numpy as np
from .shared_memory_array import SharedMemoryArray as SharedMemoryArray
from _typeshed import Incomplete

class Camera:
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    def grab_frame(self) -> np.ndarray: ...

class VideoSystem:
    save_directory: Incomplete
    camera: Incomplete
    def __init__(self, save_directory: str, camera: Camera) -> None: ...
    def start(self, listen_for_keypress: bool = ..., terminator_array_name: str = ..., save_type: str = ...) -> None: ...
    def stop_image_collection(self) -> None: ...
    def stop(self) -> None: ...
    def delete_images(self) -> None: ...
