from _typeshed import Incomplete
from ataraxis_data_structures import SharedMemoryArray
from multiprocessing import Pool as Pool
from numpy.typing import NDArray
from pynput import keyboard
from queue import Queue
from typing import Any, Generic, TypeVar

Precision_Type: Incomplete
unit_conversion: dict[Precision_Type, int]
precision: Precision_Type
T = TypeVar('T')

class MPQueue(Generic[T]):
    """A wrapper for a multiprocessing queue object that makes Queue typed. This is used to appease mypy type checker"""
    _queue: Incomplete
    def __init__(self) -> None: ...
    def put(self, item: T) -> None: ...
    def get(self) -> T: ...
    def get_nowait(self) -> T: ...
    def empty(self) -> bool: ...
    def qsize(self) -> int: ...
    def cancel_join_thread(self) -> None: ...

class Camera:
    """A wrapper class for an opencv VideoCapture object.

    Args:
        camera_id: camera id

    Attributes:
        camera_id: camera id
        specs: dictionary holding the specifications of the camera. This includes fps, frame_width, frame_height
        _vid: opencv video capture object.
    """
    specs: Incomplete
    camera_id: Incomplete
    _vid: Incomplete
    def __init__(self, camera_id: int = 0) -> None: ...
    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
    def disconnect(self) -> None:
        """Disconnects from camera."""
    @property
    def is_connected(self) -> bool:
        """Whether the camera is connected."""
    def grab_frame(self) -> NDArray[Any]:
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected or did not yield an image.

        """

class VideoSystem:
    """Provides a system for efficiently taking, processing, and saving images in real time.

    Args:
        save_directory: location where the system saves images.
        camera: camera for image collection.
        save_format: the format in which to save camera data
        num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
        num_threads: The number of image-saving threads to run per process. Applies only to image saving.

    Attributes:
        save_directory: location where the system saves images.
        camera: camera for image collection.
        _save_format: the format in which to save camera data
        _running: whether or not the video system is running.
        _input_process: multiprocessing process to control the image collection.
        _consumer_processes: list multiprocessing processes to control image saving.
        _terminator_array: multiprocessing array to keep track of process activity and facilitate safe process
            termination.
        _image_queue: multiprocessing queue to hold images before saving.
        _listener: thread to detect key_presses for key-based control of threads.
        _num_consumer_processes: number of processes to run the image consumer loop on. Applies only to image saving.
        _threads_per_process: The number of image-saving threads to run per process. Applies only to image saving.

    Raises:
        ProcessError: If the function is created not within the '__main__' scope
        ValueError: If the save format is specified to an invalid format.
        ProcessError: If the computer does not have enough cpu cores.
    """
    Save_Format_Type: Incomplete
    save_directory: Incomplete
    camera: Incomplete
    _save_format: Incomplete
    _num_consumer_processes: Incomplete
    _threads_per_process: Incomplete
    _running: bool
    _input_process: Incomplete
    _consumer_processes: Incomplete
    _terminator_array: Incomplete
    _image_queue: Incomplete
    _listener: Incomplete
    def __init__(self, save_directory: str, camera: Camera, save_format: Save_Format_Type = 'png', num_processes: int = 3, num_threads: int = 4) -> None: ...
    @staticmethod
    def _empty_function() -> None:
        """A function that passes to be used as target to a process"""
    def start(self, listen_for_keypress: bool = False, terminator_array_name: str = 'terminator_array', save_format: Save_Format_Type | None = None, num_processes: int | None = None, num_threads: int | None = None) -> None:
        """Starts the video system.

        Args:
            listen_for_keypress: If true, the video system will stop the image collection when the 'q' key is pressed
                and stop image saving when the 'w' key is pressed.
            terminator_array_name: The name of the shared_memory_array to be created. When running multiple
                video_systems concurrently, each terminator_array should have a unique name.
            save_format: the format in which to save camera data
            num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
            num_threads: The number of image-saving threads to run per process. Applies only to image saving.

        Raises:
            ProcessError: If the function is created not within the '__main__' scope.
            ValueError: If the save format is specified to an invalid format.
            ProcessError: If the computer does not have enough cpu cores.
        """
    def stop_image_production(self) -> None:
        """Stops image collection."""
    def _stop_image_saving(self) -> None:
        """Stops image saving."""
    def stop(self) -> None:
        """Stops image collection and saving. Ends all processes."""
    @staticmethod
    def _delete_files_in_directory(path: str) -> None:
        """Generic method to delete all files in a specific folder.

        Args:
            path: Location of the folder.
        Raises:
            FileNotFoundError when the path does not exist.
        """
    def delete_images(self) -> None:
        """Clears the save directory of all images.

        Raises:
            FileNotFoundError when self.save_directory does not exist.
        """
    @staticmethod
    def _produce_images_loop(camera: Camera, img_queue: MPQueue[Any], terminator_array: SharedMemoryArray, fps: float | None = None) -> None:
        """Iteratively grabs images from the camera and adds to the img_queue.

        This function loops while the third element in terminator_array (index 2) is nonzero. It grabs frames as long as
        the first element in terminator_array (index 0) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a thread and will create an infinite loop if run on its
        own.

        Args:
            camera: a Camera object to take collect images.
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 0 is zero
                and completes when index 2 is zero.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
    @staticmethod
    def _save_frame(img_queue: MPQueue[Any], save_directory: str) -> bool:
        """Saves one frame as a png image.

        Does nothing if there are no images in the queue.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            save_directory: relative path to location where image is to be saved.
        Returns:
            True if an image was saved, otherwise False.
        """
    @staticmethod
    def _frame_saver(q: Queue[Any], save_directory: str) -> None: ...
    @staticmethod
    def _save_images_loop(img_queue: MPQueue[Any], terminator_array: SharedMemoryArray, save_directory: str, num_threads: int, fps: float | None = None) -> None:
        """Iteratively grabs images from the img_queue and saves them as png files.

        This function loops while the third element in terminator_array (index 2) is nonzero. It saves images as long as
        the second element in terminator_array (index 1) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a thread and will create an infinite loop if run on its
        own.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 1 is zero
                and completes when index 2 is zero.
            save_directory: relative path to location where images are to be saved.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
    @staticmethod
    def _save_video_loop(img_queue: MPQueue[Any], terminator_array: SharedMemoryArray, save_directory: str, camera_specs: dict[str, Any], fps: float | None = None) -> None:
        """Iteratively grabs images from the img_queue and adds them to an mp4 file.

        This creates runs the ffmpeg image saving process on a separate thread. It iteratively grabs images from the
        queue on the main thread.

        This function loops while the third element in terminator_array (index 2) is nonzero. It saves images as long as
        the second element in terminator_array (index 1) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a process and will create an infinite loop if run on its
        own.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 1 is zero
                and completes when index 2 is zero.
            save_directory: relative path to location where images are to be saved.
            camera_specs: a dictionary containing specifications of the camera. Specifically, the dictionary must
                contain the camera's frames per second, denoted 'fps', and the camera frame size denoted by
                'frame_width' and 'frame_height'.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None, terminator_array: SharedMemoryArray) -> bool | None:
        """Changes terminator flags on specific key presses.

        Stops listener if both terminator flags have been set to 0. Stops the listener if video_system has stopped
        running. This method should only be used as a target to a key listener.

        Args:
            key: the key that was pressed.
            terminator_array: A multiprocessing array to hold terminate flags.
        """
