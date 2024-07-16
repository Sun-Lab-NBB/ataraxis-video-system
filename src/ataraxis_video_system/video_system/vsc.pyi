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
        save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg' is
            a lossy format
        tiff_compression_level: the amount of compression to apply for tiff image saving. Range is [0, 9] inclusive. 0 gives fastest saving but
            most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
            relevant when save_format is specified as 'tiff.'
        jpeg_quality: the amount of compression to apply for jpeg image saving. Range is [0, 100] inclusive. 0 gives highest level of compression but
            the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
            compression value is only relevant when save_format is specified as 'jpg.'
        num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
        num_threads: The number of image-saving threads to run per process. Applies only to image saving.

    Attributes:
        save_directory: location where the system saves images.
        camera: camera for image collection.
        _save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg' is
            a lossy format
        _tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
            most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
            relevant when save_format is specified as 'tiff.'
        _jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
            the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
            compression value is only relevant when save_format is specified as 'jpg.'
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
        ValueError: If a specified tiff_compression_level is not within [0, 9] inclusive.
        ValueError: If a specified jpeg_quality is not within [0, 100] inclusive.
        ProcessError: If the computer does not have enough cpu cores.
    """
    img_name: str
    vid_name: str
    Save_Format_Type: Incomplete
    save_directory: Incomplete
    camera: Incomplete
    _save_format: Incomplete
    _jpeg_quality: Incomplete
    _tiff_compression_level: Incomplete
    _num_consumer_processes: Incomplete
    _threads_per_process: Incomplete
    _running: bool
    _input_process: Incomplete
    _consumer_processes: Incomplete
    _terminator_array: Incomplete
    _image_queue: Incomplete
    _listener: Incomplete
    def __init__(self, save_directory: str, camera: Camera, save_format: Save_Format_Type = 'png', tiff_compression_level: int = 6, jpeg_quality: int = 95, num_processes: int = 3, num_threads: int = 4) -> None: ...
    @staticmethod
    def _empty_function() -> None:
        """A function that passes to be used as target to a process"""
    def start(self, listen_for_keypress: bool = False, terminator_array_name: str = 'terminator_array', save_format: Save_Format_Type | None = None, tiff_compression_level: int | None = None, jpeg_quality: int | None = None, num_processes: int | None = None, num_threads: int | None = None) -> None:
        """Starts the video system.

        Args:
            listen_for_keypress: If true, the video system will stop the image collection when the 'q' key is pressed
                and stop image saving when the 'w' key is pressed.
            terminator_array_name: The name of the shared_memory_array to be created. When running multiple
                video_systems concurrently, each terminator_array should have a unique name.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
            num_processes: number of processes to run the image consumer loop on. Applies only to image saving.
            num_threads: The number of image-saving threads to run per process. Applies only to image saving.

        Raises:
            ProcessError: If the function is created not within the '__main__' scope.
            ValueError: If the save format is specified to an invalid format.
            ValueError: If a specified tiff_compression_level is not within [0, 9] inclusive.
            ValueError: If a specified jpeg_quality is not within [0, 100] inclusive.
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
    def imwrite(filename: str, data: NDArray[Any], tiff_compression_level: int = 6, jpeg_quality: int = 95) -> None:
        """Saves an image to a specified file.

        Args:
            filename: path to image file to be created.
            data: pixel data of image.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. Range is [0, 9] inclusive. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: The amount of compression to apply for jpeg image saving. Range is [0, 100] inclusive. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
        """
    @staticmethod
    def _frame_saver(q: Queue[Any], save_directory: str, save_format: str, tiff_compression_level: int, jpeg_quality: int) -> None:
        """A method that iteratively gets an image from a queue and saves it to save_directory. This method loops until
        it pulls an image off the queue whose id is 0. This loop is not meant to be called directly, rather it is meant
        to be the target of a separate thread.

        Args:
            q: A queue to hold images before saving.
            save_directory: relative path to location where image is to be saved.
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
        """
    @staticmethod
    def _save_images_loop(img_queue: MPQueue[Any], terminator_array: SharedMemoryArray, save_directory: str, save_format: str, tiff_compression_level: int, jpeg_quality: int, num_threads: int, fps: float | None = None) -> None:
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
            save_format: the format in which to save camera data. Note 'tiff' and 'png' formats are lossless while 'jpg'
                is a lossy format
            tiff_compression_level: the amount of compression to apply for tiff image saving. 0 gives fastest saving but
                most memory used. 9 gives slowest saving but least amount of memory used. This compression value is only
                relevant when save_format is specified as 'tiff.'
            jpeg_quality: the amount of compression to apply for jpeg image saving. 0 gives highest level of compression but
                the most loss of image detail. 100 gives the lowest level of compression but no loss of image detail. This
                compression value is only relevant when save_format is specified as 'jpg.'
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
    def save_imgs_as_vid(self) -> None:
        """Converts a set of id labeled images into an mp4 video file.

        This is a wrapper class for the static method imgs_to_vid. It calls imgs_to_vid with arguments fitting a
        specific object instance.

        Raises:
            Exception: If there are no images of the specified type in the specified directory.
        """
    @staticmethod
    def imgs_to_vid(fps: int, img_directory: str = 'imgs', img_filetype: str = 'png', vid_directory: str | None = None) -> None:
        '''Converts a set of id labeled images into an mp4 video file.

        Args:
            fps: The framerate of the video to be created.
            img_directory: The directory where the images are saved.
            img_filetype: The type of image to be read. Supported types are "tiff", "png", and "jpg"
            vid_directory: The location to save the video. Defaults to the directory that the images are saved in.

        Raises:
            Exception: If there are no images of the specified type in the specified directory.
        '''
    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None, terminator_array: SharedMemoryArray) -> bool | None:
        """Changes terminator flags on specific key presses.

        Stops listener if both terminator flags have been set to 0. Stops the listener if video_system has stopped
        running. This method should only be used as a target to a key listener.

        Args:
            key: the key that was pressed.
            terminator_array: A multiprocessing array to hold terminate flags.
        """
