import multiprocessing
import os
from multiprocessing import Process, ProcessError
from multiprocessing import Queue as UntypedQueue
from typing import Any, Dict, Generic, Literal, TypeVar, cast, get_args

import cv2
import ffmpeg
import numpy as np
import tifffile as tff
from ataraxis_time import PrecisionTimer
from PIL import Image
from pynput import keyboard

from .shared_memory_array import SharedMemoryArray

# Used in many functions to convert between units
Precision_Type = Literal["ns", "us", "ms", "s"]
unit_conversion: Dict[Precision_Type, int] = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}
precision: Precision_Type = "ms"

T = TypeVar("T")


class Queue(Generic[T]):
    """A wrapper for a multiprocessing queue object that makes Queue typed. This is used in order to appease mypy type
    checker"""

    def __init__(self) -> None:
        self._queue = UntypedQueue()  # type: ignore

    def put(self, item: T) -> None:
        self._queue.put(item)

    def get(self) -> T:
        return cast(T, self._queue.get())

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def cancel_join_thread(self) -> None:
        return self._queue.cancel_join_thread()


class Camera:
    """A wrapper class for an opencv VideoCapture object.

    Args:
        id: camera id

    Attributes:
        id: camera id
        specs: dictionary holding the specifications of the camera. This includes fps, frame_width, frame_height
        _vid: opencv video capture object.
    """

    def __init__(self, id: int = 0) -> None:
        self.specs: Dict[str, Any] = {}
        self.id = id
        self._vid: cv2.VideoCapture | None = None
        self.connect()
        if self._vid is not None:
            self.specs["fps"] = self._vid.get(cv2.CAP_PROP_FPS)
            self.specs["frame_width"] = self._vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.specs["frame_height"] = self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.disconnect()

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()

    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
        self._vid = cv2.VideoCapture(self.id)
        # try:
        #     self.grab_frame()
        # except Exception:
        #     raise Exception("could not connect to camera")

    def disconnect(self) -> None:
        """Disconnects from camera."""
        if self._vid:
            self._vid.release()
            self._vid = None

    @property
    def is_connected(self) -> bool:
        """Whether or not the camera is connected."""
        return self._vid is not None

    def grab_frame(self) -> np.typing.NDArray[Any]:
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected or did not yield an image.

        """
        if self._vid:
            ret, frame = self._vid.read()
            if not ret:
                raise Exception("camera did not yield an image")
            return frame
        else:
            raise Exception("camera not connected")


class VideoSystem:
    """Provides a system for efficiently taking, processing, and saving images in real time.

    Args:
        save_directory: location where system saves images.
        camera: camera for image collection.
        save_format: the format in which to save camera data

    Attributes:
        save_directory: location where system saves images.
        camera: camera for image collection.
        _save_format: the format in which to save camera data
        _running: whether or not the video system is running.
        _input_process: multiprocessing process to control image collection.
        _save_process: multiprocessing process to control image saving.
        _terminator_array: multiprocessing array to keep track of process activity and facilitate safe process
            termination.
        _img_queue: multiprocessing queue to hold images before saving.
        _listener: thread to detect key_presses for key based control of threads.

    Raises:
        ProcessError: If the function is created not within the '__main__' scope
        ValueError: If save format is specified to an invalid format.
    """

    Save_Format_Type = Literal["png", "jpg", "tiff", "mp4"]

    def __init__(self, save_directory: str, camera: Camera, save_format: Save_Format_Type = "png"):
        # # Check to see if class was run from within __name__ = "__main__" or equivalent scope
        in_unprotected_scope: bool = False
        try:
            p = multiprocessing.Process(target=VideoSystem._empty_function)
            p.start()
            p.join()
        except RuntimeError:
            in_unprotected_scope = True

        if in_unprotected_scope:
            raise ProcessError("Instantiation method outside of '__main__' scope")

        if save_format not in get_args(VideoSystem.Save_Format_Type):
            raise ValueError("Invalid save format.")

        self.save_directory: str = save_directory
        self.camera: Camera = camera
        self._save_format = save_format
        self._running: bool = False

        self._input_process: Process | None = None
        self._save_process: Process | None = None
        self._terminator_array: SharedMemoryArray | None = None
        self._image_queue: Queue[Any] | None = None
        

        self._listener: keyboard.Listener | None = None

    @staticmethod
    def _empty_function() -> None:
        """A function that passes to be used as target to a process"""
        pass

    def start(
        self,
        listen_for_keypress: bool = False,
        terminator_array_name: str = "terminator_array",
        save_format: Save_Format_Type | None = None,
    ) -> None:
        """Starts the video system.

        Args:
            listen_for_keypress: If true, the video system will stop image collection when the 'q' key is pressed and
                stop image saving when the 'w' key is pressed.
            terminator_array_name: The name of the shared_memory_array to be created. When running multiple
                video_systems concurrently, each terminator_array should have a unique name.
            save_format: the format in which to save camera data

        Raises:
            ProcessError: If the function is created not within the '__main__' scope.
            ValueError: If save format is specified to an invalid format.
        """

        # Check to see if class was run from within __name__ = "__main__" or equivalent scope
        in_unprotected_scope: bool = False
        try:
            p = multiprocessing.Process(target=VideoSystem._empty_function)
            p.start()
            p.join()
        except RuntimeError:
            in_unprotected_scope = True

        if in_unprotected_scope:
            raise ProcessError("Instantiation method outside of '__main__' scope")

        if save_format is not None:
            if save_format in get_args(VideoSystem.Save_Format_Type):
                self._save_format = save_format
            else:
                raise ValueError("Invalid save format.")

        self.delete_images()

        self._image_queue = Queue()

        prototype = np.array(
            [1, 1, 1], dtype=np.int32
        )  # First entry represents whether input stream is active, second entry represents whether output stream is active

        self._terminator_array = SharedMemoryArray.create_array(
            terminator_array_name,
            prototype,
        )

        self._input_process = Process(
            target=VideoSystem._input_stream,
            args=(self.camera, self._image_queue, self._terminator_array),
            daemon=True,
        )

        if self._save_format in {"tiff", "png", "jpg"}:
            self._save_process = Process(
                target=VideoSystem._save_images_loop,
                args=(self._image_queue, self._terminator_array, self.save_directory),
                daemon=True,
            )
        else:  # self._save_format == "mp4"
            self._save_process = Process(
                target=VideoSystem._save_video_loop,
                args=(self._image_queue, self._terminator_array, self.save_directory, self.camera.specs),
                daemon=True,
            )
        self._input_process.start()
        self._save_process.start()

        if listen_for_keypress:
            self._listener = keyboard.Listener(on_press=lambda x: self._on_press(x, self._terminator_array))
            self._listener.start()  # start to listen on a separate thread

        self._running = True

    def stop_image_production(self) -> None:
        """Stops image collection."""
        if self._running == True:
            if self._terminator_array is not None:
                self._terminator_array.connect()
                self._terminator_array.write_data(slice(0, 1), np.array([0], dtype=np.int32))
                self._terminator_array.disconnect()
            else:  # This error should never occur
                error_message = (
                    "Failure to start the stop image production process because _terminator_array is not initialized."
                )
                raise TypeError(error_message)

    # possibly delete this function
    def _stop_image_saving(self) -> None:
        """Stops image saving."""
        if self._running == True:
            if self._terminator_array is not None:
                self._terminator_array.connect()
                self._terminator_array.write_data(slice(1, 2), np.array([0], dtype=np.int32))
                self._terminator_array.disconnect()
            else:  # This error should never occur
                error_message = (
                    "Failure to start the stop image saving process because _terminator_array is not initialized."
                )
                raise TypeError(error_message)

    def stop(self) -> None:
        """Stops image collection and saving. Ends all processes."""
        if self._running == True:
            self._image_queue = Queue()  # A weak way to empty queue
            if self._terminator_array is not None:
                self._terminator_array.connect()
                self._terminator_array.write_data(slice(0, 3), np.array([0, 0, 0], dtype=np.int32))
                self._terminator_array.disconnect()
            else:  # This error should never occur
                error_message = "Failure to start the stop video system  because _terminator_array is not initialized."
                raise TypeError(error_message)

            if self._save_process is not None:
                self._save_process.join()
            else:  # This error should never occur
                error_message = "Failure to start the stop video system  because _save_process is not initialized."
                raise TypeError(error_message)

            if self._input_process is not None:
                self._input_process.join()
            else:  # This error should never occur
                error_message = "Failure to start the stop video system  because _input_process is not initialized."
                raise TypeError(error_message)

            if self._listener is not None:
                self._listener.stop()
                self._listener = None

            self._running = False

    # def __del__(self):
    #     """Ensures that system is stopped upon garbage collection. """
    #     self.stop()

    @staticmethod
    def _delete_files_in_directory(path: str) -> None:
        """Generic method to delete all files in a specific folder.

        Args:
            path: Location of the folder.
        Raises:
            FileNotFoundError when path does not exist.
        """
        with os.scandir(path) as entrys:
            for entry in entrys:
                if entry.is_file():
                    os.unlink(entry.path)

    def delete_images(self) -> None:
        """Clears the save directory of all images.

        Raises:
            FileNotFoundError when self.save_directory does not exist.
        """
        VideoSystem._delete_files_in_directory(self.save_directory)

    @staticmethod
    def _input_stream(
        camera: Camera, img_queue: Queue[Any], terminator_array: SharedMemoryArray, fps: float | None = None
    ) -> None:
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
        img_queue.cancel_join_thread()
        camera.connect()
        terminator_array.connect()
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        first: bool = True
        while terminator_array.read_data(2):
            if terminator_array.read_data(0):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    img_queue.put(camera.grab_frame())
                    if fps:
                        run_timer.reset()
        camera.disconnect()
        terminator_array.disconnect()

    @staticmethod
    def _save_frame(img_queue: Queue[Any], save_directory: str, img_id: int) -> bool:
        """Saves one frame as a png image.

        Does nothing if there are no images in the queue.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            save_directory: relative path to location  where image is to be saved.
            img_id: id tag for image.
        Returns:
            True if an image was saved, otherwise False.
        """
        if not img_queue.empty():  # empty is unreliable way to check if Queue is empty
            frame = img_queue.get()
            # print(frame.shape)
            filename = os.path.join(save_directory, "img" + str(img_id) + ".png")

            # print(np.min(frame), np.max(frame))
            # temp = frame[:, :, 0]
            # frame[:,:,0] = frame[:, :, 1]
            # frame[:, :, 1]=temp

            # tff.imwrite(
            #         filename,
            #         data=frame,
            #         photometric='rgb',
            #     )
            # frame = tff.imread(
            #     filename
            # )
            # filename = filename[:-3] + 'png'
            cv2.imwrite(filename, frame)
            # print('saved image')
            return True
        return False

    @staticmethod
    def _save_images_loop(
        img_queue: Queue[Any], terminator_array: SharedMemoryArray, save_directory: str, fps: float | None = None
    ) -> None:
        """Iteratively grabs images from the img_queue and saves them as png files.

        This function loops while the third element in terminator_array (index 2) is nonzero. It saves images as long as
        the second element in terminator_array (index 1) is nonzero. This function can be run at a specific fps or as
        fast as possible. This function is meant to be run as a thread and will create an infinite loop if run on its
        own.

        Args:
            img_queue: A multiprocessing queue to hold images before saving.
            terminator_array: A multiprocessing array to hold terminate flags, the function idles when index 1 is zero
                and completes when index 2 is zero.
            save_directory: relative path to location  where images are to be saved.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """

        terminator_array.connect()
        num_imgs_saved: int = 0
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        img_queue.cancel_join_thread()
        while terminator_array.read_data(2):
            if terminator_array.read_data(1):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    saved = VideoSystem._save_frame(img_queue, save_directory, num_imgs_saved)
                    if saved:
                        num_imgs_saved += 1
                    if fps:
                        run_timer.reset()
        terminator_array.disconnect()

    @staticmethod
    def _save_video_loop(
        img_queue: Queue[Any],
        terminator_array: SharedMemoryArray,
        save_directory: str,
        camera_specs: Dict[str, Any],
        fps: float | None = None,
    ) -> None:
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
            save_directory: relative path to location  where images are to be saved.
            camera_specs: a dictionary containing specifications of the camera. Specifically the dictionary must contain
                the camera's frames per second, denoted 'fps', and the camera frame size denoted by 'frame_width' and
                'frame_height'.
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible.
        """
        filename = os.path.join(save_directory, "video.mp4")

        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(camera_specs["fps"]),
                format="rawvideo",
                pix_fmt="bgr24",
                s="{}x{}".format(int(camera_specs["frame_width"]), int(camera_specs["frame_height"])),
            )
            .output(filename, vcodec="h264", pix_fmt="nv21", **{"b:v": 2000000})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        terminator_array.connect()
        num_imgs_saved: int = 0
        run_timer: PrecisionTimer = PrecisionTimer(precision)
        img_queue.cancel_join_thread()
        while terminator_array.read_data(2):
            if terminator_array.read_data(1):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    if not img_queue.empty():
                        image = img_queue.get()
                        ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())
                    if fps:
                        run_timer.reset()
        terminator_array.disconnect()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode, terminator_array: SharedMemoryArray) -> bool | None:
        """Changes terminator flags on specific key presses.

        Stops listener if both terminator flags have been set to 0. Stops the listener if video_system has stopped
        running. This method should only be used as a target to a key listener.

        Args:
            key: the key that was pressed.
            terminator_array: A multiprocessing array to hold terminate flags.
        """
        try:
            print(key.char)
            if key.char == "q":
                self.stop_image_production()
                print("Stopped taking images")
            elif key.char == "w":
                self._stop_image_saving()
                print("Stopped saving images")
        except AttributeError:
            pass
        if self._running:
            terminator_array.connect()
            if not terminator_array.read_data(0) and not terminator_array.read_data(1):
                terminator_array.disconnect()
                self.stop()
                return False  # stop listener
        else:
            return False
        return None
