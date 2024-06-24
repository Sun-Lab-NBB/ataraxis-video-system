import multiprocessing
import os
from multiprocessing import Process, ProcessError, Queue
from typing import Literal, get_args

import cv2
import ffmpeg
import numpy as np
from ataraxis_time import PrecisionTimer
from pynput import keyboard

from .shared_memory_array import SharedMemoryArray


class Camera:
    """A wrapper class for an opencv VideoCapture object.

    Attributes:
        _vid: opencv video capture object.
    """

    def __init__(self) -> None:
        self._vid: cv2.VideoCapture | None = None

    def __del__(self) -> None:
        """Ensures that camera is disconnected upon garbage collection."""
        self.disconnect()

    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
        self._vid = cv2.VideoCapture(0)

    def disconnect(self) -> None:
        """Disconnects from camera."""
        if self._vid:
            self._vid.release()
            self._vid = None

    @property
    def is_connected(self) -> bool:
        """Whether or not the camera is connected."""
        return self._vid is not None

    def grab_frame(self) -> np.ndarray:
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected.

        """
        if self._vid:
            ret, frame = self._vid.read()
            # if not ret:
            #     raise Exception("camera did not yield an image")
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
        Exeption: If save format is specified to an invalid format.
    """

    Save_Format_Type = Literal["png", "jpg", "tif", "mp4"]

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
            raise Exception("Invalid save format.")

        self.save_directory: str = save_directory
        self.camera: Camera = camera
        self._save_format = save_format
        self._running: bool = False
        self._input_process: Process | None = None
        self._save_process: Process | None = None
        self._terminator_array: SharedMemoryArray | None = None
        self._image_queue: Queue | None = None
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
            Exeption: If save format is specified to an invalid format.
        """

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

        if save_format is not None:
            if save_format in get_args(VideoSystem.Save_Format_Type):
                self._save_format = save_format
            else:
                raise Exception("Invalid save format.")

        if save_format in {"tif", "png", "jpg"}:
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
                args=(self.camera, self._image_queue, self._terminator_array, None),
                daemon=True,
            )
            self._save_process = Process(
                target=VideoSystem._save_images_loop,
                args=(self._image_queue, self._terminator_array, self.save_directory, 1),
                daemon=True,
            )

            self._input_process.start()
            self._save_process.start()
            self._running = True

            if listen_for_keypress:
                self._listener = keyboard.Listener(on_press=lambda x: self._on_press(x, self._terminator_array))
                self._listener.start()  # start to listen on a separate thread
        elif self._save_format == "mp4":
            pass

    def stop_image_collection(self) -> None:
        """Stops image collection."""
        if self._running == True:
            self._terminator_array.connect()
            self._terminator_array.write_data(slice(0, 1), np.array([0]))
            self._terminator_array.disconnect()

    # possibly delete this function
    def _stop_image_saving(self) -> None:
        """Stops image saving."""
        if self._running == True:
            self._terminator_array.connect()
            self._terminator_array.write_data(slice(1, 2), np.array([0]))
            self._terminator_array.disconnect()

    def stop(self) -> None:
        """Stops image collection and saving. Ends all processes."""
        if self._running == True:
            self._image_queue = Queue()  # A weak way to empty queue
            self._terminator_array.connect()
            self._terminator_array.write_data(slice(0, 3), np.array([0, 0, 0]))
            self._terminator_array.disconnect()
            self._save_process.join()
            self._input_process.join()
            self._running = False
            if self._listener is not None:
                self._listener.stop()
                self._listener = None

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
        camera: Camera, img_queue: Queue, terminator_array: SharedMemoryArray, fps: float | None = None
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

        unit_conversion = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}
        precision = "ms"

        img_queue.cancel_join_thread()
        camera.connect()
        terminator_array.connect()
        run_timer: PrecisionTimer | None = PrecisionTimer(precision) if fps else None
        frames: int = 0
        first: bool = True
        while terminator_array.read_data(2):
            if terminator_array.read_data(0):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    img_queue.put(camera.grab_frame())
                    frames += 1
                    if fps:
                        run_timer.reset()
        camera.disconnect()
        terminator_array.disconnect()

    @staticmethod
    def _save_frame(img_queue: Queue, save_directory: str, img_id: int) -> bool:
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
            filename = os.path.join(save_directory, "img" + str(img_id) + ".png")
            cv2.imwrite(filename, frame)
            return True
        return False

    @staticmethod
    def _save_images_loop(
        img_queue: Queue, terminator_array: SharedMemoryArray, save_directory: str, fps: float | None = None
    ) -> None:
        """Iteratively grabs images from the camera and adds to the img_queue.

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

        unit_conversion: dict = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}
        precision: str = "ms"

        terminator_array.connect()
        num_imgs_saved: int = 0
        run_timer: PrecisionTimer | None = PrecisionTimer(precision) if fps else None
        frames: int = 0
        img_queue.cancel_join_thread()
        while terminator_array.read_data(2):
            if terminator_array.read_data(1):
                if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                    saved = VideoSystem._save_frame(img_queue, save_directory, num_imgs_saved)
                    if saved:
                        num_imgs_saved += 1
                    frames += 1
                    if fps:
                        run_timer.reset()
        terminator_array.disconnect()

    @staticmethod
    def _save_mp4():
        process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(videoCapture.get(cv2.CAP_PROP_FPS)),
                format="rawvideo",
                pix_fmt="bgr24",
                s="{}x{}".format(
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
            )
            .output("src\\ffmpeg_backend\\webcam.mp4", vcodec="h264", pix_fmt="nv21", **{"b:v": 2000000})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode, terminator_array: SharedMemoryArray) -> bool | None:
        """Changes terminator flags on specific key presses.

        Stops listener if both terminator flags have been set to 0. Stops the listener if video_system has stopped
        running. This method should only be used as a target to a key listener.

        Args:
            key: the key that was pressed.
            terminator_array: A multiprocessing array to hold terminate flags.
        """
        try:
            if key.char == "q":
                self.stop_image_collection()
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
        return
