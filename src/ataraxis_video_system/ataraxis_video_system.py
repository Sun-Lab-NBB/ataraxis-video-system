import cv2
import os
from high_precision_timer.precision_timer import PrecisionTimer
from multiprocessing import Process, Queue, ProcessError
import multiprocessing
import numpy as np
from pynput import keyboard
from shared_memory_array import SharedMemoryArray
from camera import Camera
import queue  #

unit_conversion = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}
precision = "ms"


class VideoSystem:
    """Provides a system for efficiently taking, processing, and saving images in real time.

    Notes:

    Args:
        save_directory: location where system saves images

    Attributes:
        save_directory: location where system saves images

    Raises:

    """

    def __init__(self, save_directory: str, camera: Camera):
        self.save_directory = save_directory
        self.camera = camera
        self.running = False
        self._input_process = None
        self._save_process = None
        self._terminator_array = None
        self._img_queue = None
        if multiprocessing.current_process().daemon:
            raise ProcessError("Instantiation method outside of main scope")

    def start(self, listen_for_keypress = False):
        """Starts the video system until terminated by keypress.

        Pressing q ends image collection, Pressing w ends image saving.
        """
        self.delete_images()

        self._img_queue = Queue()

        prototype = np.array([1, 1, 1], dtype=np.int32)  # First entry represents whether input stream is active, second entry represents whether output stream is active
        self._terminator_array = SharedMemoryArray.create_array("terminator_array", prototype,)

        self._input_process = Process(
            target=VideoSystem._input_stream,
            args=(self.camera, self._img_queue, self._terminator_array, None)
        )
        self._save_process = Process(
            target=VideoSystem._save_images_loop,
            args=(self._img_queue, self._terminator_array, self.save_directory, 1)
        )

        self._input_process.start()
        self._save_process.start()
        self.running = True

        if listen_for_keypress:
            listener = keyboard.Listener(on_press=lambda x: self._on_press(x, self._terminator_array))
            listener.start()  # start to listen on a separate thread
            listener.join()

    def stop_image_collection(self):
        if self.running == True:
            self._terminator_array.connect()
            self._terminator_array.write_data(0, np.array([0]))
            self._terminator_array.disconnect()


    # possibly delete this function
    def _stop_image_saving(self):
        if self.running == True:
            self._terminator_array.connect()
            self._terminator_array.write_data(1, np.array([0]))
            self._terminator_array.disconnect()    


    def stop(self):
        if self.running == True:
            self._img_queue = Queue() # A weak way to empty queue
            self._terminator_array.connect()
            self._terminator_array.write_data(slice(0, 3), np.array([0, 0, 0]))
            self._terminator_array.disconnect()
            self._save_process.join()
            self._input_process.join()
            self.running = False
        

    def __del__(self):
        """ """
        # self.stop()

    @staticmethod
    def _delete_files_in_directory(path):
        """Generic method to delete all files in a specific folder.

        Args:
            path: Location of the folder
        Raises:
            FileNotFoundError when path does not exist
        """
        with os.scandir(path) as entrys:
            for entry in entrys:
                if entry.is_file():
                    os.unlink(entry.path)

    def delete_images(self):
        """Clears the save directory of all images

        Raises:
            FileNotFoundError when self.save_directory does not exist
        """
        VideoSystem._delete_files_in_directory(self.save_directory)

    @staticmethod
    def _input_stream(camera: Camera, img_queue: Queue, terminator_array: SharedMemoryArray, fps: int = None):
        """Iteratively grabs images from the camera and adds to the img_queue.

        This function loops while the third element in terminator_array (index 2) is nonzero. It grabs frames as long as the first element in terminator_array (index 0) is nonzero. This function can be run at a specific fps or as fast as possible. This function is meant to be run as a thread will create an infinite loop if run on its own.

        Args:
            camera: a Camera object which
            img_queue: A multiprocessing queue to hold images before saving
            terminator_array: A multiprocessing array to hold terminate flags, the first element (index 0) is relevant for loop termination
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible
        """
        img_queue.cancel_join_thread()
        camera.connect()
        terminator_array.connect()
        run_timer = PrecisionTimer(precision) if fps else None
        frames = 0
        first = True
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
    def _save_frame(img_queue : Queue, save_directory : str, img_id: int) -> bool:
        """Saves one frame as a png image.

        Does nothing if there are no images to save.

        Args:
            img_queue: A multiprocessing queue to hold images before saving
            save_directory: relative path to location  where image is to be saved
            img_id: id tag for image
        Returns:
            True if an image was saved, otherwise False
        """
        if not img_queue.empty(): # empty is unreliable way to check if Queue is empty
            frame = img_queue.get()
            filename = os.path.join(save_directory, "img" + str(img_id) + ".png")
            cv2.imwrite(filename, frame)
            return True
        return False

    @staticmethod
    def _save_images_loop(img_queue, terminator_array, save_directory, fps=None):
        """Iteratively grabs images from the camera and adds to the img_queue.

        This function loops while the second element in terminator_array is nonzero. This function can be run at a specific fps or as fast as possible.

        Args:
            img_queue: A multiprocessing queue to hold images before saving
            terminator_array: A multiprocessing array to hold terminate flags, the second element (index 1) is relevant for loop termination
            save_directory: relative path to location  where images are to be saved
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible
        """
        terminator_array.connect()
        num_imgs_saved = 0
        run_timer = PrecisionTimer(precision) if fps else None
        frames = 0
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

    def _on_press(self, key, terminator_array) -> bool:
        """Changes terminator flags on specific key presses

        Stops listener when both terminator flags have been set to 0. Stops the listener if video_system has stopped 
        running.

        Args:
            key: the key that was pressed
            terminator_array: A multiprocessing array to hold terminate flags
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
        if self.running:
            terminator_array.connect()
            if not terminator_array.read_data(0) and not terminator_array.read_data(1):
                terminator_array.disconnect()
                self.stop()
                return False  # stop listener
        else:
            return False
        
