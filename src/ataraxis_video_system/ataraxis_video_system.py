import cv2
import os
from high_precision_timer.precision_timer import PrecisionTimer
from multiprocessing import Process, Queue, ProcessError
import multiprocessing
import numpy as np
from pynput import keyboard
from shared_memory_array import SharedMemoryArray
from camera import Camera

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
        self.input_process = None
        self.save_process = None
        if multiprocessing.current_process().daemon:
            raise ProcessError('Instantiation method outside of main scope')

    def start(self):
        """Starts the video system until terminated by keypress.

        Pressing q ends image collection, Pressing w ends image saving.
        """
        print('running')
        self.delete_images()

        data_queue = Queue()

        prototype = np.array([1, 1], dtype=np.int32) # First entry represents whether input stream is active, second entry represents whether output stream is active
        terminator_array = SharedMemoryArray.create_array("terminator_array", prototype)

        listener = keyboard.Listener(on_press=lambda x: self.on_press(x, terminator_array))
        self.input_process = Process(target=VideoSystem._input_stream, args=(self.camera, data_queue, terminator_array, 2,), daemon=True,)
        self.save_process = Process(
            target=VideoSystem._save_images_loop,
            args=(
                data_queue,
                terminator_array,
                self.save_directory,
                1,
            ),
            daemon=True,
        )

        listener.start()  # start to listen on a separate thread
        self.input_process.start()
        self.save_process.start()

        self.input_process.join()
        print("input stream joined")
        self.input_process.join()
        print("save images joined")

        listener.join()
        print("listener joined")

    def stop():
        pass

    def __del__(self):
        """ """
        pass

    @staticmethod
    def _delete_files_in_directory(path):
        """Generic method to delete all files in a specific folder.

        Args:
            path: Location of the folder
        """
        try:
            with os.scandir(path) as entrys:
                for entry in entrys:
                    if entry.is_file():
                        os.unlink(entry.path)
        except OSError:
            print("Error occurred while deleting files.")

    def delete_images(self):
        """Clears the save directory of all images"""
        VideoSystem._delete_files_in_directory(self.save_directory)

    @staticmethod
    def _input_stream(
        camera: Camera,
        img_queue: Queue,
        terminator_array: SharedMemoryArray,
        fps: int = None,
    ):
        """Iteratively grabs images from the camera and adds to the img_queue.

        This function loops while the first element in terminator_array is nonzero. This function can be run at a
        specific fps or as fast as possible

        Args:
            camera: a Camera object which
            img_queue: A multiprocessing queue to hold images before saving
            terminator_array: A multiprocessing array to hold terminate flags, the first element (index 0) is relevant for loop termination
            fps: frames per second of loop. If fps is None, the loop will run as fast as possible
        """
        camera.connect()
        terminator_array.connect()
        run_timer = PrecisionTimer(precision) if fps else None
        frames = 0
        while terminator_array.read_data(0):
            if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                img_queue.put(camera.grab_frame())
                frames += 1
                run_timer.reset()
        camera.disconnect()
        terminator_array.disconnect()
        print("input stream finished")

    @staticmethod
    def _save_frame(img_queue, save_directory, img_id: int):
        """Saves one frame as a png image.

        Does nothing if there are no images to save.

        Args:
            img_queue: A multiprocessing queue to hold images before saving
            save_directory: relative path to location  where image is to be saved
            img_id: id tag for image
        """
        if not img_queue.empty():
            frame = img_queue.get()
            filename = save_directory + "\\img" + str(img_id) + ".png"
            cv2.imwrite(filename, frame)
            return True
        return False

    @staticmethod
    def _empty_queue(q):
        """Generic method to empty a multiprocessing queue.

        Args:
            q: the queue to be emptied
        """
        while not q.empty():
            q.get()

    @staticmethod
    def _save_images_loop(data_queue, terminator_array, save_directory, fps=None):
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
        while terminator_array.read_data(1):
            if not fps or run_timer.elapsed / unit_conversion[precision] >= 1 / fps:
                saved = VideoSystem._save_frame(data_queue, save_directory, num_imgs_saved)
                if saved:
                    num_imgs_saved += 1
                frames += 1
                run_timer.reset()
        terminator_array.disconnect()
        # empty_queue(data_queue)
        data_queue.close()
        data_queue.cancel_join_thread()
        print("save images finished")

    def on_press(self, key, terminator_array) -> bool:
        """Changes terminator flags on specific key presses

        Stops listener when both terminator flags have been set to 0

        Args:
            key: the key that was pressed
            terminator_array: A multiprocessing array to hold terminate flags
        """
        try:
            if key.char == "q":
                terminator_array.connect()
                terminator_array.write_data(0, np.array([0]))
                print("The 'q' key was pressed.")
            elif key.char == "w":
                terminator_array.connect()
                terminator_array.write_data(1, np.array([0]))
                print("The 'w' key was pressed.")
        except AttributeError:
            pass
        if not terminator_array.read_data(0) and not terminator_array.read_data(1):
            return False  # stop listener
