# import the opencv library
import cv2
import os
from high_precision_timer.precision_timer import PrecisionTimer
from multiprocessing import Process, Queue
import numpy as np  # temp
from pynput import keyboard
from shared_memory_array import SharedMemoryArray

d = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}

unit = "ms"


def delete_files_in_directory(path):
    try:
        with os.scandir(path) as img_files:
            for img_file in img_files:
                if img_file.is_file():
                    os.unlink(img_file.path)
    except OSError:
        print("Error occurred while deleting files.")


def run_time_control(func, fps, verbose=False):
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)

    frames = 0

    while True:
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            func()
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print("fps:", frames)
            check_timer.reset()
            frames = 0


def get_frame(camera, img_queue):
    ret, frame = camera.read()
    img_queue.put(frame)
    print(ret)


def input_stream(data_queue, terminator_array, fps=None, verbose=False):
    terminator_array.connect()
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)
    frames = 0
    vid = cv2.VideoCapture(0)
    while terminator_array.read_data(0):
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            get_frame(vid, data_queue)
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print("fps:", frames)
            check_timer.reset()
            frames = 0
    vid.release()
    terminator_array.disconnect()
    print("input stream finished")


def save_frame(img_queue, img_id):
    if not img_queue.empty():
        frame = img_queue.get()
        filename = "imgs\\img" + str(img_id) + ".png"
        cv2.imwrite(filename, frame)
        return True
    return False


def empty_queue(q):
    while not q.empty():
        q.get()
    print("queue empty:", q.empty())


def save_images_loop(data_queue, terminator_array, fps=None, verbose=False):
    terminator_array.connect()
    num_imgs_saved = 0
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)
    frames = 0
    while terminator_array.read_data(1):
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            saved = save_frame(data_queue, num_imgs_saved)
            if saved:
                num_imgs_saved += 1
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print("fps:", frames)
            check_timer.reset()
            frames = 0
    terminator_array.disconnect()
    # empty_queue(data_queue)
    data_queue.close()
    data_queue.cancel_join_thread()
    print("save images finished")


def on_press(key, data_queue, terminator_array):
    if key == keyboard.Key.esc:
        terminator_array.disconnect()
        return False  # stop listener
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


if __name__ == "__main__":
    delete_files_in_directory("imgs")

    data_queue = Queue()

    prototype = np.array(
        [1, 1], dtype=np.int32
    )  # First entry represents whether input stream is active, second entry represents whether output stream is active
    terminator_array = SharedMemoryArray.create_array("terminator_array", prototype)

    # p1 = Process(target=run_time_control, args=(lambda: get_frame(q), 2), daemon=True)
    # p2 = Process(target=save_frame, args=(lambda: save_frame(q), 1), daemon=True)

    listener = keyboard.Listener(on_press=lambda x: on_press(x, data_queue, terminator_array))
    p1 = Process(
        target=input_stream,
        args=(
            data_queue,
            terminator_array,
            2,
        ),
        daemon=True,
    )
    p2 = Process(
        target=save_images_loop,
        args=(
            data_queue,
            terminator_array,
            1,
        ),
        daemon=True,
    )

    listener.start()  # start to listen on a separate thread
    p1.start()
    p2.start()

    p1.join()
    print("input stream joined")
    p2.join()
    print("save images joined")

    listener.join()
    print("listener joined")

    cv2.destroyAllWindows()

    print("Done")


# run_time_control(foo, 5, verbose=True)
