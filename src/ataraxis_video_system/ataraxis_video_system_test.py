import pytest

import cv2
import os
from multiprocessing import Process, Queue, ProcessError
import numpy as np
from pynput import keyboard
from shared_memory_array import SharedMemoryArray
from ataraxis_video_system import VideoSystem, Camera
from PIL import Image, ImageDraw, ImageChops
import random
import time
import tempfile

@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = os.path.join(temp_dir, 'test_metadata')
        os.makedirs(metadata_path, exist_ok=True)
        yield(metadata_path)

@pytest.fixture
def camera():
    return Camera()

@pytest.fixture
def video_system(camera, temp_directory):
    yield VideoSystem(temp_directory, camera)

def test_delete_files_in_directory(temp_directory):
    test_directory = temp_directory

    for i in range(10):
        path = os.path.join(test_directory, "file" + str(i) + ".txt")
        with open(path, "w") as file:
            text = "file " + str(i) + ": to be deleted . . ."
            file.write(text)
    assert os.listdir(test_directory)
    VideoSystem._delete_files_in_directory(test_directory)
    assert not os.listdir(test_directory)

    # Test that system raises an error if writing to a directory that doesn't exist
    os.rmdir(test_directory)
    with pytest.raises(FileNotFoundError):
        VideoSystem._delete_files_in_directory(test_directory)


def create_image(seed: int):
    """A helper function that makes an image for testing purposes
    Args:
        seed: int value. Two images created with the same seed will be identical, images with different seeds will be unique
    """
    width, height = 200, 200
    color = random_color(seed)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.ellipse([70, 70, 130, 130], fill=color, outline=(0, 0, 0))  # Green circle with black outline

    return image


def random_color(seed):
    """helper function to get color from seed"""
    random.seed(seed)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def images_are_equal(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size != image2.size or image1.mode != image2.mode:
        return False

    diff = ImageChops.difference(image1, image2)
    return not diff.getbbox()  # Returns True if images are identical


def test_delete_images(video_system):
    test_directory = video_system.save_directory

    # Add some text files to folder
    for i in range(10):
        path = os.path.join(test_directory, "file" + str(i) + ".txt")
        with open(path, "w") as file:
            text = "file " + str(i) + ": to be deleted . . ."
            file.write(text)

    # Add some png images to folder
    for i in range(10):
        path = os.path.join(test_directory, "img" + str(i) + ".png")
        create_image(i * 10).save(path)

    assert os.listdir(test_directory)
    video_system.delete_images()
    assert not os.listdir(test_directory)


    # Test that system raises an error if writing to a directory that doesn't exist
    os.rmdir(test_directory)
    with pytest.raises(FileNotFoundError):
        video_system.delete_images()

def test_input_stream(camera):
    test_queue = Queue()

    # Bad prototypes 1 and 2: when third element is 0, the loop terminates immediately

    bad_prototype1 = np.array([0, 0, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array1", bad_prototype1)
    VideoSystem._input_stream(camera, test_queue, test_array)

    assert test_queue.qsize() == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array2", bad_prototype2)
    VideoSystem._input_stream(camera, test_queue, test_array)

    assert test_queue.qsize() == 0

    # Bad prototype 3: with first element 0 and third element 1, loop will run but will take no pictures

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array3", bad_prototype3)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._input_stream,
        args=(
            camera,
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()
    time.sleep(3)

    test_array.write_data(slice(2, 3), np.array([0]))
    input_stream_process.join()

    assert test_queue.qsize() == 0

    # Run input_stream as fast as possible for three seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._input_stream,
        args=(
            Camera(),
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()
    time.sleep(3)
    test_array.write_data(slice(2, 3), np.array([0]))
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    test_queue = Queue()
    assert test_queue.qsize() == 0

    # Run input_stream at 1 fps for 3 seconds
    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._input_stream,
        args=(Camera(), test_queue, test_array, 3),
    )
    input_stream_process.start()
    time.sleep(3)
    test_array.write_data(slice(2, 3), np.array([0]))
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    assert test_queue.qsize() < 5

    test_queue = Queue()
    assert test_queue.qsize() == 0


def test_save_frame(temp_directory):
    test_directory = temp_directory

    num_images = 25
    test_queue = Queue()
    

    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put(cv_img)
    
    # Test if video system correctly saves all images
    for i in range(num_images):
        assert VideoSystem._save_frame(test_queue, test_directory, i)

    # Test if VideoSystem returns False once queue has been emptied
    assert not VideoSystem._save_frame(test_queue, test_directory, 31)

    # Test if image saving was performed correctly
    for i in range(num_images):
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        assert images_are_equal(PIL_path, os.path.join(test_directory, 'img' + str(i) + '.png'))

def test_save_images_loop(temp_directory):
    test_directory = temp_directory

    test_queue = Queue()

    # Add images to the queue 
    num_images = 25
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put(cv_img)
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images
    assert len(os.listdir(test_directory)) == 0

    # Bad prototypes 1 and 2: when third element is 0, the loop terminates immediately

    bad_prototype1 = np.array([0, 0, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array1", bad_prototype1)
    VideoSystem._save_images_loop(test_queue, test_array, test_directory)

    assert len(os.listdir(test_directory)) == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array2", bad_prototype2)
    VideoSystem._save_images_loop(test_queue, test_array, test_directory)

    assert len(os.listdir(test_directory)) == 0

    # Bad prototype 3: with second element 0 and third element 1, loop will run but save no images

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array3", bad_prototype3)
    test_array.connect()

    save_process = Process(target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory))
    save_process.start()
    time.sleep(3)

    test_array.write_data(slice(2, 3), np.array([0]))
    save_process.join()

    assert len(os.listdir(test_directory)) == 0

    # Run sav_images_loop as fast as possible for three seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    save_process = Process(
        target=VideoSystem._save_images_loop,
        args=(test_queue, test_array, test_directory)
    )
    save_process.start()
    time.sleep(3)
    test_array.write_data(slice(2, 3), np.array([0]))
    save_process.join()
    test_array.disconnect()

    assert len(os.listdir(test_directory)) == num_images

    # clear directory and reload images into queue
    VideoSystem._delete_files_in_directory(test_directory)
    num_images = 25
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put(cv_img)
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images
    assert len(os.listdir(test_directory)) == 0

    # Run sav_images_loop at 1 fps for 3 seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    save_process = Process(
        target=VideoSystem._save_images_loop,
        args=(test_queue, test_array, test_directory, 1)
    )
    save_process.start()
    time.sleep(3)
    test_array.write_data(slice(2, 3), np.array([0]))
    save_process.join()
    test_array.disconnect()

    assert 1 < len(os.listdir(test_directory))  < 4


def test_start(video_system):
    test_directory = video_system.save_directory

    assert len(os.listdir(test_directory)) == 0

    assert not video_system._running
    assert not video_system._input_process
    assert not video_system._save_process
    assert not video_system._terminator_array
    assert not video_system._image_queue
    assert not video_system.camera.is_connected

    video_system.start()

    assert video_system._running
    assert video_system._input_process
    assert video_system._save_process
    assert video_system._terminator_array
    assert video_system._image_queue

    time.sleep(3)
        
    assert len(os.listdir(test_directory)) > 0
    assert video_system._image_queue.qsize() > 0

    video_system.stop()

def test_stop_image_collection(video_system):
    test_directory = video_system.save_directory
    video_system.start()
    time.sleep(3)
    video_system.stop_image_collection()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    time.sleep(3)
    # Once you stop taking images, the size of the queue should start decreasing
    assert images_taken > video_system._image_queue.qsize()

    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) > images_saved

    video_system.stop()

def test_stop_image_saving(video_system):
    test_directory = video_system.save_directory
    video_system.start()
    time.sleep(3)
    video_system._stop_image_saving()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    time.sleep(3)

    # Once you stop saving images, the size of the queue should continue to increase
    assert images_taken < video_system._image_queue.qsize()

    # Once you stop saving images, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    video_system.stop()

def test_stop(video_system):
    test_directory = video_system.save_directory
    video_system.start()
    time.sleep(3)
    video_system.stop()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    time.sleep(3)

    # Once you stop the video system, the size of the queue should continue to increase
    assert images_taken == video_system._image_queue.qsize()

    # Once you stop the video system, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    assert not video_system._running


def test_key_listener(video_system):
    test_directory = video_system.save_directory
    controller = keyboard.Controller()

    video_system.start(listen_for_keypress = True)
    time.sleep(3)

    controller.press('q')
    controller.release('q')

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    time.sleep(3)

    # Once you stop taking images, the size of the queue should start decreasing
    assert images_taken > video_system._image_queue.qsize()
    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) > images_saved

    controller.press('w')
    controller.release('w')

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    time.sleep(3)

    assert True

    # Once you stop the video system, the size of the queue should continue to increase
    assert images_taken == video_system._image_queue.qsize()

    # Once you stop the video system, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    assert not video_system._running

def test_camera(camera):
    assert not camera.is_connected

    with pytest.raises(Exception):
        camera.grab_frame()

    camera.connect()
    assert camera.is_connected

    frame = camera.grab_frame()
    assert frame is not None

    camera.disconnect()
    assert not camera.is_connected
