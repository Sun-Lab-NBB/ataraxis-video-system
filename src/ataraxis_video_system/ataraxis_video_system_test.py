import pytest

import cv2
import os
from multiprocessing import Process, Queue, ProcessError
import multiprocessing
import numpy as np
from pynput import keyboard
from shared_memory_array import SharedMemoryArray
from ataraxis_video_system import VideoSystem
from camera import Camera
from PIL import Image, ImageDraw, ImageChops
import random
import time


@pytest.fixture
def camera():
    return Camera()

@pytest.fixture
def video_system(camera):
    # Makes sure test doesn't delete contents of an actual file
    test_directory = 'foo'
    while os.path.exists(test_directory):
        test_directory += 'o'
    return VideoSystem(test_directory, camera)

def test_start(video_system):
    pass

def test_stop():
    pass

def test__delete_files_in_directory():
    
    # Makes sure test doesn't delete contents of an actual file
    test_directory = 'foo'
    while os.path.exists(test_directory):
        test_directory += 'o'

    # Test that system raises an error if writing to a directory that doesn't exist
    with pytest.raises(FileNotFoundError):
        VideoSystem._delete_files_in_directory(test_directory)

    # Creates a directory and adds text files to it
    os.makedirs(test_directory)
    for i in range(10):
        path = test_directory + '//file' + str(i) + '.txt'
        with open(path, 'w') as file:
            text = "file " + str(i) + ": to be deleted . . ."
            file.write(text)
    assert os.listdir(test_directory)
    VideoSystem._delete_files_in_directory(test_directory)
    assert not os.listdir(test_directory)
    os.rmdir(test_directory)

def create_image(seed: int):
    """A helper function that makes an image for testing purposes
    Args:
        seed: int value. Two images created with the same seed will be identical, images with different seeds will be unique
    """
    width, height = 200, 200
    color = random_color(seed)

    image = Image.new('RGB', (width, height), (255, 255, 255))
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

def test_delete_images(video_system):
    test_directory = video_system.save_directory

    # Test that system raises an error if writing to a directory that doesn't exist
    with pytest.raises(FileNotFoundError):
        video_system.delete_images()
    
    os.makedirs(test_directory)

    # Add some text files to folder
    for i in range(10):
        path = test_directory + '//file' + str(i) + '.txt'
        with open(path, 'w') as file:
            text = 'file ' + str(i) + ': to be deleted . . .'
            file.write(text)

    # Add some png images to folder
    for i in range(10):
        path = test_directory + '//img' + str(i) + '.png'
        create_image(i * 10).save(path)

    assert os.listdir(test_directory)
    video_system.delete_images()
    assert not os.listdir(test_directory)
    
    os.rmdir(test_directory)

def test__empty_queue():
    pass

def test__input_stream(camera):
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

    input_stream_process = Process(target=VideoSystem._input_stream, args=(camera, test_queue, test_array,))
    input_stream_process.start()
    time.sleep(3)

    arr = np.ndarray(shape=1, dtype=np.int32)
    arr[0] = 0
    test_array.write_data(slice(2,3), np.array([0]))
    input_stream_process.join()

    assert test_queue.qsize() == 0

    # Run input_stream as fast as possible for three seconds

    prototype = np.array([1, 1, 1], dtype=np.int32) 
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    input_stream_process = Process(target=VideoSystem._input_stream, args=(Camera(), test_queue, test_array, ),)
    input_stream_process.start()
    time.sleep(3)
    test_array.write_data(slice(2,3), np.array([0]))
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    VideoSystem._empty_queue(test_queue)
    assert test_queue.qsize() == 0

    # Run input_stream at 1 fps for 3 seconds
    prototype = np.array([1, 1, 1], dtype=np.int32) 
    test_array = SharedMemoryArray.create_array("test_array4", prototype)
    test_array.connect()

    input_stream_process = Process(target=VideoSystem._input_stream, args=(Camera(), test_queue, test_array, 3),)
    input_stream_process.start()
    time.sleep(3)
    test_array.write_data(slice(2,3), np.array([0]))
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    assert test_queue.qsize() < 5

    VideoSystem._empty_queue(test_queue)
    assert test_queue.qsize() == 0


def test__save_frame():
    pass


def test__save_images_loop():
    pass

def test_on_press():
    pass
