import os
import time
import random
import tempfile
from threading import Thread
from multiprocessing import Process

from PIL import Image, ImageDraw, ImageChops
import cv2
import numpy as np
from pynput import keyboard
import pytest
from ataraxis_data_structures import SharedMemoryArray

from ataraxis_video_system import Camera, VideoSystem
from ataraxis_video_system.video_system.vsc import MPQueue


@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = os.path.join(temp_dir, "test_metadata")
        os.makedirs(metadata_path, exist_ok=True)
        yield metadata_path


@pytest.fixture
def camera():
    return Camera()


@pytest.fixture
def video_system(camera, temp_directory):
    yield VideoSystem(temp_directory, camera)


def increment_name(name):
    """Way to change a string to potentially make it unique"""
    lst = []
    i = len(name) - 1
    while i >= 0 and name[i].isnumeric():
        lst.append(name[i])
        i -= 1
    if lst:
        lst.reverse()
        return name[: i + 1] + str(int("".join(lst)) + 1)
    return name + "0"


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
        seed: Two images created with the same seed will be identical, images with different seeds will be unique.
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
    return r, g, b


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


@pytest.mark.xdist_group(name="uses_camera_group")
def test_produce_images_loop(camera):
    test_queue = MPQueue()

    # Bad prototypes 1 and 2: when the third element is 0, the loop terminates immediately

    bad_prototype1 = np.array([0, 0, 0], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array1", prototype=bad_prototype1)
    VideoSystem._produce_images_loop(camera, test_queue, test_array)

    assert test_queue.qsize() == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array2", prototype=bad_prototype1)
    VideoSystem._produce_images_loop(camera, test_queue, test_array)

    assert test_queue.qsize() == 0

    # Bad prototype 3: with first element 0 and third element 1, loop will run but will take no pictures

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array3", prototype=bad_prototype1)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(
            camera,
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()
    time.sleep(2)

    test_array.write_data(index=2, data=0)
    input_stream_process.join()

    assert test_queue.qsize() == 0

    # Run input_stream as fast as possible for two seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_input_array4", prototype=prototype)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(
            Camera(),
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    test_queue = MPQueue()
    assert test_queue.qsize() == 0

    # Run input_stream at 1 fps for 3 seconds
    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_input_array5", prototype=prototype)
    test_array.connect()

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(Camera(), test_queue, test_array, 3),
    )
    input_stream_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
    input_stream_process.join()
    test_array.disconnect()

    assert test_queue.qsize() > 0
    assert test_queue.qsize() < 6

    test_queue = MPQueue()
    assert test_queue.qsize() == 0


def test_save_frame(temp_directory):
    test_directory = temp_directory

    num_images = 25
    test_queue = MPQueue()

    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))

    # Test if the video system correctly saves all images
    for i in range(num_images):
        assert VideoSystem._save_frame(test_queue, test_directory)

    # Test if VideoSystem returns False once queue has been emptied
    assert not VideoSystem._save_frame(test_queue, test_directory)

    # Test if image saving was performed correctly
    for i in range(num_images):
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        assert images_are_equal(PIL_path, os.path.join(test_directory, "img" + str(i) + ".png"))


def test_save_images_loop(temp_directory):
    test_directory = temp_directory

    num_images = 25
    test_queue = MPQueue()

    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))

    test_directory = temp_directory

    num_images = 25
    test_queue = MPQueue()

    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))
    assert True
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images
    assert len(os.listdir(test_directory)) == 0

    # Bad prototypes 1 and 2: when the third element is 0, the loop terminates immediately

    bad_prototype1 = np.array([0, 0, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array1", prototype=bad_prototype1)
    VideoSystem._save_images_loop(test_queue, test_array, test_directory, 5)

    assert len(os.listdir(test_directory)) == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array2", prototype=bad_prototype2)
    VideoSystem._save_images_loop(test_queue, test_array, test_directory, 5)

    assert len(os.listdir(test_directory)) == 0

    # Bad prototype 3: with second element 0 and third element 1, loop will run but save no images

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array3", prototype=bad_prototype3)
    test_array.connect()

    save_process = Thread(target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, 5))
    save_process.start()
    time.sleep(2)

    test_array.write_data(index=2, data=0)
    save_process.join()

    assert len(os.listdir(test_directory)) == 0

    # Run save_images_loop as fast as possible for two seconds

    print(test_queue.qsize())

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array4", prototype=prototype)
    test_array.connect()

    save_process = Thread(target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, 5))
    save_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
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
        test_queue.put((cv_img, i))
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images
    assert len(os.listdir(test_directory)) == 0

    # Run sav_images_loop at 1 fps for 3 seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array5", prototype=prototype)
    test_array.connect()

    save_process = Thread(target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, 5, 1))
    save_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
    save_process.join()
    test_array.disconnect()
    assert 1 < len(os.listdir(test_directory)) < 4

    # Empty the queue
    test_queue = MPQueue()
    assert False


def test_save_video_loop(temp_directory, camera):
    test_directory = temp_directory
    test_queue = MPQueue()

    # Add images to the queue
    num_images = 25
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))
    VideoSystem._delete_files_in_directory(test_directory)

    # Run save_video_loop as fast as possible for two seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array5", prototype=prototype)
    test_array.connect()

    save_process = Process(
        target=VideoSystem._save_video_loop, args=(test_queue, test_array, test_directory, camera.specs)
    )
    save_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
    save_process.join()
    test_array.disconnect()

    assert "video.mp4" in os.listdir(test_directory)

    VideoSystem._delete_files_in_directory(test_directory)

    assert "video.mp4" not in os.listdir(test_directory)

    # Run save loop at 1 fps for two seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array6", prototype=prototype)
    test_array.connect()

    save_process = Process(
        target=VideoSystem._save_video_loop, args=(test_queue, test_array, test_directory, camera.specs, 1)
    )
    save_process.start()
    time.sleep(2)
    test_array.write_data(index=2, data=0)
    save_process.join()
    test_array.disconnect()


@pytest.mark.xdist_group(name="uses_camera_group")
def test_start(video_system):
    test_directory = video_system.save_directory

    # Run in the video system in image saving mode

    assert len(os.listdir(test_directory)) == 0
    assert not video_system._running
    assert not video_system._input_process
    assert not video_system._save_process
    assert not video_system._terminator_array
    assert not video_system._image_queue
    assert not video_system.camera.is_connected

    # i = 0
    # name = "terminator_array"
    # test_array = None
    # while i < 100:
    #     try:
    #         video_system.start(terminator_array_name=name)
    #         i = 100
    #     except FileExistsError:
    #         name = increment_name(name)
    #         i += 1
    video_system.start(terminator_array_name="terminator_array")

    assert video_system._running
    assert video_system._input_process
    assert video_system._save_process
    assert video_system._terminator_array
    assert video_system._image_queue

    time.sleep(10)
    print(video_system._image_queue.qsize())

    assert len(os.listdir(test_directory)) > 0

    video_system.stop()
    video_system.delete_images()

    assert False


@pytest.mark.xdist_group(name="uses_camera_group")
def test_mp4_save(video_system):
    test_directory = video_system.save_directory

    # Run in the video system in video saving mode

    assert len(os.listdir(test_directory)) == 0
    assert not video_system._running
    assert not video_system._input_process
    assert not video_system._save_process
    assert not video_system._terminator_array
    assert not video_system._image_queue
    assert not video_system.camera.is_connected

    video_system.start(terminator_array_name="terminator_array1", save_format="mp4")

    assert video_system._running
    assert video_system._input_process
    assert video_system._save_process
    assert video_system._terminator_array
    assert video_system._image_queue

    time.sleep(2)

    video_system.stop()

    assert "video.mp4" in os.listdir(test_directory)


@pytest.mark.xdist_group(name="uses_camera_group")
def test_stop_image_production(video_system):
    test_directory = video_system.save_directory

    # i = 0
    # name = "terminator_array"
    # test_array = None
    # while i < 100:
    #     try:
    #         video_system.start(terminator_array_name=name)
    #         i = 100
    #     except FileExistsError:
    #         name = increment_name(name)
    #         i += 1
    video_system.start(terminator_array_name="terminator_array2")

    time.sleep(2)
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    video_system.stop_image_production()
    time.sleep(2)
    # Once you stop taking images, the size of the queue should start decreasing
    assert images_taken >= video_system._image_queue.qsize()

    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) > images_saved

    video_system.stop()


@pytest.mark.xdist_group(name="uses_camera_group")
def test_stop_image_saving(video_system):
    test_directory = video_system.save_directory

    # i = 0
    # name = "terminator_array"
    # test_array = None
    # while i < 100:
    #     try:
    #         video_system.start(terminator_array_name=name)
    #         i = 100
    #     except FileExistsError:
    #         name = increment_name(name)
    #         i += 1
    video_system.start(terminator_array_name="terminator_array3")

    time.sleep(2)
    video_system._stop_image_saving()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    time.sleep(2)

    # Once you stop saving images, the size of the queue should continue to increase
    assert images_taken < video_system._image_queue.qsize()

    # Once you stop saving images, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    video_system.stop()


@pytest.mark.xdist_group(name="uses_camera_group")
def test_stop(video_system):
    test_directory = video_system.save_directory

    # i = 0
    # name = "terminator_array"
    # test_array = None
    # while i < 100:
    #     try:
    #         video_system.start(terminator_array_name=name)
    #         i = 100
    #     except FileExistsError:
    #         name = increment_name(name)
    #         i += 1
    video_system.start(terminator_array_name="terminator_array4")

    time.sleep(2)
    video_system.stop()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    time.sleep(2)

    # Once you stop the video system, the size of the queue should continue to increase
    assert images_taken == video_system._image_queue.qsize()

    # Once you stop the video system, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    assert not video_system._running


@pytest.mark.xdist_group(name="uses_camera_group")
def test_key_listener(video_system):
    test_directory = video_system.save_directory
    controller = keyboard.Controller()

    video_system.start(listen_for_keypress=True, terminator_array_name="terminator_array5")

    time.sleep(2)

    # Print the random key to make sure key listener can handle all keys without throwing error
    controller.press(keyboard.Key.alt)
    controller.release(keyboard.Key.alt)

    # Stop image production
    controller.press("q")
    controller.release("q")
    controller.press(keyboard.Key.delete)
    controller.release(keyboard.Key.delete)

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    time.sleep(2)

    # Once you stop taking images, the size of the queue should start decreasing
    assert images_taken >= video_system._image_queue.qsize()
    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) >= images_saved

    # Stop image saving
    controller.press("w")
    controller.release("w")
    controller.press(keyboard.Key.delete)
    controller.release(keyboard.Key.delete)

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    time.sleep(2)

    # Once you stop the video system, the size of the queue should not continue to increase
    assert images_taken == video_system._image_queue.qsize()

    # Once you stop the video system, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) == images_saved

    assert not video_system._running

    # Now check that with manual stopping in interactive mode, the key_listener actually closes
    video_system.start(listen_for_keypress=True, terminator_array_name="terminator_array6")
    time.sleep(2)

    assert video_system._running
    video_system.stop()
    assert not video_system._running

    # Key listener has a safety mechanism where the listener stops if running gets set to false. This normally doesn't
    # occur, so it has to be tested directly
    video_system.start(listen_for_keypress=True, terminator_array_name="terminator_array7")
    time.sleep(2)
    assert video_system._running
    assert video_system._listener._running
    video_system._running = False
    # Press a key to trigger on_press callback to close listener
    assert not video_system._running
    assert video_system._listener._running
    controller.press(keyboard.Key.alt)
    controller.release(keyboard.Key.alt)
    time.sleep(1)
    assert not video_system._listener._running
    video_system._running = True
    video_system.stop()
    assert not video_system._running


@pytest.mark.xdist_group(name="uses_camera_group")
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

    # Now try to make a camera with incorrect id
    with pytest.raises(Exception):
        bad_camera = Camera(1000)
        bad_camera.connect()
        bad_camera.grab_frame()
        bad_camera.disconnect()

    with pytest.raises(Exception, match="camera did not yield an image"):
        bad_camera = Camera(-1)
        bad_camera.connect()
        bad_camera.grab_frame()
        bad_camera.disconnect()


# For type checking purposes, some object | None type variables have been put in an if not None call else raise
# TypeError structure. These errors should never get raised by user but are tested here to get code coverage.
@pytest.mark.xdist_group(name="uses_camera_group")
def test_type_errors(video_system):
    video_system.start()

    temp = video_system._terminator_array
    video_system._terminator_array = None

    with pytest.raises(TypeError):
        video_system.stop_image_production()

    with pytest.raises(TypeError):
        video_system._stop_image_saving()

    with pytest.raises(TypeError):
        video_system.stop()

    video_system._terminator_array = temp
    video_system.stop()

    video_system.start()

    temp = video_system._save_process
    video_system._save_process = None

    with pytest.raises(TypeError):
        video_system.stop()

    video_system._save_process = temp

    video_system.start()
    video_system.stop()

    video_system.start()

    temp = video_system._input_process
    video_system._input_process = None

    with pytest.raises(TypeError):
        video_system.stop()

    video_system._input_process = temp

    video_system.start()
    video_system.stop()
