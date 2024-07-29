import os
from queue import Queue
import random
import tempfile
from threading import Thread
import subprocess
import multiprocessing
from multiprocessing import Process, ProcessError

from PIL import Image, ImageDraw, ImageChops
import cv2
import numpy as np
import pytest
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import SharedMemoryArray

from ataraxis_video_system import Camera, MPQueue, VideoSystem


class MockCamera:
    """A wrapper class for an opencv VideoCapture object.

    Args:
        camera_id: camera id

    Attributes:
        camera_id: camera id
        specs: dictionary holding the specifications of the camera. This includes fps, frame_width, frame_height
        _vid: opencv video capture object.
    """

    def __init__(self, camera_id: int = 0) -> None:
        self.specs = {"fps": 30.0, "frame_width": 640.0, "frame_height": 480.0}
        self.camera_id = camera_id
        self._vid = None

    def connect(self) -> None:
        """Connects to camera and prepares for image collection."""
        self._vid = True

    def disconnect(self) -> None:
        """Disconnects from camera."""
        self._vid = None

    @property
    def is_connected(self) -> bool:
        """Whether the camera is connected."""
        return self._vid is not None

    def grab_frame(self):
        """Grabs an image from the camera.

        Raises:
            Exception if camera isn't connected or did not yield an image.

        """
        if self._vid:
            return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        else:
            raise Exception("camera not connected")


@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = os.path.join(temp_dir, "test_metadata")
        os.makedirs(metadata_path, exist_ok=True)
        yield metadata_path


@pytest.fixture
def webcam():
    return Camera()


@pytest.fixture
def mock_camera():
    return MockCamera()


@pytest.fixture
def video_system(mock_camera, temp_directory):
    yield VideoSystem(temp_directory, mock_camera, mp4_config={"crf": 25, "refs": 10})


wait_time = 5


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


def test_produce_images_loop(mock_camera):
    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()

    # Bad prototypes 1 and 2: when the third element is 0, the loop terminates immediately

    bad_prototype1 = np.array([0, 0, 0], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array1", prototype=bad_prototype1)
    VideoSystem._produce_images_loop(mock_camera, test_queue, test_array)
    test_array.disconnect()
    test_array._buffer.unlink()

    assert test_queue.qsize() == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array2", prototype=bad_prototype1)
    VideoSystem._produce_images_loop(mock_camera, test_queue, test_array)
    test_array.disconnect()
    test_array._buffer.unlink()

    assert test_queue.qsize() == 0

    # Bad prototype 3: with first element 0 and third element 1, loop will run but will take no pictures

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)

    test_array = SharedMemoryArray.create_array(name="test_input_array3", prototype=bad_prototype1)

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(
            mock_camera,
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()
    test_array.write_data(index=2, data=0)
    input_stream_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    assert test_queue.qsize() == 0

    # Run input_stream as fast as possible until it saves 10 images

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_input_array4", prototype=prototype)

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(
            mock_camera,
            test_queue,
            test_array,
        ),
    )
    input_stream_process.start()

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and test_queue.qsize() < 10:
        pass

    assert test_queue.qsize() >= 10

    test_array.write_data(index=2, data=0)
    test_manager.shutdown()
    input_stream_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()

    # Run input_stream at 60 fps until is saves 10 images
    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_input_array5", prototype=prototype)

    input_stream_process = Process(
        target=VideoSystem._produce_images_loop,
        args=(mock_camera, test_queue, test_array, False, 60),
    )
    input_stream_process.start()
    timer.reset()
    while timer.elapsed <= wait_time and test_queue.qsize() < 10:
        pass
    assert test_queue.qsize() >= 10
    test_array.write_data(index=2, data=0)
    test_manager.shutdown()
    input_stream_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()


def test_imwrite(temp_directory):
    test_directory = temp_directory

    # png

    img = create_image(81)
    PIL_path = os.path.join(test_directory, "PIL_img" + ".png")
    img.save(PIL_path)
    cv_img = cv2.imread(PIL_path)
    path = os.path.join(test_directory, "img.png")
    VideoSystem.imwrite(path, cv_img)
    assert images_are_equal(PIL_path, os.path.join(test_directory, path))

    VideoSystem._delete_files_in_directory(test_directory)
    assert len(os.listdir(test_directory)) == 0

    # jpg

    img = create_image(81)
    PIL_path = os.path.join(test_directory, "PIL_img" + ".png")
    img.save(PIL_path)
    cv_img = cv2.imread(PIL_path)
    path = os.path.join(test_directory, "img.jpg")
    VideoSystem.imwrite(path, cv_img)
    assert os.path.exists(path)

    VideoSystem._delete_files_in_directory(test_directory)
    assert len(os.listdir(test_directory)) == 0

    # tiff
    img = create_image(81)
    PIL_path = os.path.join(test_directory, "PIL_img" + ".png")
    img.save(PIL_path)
    cv_img = cv2.imread(PIL_path)
    path = os.path.join(test_directory, "img.tiff")
    VideoSystem.imwrite(path, cv_img)
    assert os.path.exists(path)


def test_frame_saver(temp_directory):
    test_directory = temp_directory

    num_images = 25

    # Png saving
    test_queue = Queue()

    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))

    test_queue.put((None, -1))

    VideoSystem._frame_saver(test_queue, test_directory, "png", 6, 95)

    assert test_queue.empty()
    assert test_queue.all_tasks_done

    # Test if image saving was performed correctly
    for i in range(num_images):
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        assert images_are_equal(PIL_path, os.path.join(test_directory, "img" + str(i) + ".png"))

    VideoSystem._delete_files_in_directory(test_directory)
    assert len(os.listdir(test_directory)) == 0

    # jpg saving
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))

    test_queue.put((None, -1))

    VideoSystem._frame_saver(test_queue, test_directory, "jpg", 6, 95)

    assert test_queue.empty()
    assert test_queue.all_tasks_done

    # Test if image saving was performed correctly
    for i in range(num_images):
        assert os.path.exists(os.path.join(test_directory, "img" + str(i) + ".jpg"))

    VideoSystem._delete_files_in_directory(test_directory)
    assert len(os.listdir(test_directory)) == 0

    # tiff saving
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))

    test_queue.put((None, -1))

    VideoSystem._frame_saver(test_queue, test_directory, "tiff", 6, 95)

    assert test_queue.empty()
    assert test_queue.all_tasks_done

    # Test if image saving was performed correctly
    for i in range(num_images):
        assert os.path.exists(os.path.join(test_directory, "img" + str(i) + ".tiff"))

    VideoSystem._delete_files_in_directory(test_directory)
    assert len(os.listdir(test_directory)) == 0


def test_save_images_loop(temp_directory):
    test_directory = temp_directory

    num_images = 25
    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()

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
    VideoSystem._save_images_loop(test_queue, test_array, test_directory, "png", 6, 95, 5, None)
    test_array.disconnect()
    test_array._buffer.unlink()

    assert len(os.listdir(test_directory)) == 0

    bad_prototype2 = np.array([1, 1, 0], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array2", prototype=bad_prototype2)
    VideoSystem._save_images_loop(test_queue, test_array, test_directory, "png", 6, 95, 5, None)
    test_array.disconnect()
    test_array._buffer.unlink()

    assert len(os.listdir(test_directory)) == 0

    # Bad prototype 3: with second element 0 and third element 1, loop will run but save no images

    bad_prototype3 = np.array([0, 0, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array3", prototype=bad_prototype3)

    save_process = Thread(
        target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, "png", 6, 95, 5, None)
    )
    save_process.start()

    test_array.write_data(index=2, data=0)
    save_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    assert len(os.listdir(test_directory)) == 0

    # Run save_images_loop as fast as possible for two seconds

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array4", prototype=prototype)

    save_process = Process(
        target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, "png", 6, 95, 5, None)
    )
    save_process.start()

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < num_images:
        pass

    test_array.write_data(index=2, data=0)
    test_manager.shutdown()
    save_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    assert len(os.listdir(test_directory)) == num_images

    # clear directory and reload images into queue

    VideoSystem._delete_files_in_directory(test_directory)
    num_images = 25
    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images
    assert len(os.listdir(test_directory)) == 0

    # Run sav_images_loop at 60 fps until it saves all images

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array5", prototype=prototype)

    save_process = Thread(
        target=VideoSystem._save_images_loop, args=(test_queue, test_array, test_directory, "png", 6, 95, 5, 60)
    )
    save_process.start()
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < num_images:
        pass
    test_array.write_data(index=2, data=0)
    test_manager.shutdown()
    save_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    assert len(os.listdir(test_directory)) > 0


def test_save_video_loop(temp_directory, mock_camera):
    test_directory = temp_directory
    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()
    num_images = 25
    timer = PrecisionTimer("s")
    config = {
        "codec": "h264",
        "preset": "slow",
        "profile": "main",
        "crf": 28,
        "quality": 23,
        "threads": 0,
    }

    # Load images to the queue
    num_images = 25
    for i in range(num_images):
        img = create_image(i * 10)
        PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
        img.save(PIL_path)
        cv_img = cv2.imread(PIL_path)
        test_queue.put((cv_img, i))
    VideoSystem._delete_files_in_directory(test_directory)

    assert test_queue.qsize() == num_images

    # Run save loop at 60 fps until it saves 10 frames

    prototype = np.array([1, 1, 1], dtype=np.int32)
    test_array = SharedMemoryArray.create_array(name="test_save_array6", prototype=prototype)

    save_process = Process(
        target=VideoSystem._save_video_loop,
        args=(test_queue, test_array, test_directory, mock_camera.specs, config, 60),
    )
    save_process.start()

    timer.reset()
    while timer.elapsed <= wait_time and test_queue.qsize() > 0:
        pass
    assert test_queue.qsize() == 0

    test_array.write_data(index=2, data=0)
    test_manager.shutdown()
    save_process.join()
    test_array.disconnect()
    test_array._buffer.unlink()

    assert "video.mp4" in os.listdir(test_directory)

    VideoSystem._delete_files_in_directory(test_directory)

    assert "video.mp4" not in os.listdir(test_directory)

    def save_video_test_by_codec(codec, i):
        test_manager = multiprocessing.Manager()
        test_queue = test_manager.Queue()
        config["codec"] = codec

        # Add images to the queue
        for i in range(num_images):
            img = create_image(i * 10)
            PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
            img.save(PIL_path)
            cv_img = cv2.imread(PIL_path)
            test_queue.put((cv_img, i))
        VideoSystem._delete_files_in_directory(test_directory)

        # Run save_video_loop as fast as possible until it creates a video

        prototype = np.array([1, 1, 1], dtype=np.int32)
        test_array = SharedMemoryArray.create_array(name="test_save_array" + str(i), prototype=prototype)

        save_process = Process(
            target=VideoSystem._save_video_loop,
            args=(test_queue, test_array, test_directory, mock_camera.specs, config),
        )
        save_process.start()

        timer.reset()
        while timer.elapsed <= wait_time and test_queue.qsize() > 0:
            pass
        assert test_queue.qsize() == 0

        test_array.write_data(index=2, data=0)
        test_manager.shutdown()
        save_process.join()
        test_array.disconnect()
        test_array._buffer.unlink()

        assert "video.mp4" in os.listdir(test_directory)

        VideoSystem._delete_files_in_directory(test_directory)

        assert "video.mp4" not in os.listdir(test_directory)

    save_video_test_by_codec("h264", 7)
    save_video_test_by_codec("libx264", 8)
    save_video_test_by_codec("hevc", 9)
    save_video_test_by_codec("libx265", 10)


def gpu_available():
    try:
        subprocess.run(
            f"nvidia-smi",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


@pytest.mark.skipif(not gpu_available(), reason="No gpu available on this computer")
def test_save_video_loop_gpu(temp_directory, mock_camera):
    test_directory = temp_directory
    test_manager = multiprocessing.Manager()
    test_queue = test_manager.Queue()
    num_images = 25
    timer = PrecisionTimer("s")
    config = {
        "codec": "h264",
        "preset": "slow",
        "profile": "main",
        "crf": 28,
        "quality": 23,
        "threads": 0,
    }

    def save_video_test_by_codec_gpu(codec, i):
        config["codec"] = codec

        # Add images to the queue
        for i in range(num_images):
            img = create_image(i * 10)
            PIL_path = os.path.join(test_directory, "PIL_img" + str(i) + ".png")
            img.save(PIL_path)
            cv_img = cv2.imread(PIL_path)
            test_queue.put((cv_img, i))
        VideoSystem._delete_files_in_directory(test_directory)

        # Run save_video_loop as fast as possible until it creates a video

        prototype = np.array([1, 1, 1], dtype=np.int32)
        test_array = SharedMemoryArray.create_array(name="test_save_array_gpu" + str(i), prototype=prototype)

        save_process = Process(
            target=VideoSystem._save_video_loop,
            args=(test_queue, test_array, test_directory, mock_camera.specs, config),
        )
        save_process.start()

        timer.reset()
        while timer.elapsed <= wait_time and test_queue.qsize() > 0:
            pass
        assert test_queue.qsize() == 0

        test_array.write_data(index=2, data=0)
        test_manager.shutdown()
        save_process.join()
        test_array.disconnect()
        test_array._buffer.unlink()

        assert "video.mp4" in os.listdir(test_directory)

        VideoSystem._delete_files_in_directory(test_directory)

        assert "video.mp4" not in os.listdir(test_directory)

    save_video_test_by_codec_gpu("h264_mf", 7)
    save_video_test_by_codec_gpu("h264_nvenc", 8)
    save_video_test_by_codec_gpu("hevc_mf", 9)
    save_video_test_by_codec_gpu("hevc_nvenc", 10)


def test_imgs_to_vid(video_system):
    test_directory = video_system.save_directory

    with pytest.raises(Exception):
        VideoSystem.imgs_to_vid(video_system.camera.specs["fps"], img_directory=test_directory)

    video_system.start(terminator_array_name="terminator_array8", display_video=False)
    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass
    video_system.stop()

    VideoSystem.imgs_to_vid(video_system.camera.specs["fps"], img_directory=test_directory)

    assert "video.mp4" in os.listdir(test_directory)

    video_system.delete_images()

    assert "video.mp4" not in os.listdir(test_directory)

    test_directory = video_system.save_directory

    video_system.start(terminator_array_name="terminator_array9", display_video=False)
    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass
    video_system.stop()

    video_system.save_imgs_as_vid()

    assert "video.mp4" in os.listdir(test_directory)

    video_system.delete_images()

    assert "video.mp4" not in os.listdir(test_directory)


def test_start(video_system):
    test_directory = video_system.save_directory

    # Run in the video system in image saving mode

    assert len(os.listdir(test_directory)) == 0
    assert not video_system._running
    assert not video_system._producer_process
    assert not video_system._terminator_array
    assert not video_system._image_queue
    assert not video_system.camera.is_connected

    video_system.start(
        terminator_array_name="terminator_array",
        tiff_compression_level=5,
        jpeg_quality=90,
        display_video=False,
        mp4_config={"crf": 25, "refs": 9},
    )

    assert video_system._running
    assert video_system._producer_process
    assert video_system._terminator_array
    assert video_system._image_queue

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass

    video_system.stop()

    assert len(os.listdir(test_directory)) >= 10
    video_system.delete_images()
    assert len(os.listdir(test_directory)) == 0


def test_mp4_save(video_system):
    test_directory = video_system.save_directory

    # Run in the video system in video saving mode

    assert len(os.listdir(test_directory)) == 0
    assert not video_system._running
    assert not video_system._producer_process
    assert not video_system._terminator_array
    assert not video_system._image_queue
    assert not video_system.camera.is_connected

    video_system.start(terminator_array_name="terminator_array1", save_format="mp4", display_video=True)

    assert video_system._running
    assert video_system._producer_process
    assert video_system._terminator_array
    assert video_system._image_queue

    timer = PrecisionTimer("s")

    timer.delay_noblock(wait_time)

    video_system.stop()

    assert "video.mp4" in os.listdir(test_directory)

    cap = cv2.VideoCapture(os.path.join(video_system.save_directory, "video.mp4"))

    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


def test_stop_image_production(video_system):
    test_directory = video_system.save_directory

    video_system.start(terminator_array_name="terminator_array2", display_video=False)

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))
    video_system.stop_image_production()

    timer.reset()
    while timer.elapsed <= wait_time and video_system._image_queue.qsize() > 0:
        pass

    # Once you stop taking images, the size of the queue should become 0
    assert video_system._image_queue.qsize() == 0

    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) >= images_saved

    video_system.stop()

    assert not video_system._running
    assert not video_system.camera.is_connected


def test_stop_image_saving(video_system):
    test_directory = video_system.save_directory

    video_system.start(terminator_array_name="terminator_array3", display_video=False)

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass

    video_system._stop_image_saving()
    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    # Wait to save 10 more images
    timer.reset()
    while timer.elapsed <= wait_time and video_system._image_queue.qsize() < images_taken + 10:
        pass

    # You should have saved 10 more images
    assert video_system._image_queue.qsize() >= images_taken + 10

    # Once you stop saving images, the number of saved images should remain the same
    assert len(os.listdir(test_directory)) >= images_saved

    video_system.stop()

    assert not video_system._running
    assert not video_system.camera.is_connected


def test_stop(video_system):
    test_directory = video_system.save_directory

    video_system.start(terminator_array_name="terminator_array4", num_processes=5, num_threads=6, display_video=False)

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        pass

    video_system.stop()

    assert len(os.listdir(test_directory)) >= 10

    assert not video_system._running
    assert not video_system.camera.is_connected


def test_key_listener(video_system):
    test_directory = video_system.save_directory
    delay_timer = PrecisionTimer("us")

    video_system.start(
        listen_for_keypress=True, terminator_array_name="terminator_array_key_listener", display_video=False
    )

    timer = PrecisionTimer("s")
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        delay_timer.delay_noblock(1)

    images_taken = video_system._image_queue.qsize()
    images_saved = len(os.listdir(test_directory))

    # Stop image production
    video_system._on_press_q()

    timer.reset()
    while timer.elapsed <= wait_time and video_system._image_queue.qsize() > 0:
        delay_timer.delay_noblock(1)

    # Stop image saving
    video_system._on_press_w()

    timer.delay_noblock(1)

    # Once you stop taking images, the number of saved images should continue to increase
    assert len(os.listdir(test_directory)) >= images_saved

    assert not video_system._running
    assert not video_system.camera.is_connected

    video_system.delete_images()
    assert len(os.listdir(test_directory)) == 0

    # Now check that with manual stopping in interactive mode, the key_listener actually closes
    video_system.start(listen_for_keypress=True, terminator_array_name="terminator_array6", display_video=False)
    timer.reset()
    while timer.elapsed <= wait_time and len(os.listdir(test_directory)) < 10:
        delay_timer.delay_noblock(1)

    assert video_system._running
    video_system.stop()

    assert len(os.listdir(test_directory)) >= 10
    assert not video_system._running

    video_system.delete_images()
    assert len(os.listdir(test_directory)) == 0


def camera_available():
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cap.release()
        return True
    return False


@pytest.mark.skipif(not camera_available(), reason="No camera available on this computer")
@pytest.mark.xdist_group(name="uses_camera_group")
def test_camera(webcam):
    assert not webcam.is_connected

    with pytest.raises(Exception):
        webcam.grab_frame()

    webcam.connect()
    assert webcam.is_connected

    frame = webcam.grab_frame()
    assert frame is not None

    webcam.disconnect()
    assert not webcam.is_connected

    # Now try to make a camera with incorrect id
    with pytest.raises(Exception):
        bad_camera = Camera(1000)
        bad_camera.connect()
        bad_camera.grab_frame()
        bad_camera.disconnect()
    bad_camera.disconnect()

    with pytest.raises(Exception, match="camera did not yield an image"):
        bad_camera = Camera(-1)
        bad_camera.connect()
        bad_camera.grab_frame()
        bad_camera.disconnect()
    bad_camera.disconnect()


# For type checking purposes, some object | None type variables have been put in an if not None call else raise
# TypeError structure. These errors should never get raised by user but are tested here to get code coverage.
def test_type_errors(video_system):
    video_system.start(terminator_array_name="terminator_array10", display_video=False)

    temp = video_system._terminator_array
    video_system._terminator_array = None

    with pytest.raises(TypeError):
        video_system.stop_image_production()

    with pytest.raises(TypeError):
        video_system._stop_image_saving()

    with pytest.raises(TypeError):
        video_system.stop()

    with pytest.raises(TypeError):
        video_system._on_press("e")  # This is an uncaught key

    video_system._terminator_array = temp
    video_system.stop()

    video_system.start(terminator_array_name="terminator_array11", display_video=False)

    temp = video_system._mpManager
    video_system._mpManager = None

    with pytest.raises(TypeError):
        video_system.stop()

    video_system._mpManager = temp

    video_system.stop()

    video_system.start(terminator_array_name="terminator_array16", display_video=False)

    temp = video_system._producer_process
    video_system._producer_process = None

    with pytest.raises(TypeError):
        video_system.stop()

    video_system._producer_process = temp
    video_system.stop()

    # To cover the MPQueue class. This class is never instantiated by video_system, it is purely for typing.
    q = MPQueue()
    q.put(1)
    q.put(1)
    q.get()
    q.get_nowait()
    q.empty()
    # q.qsize() Don't want to test this function because it doesn't work on mac: instead this function is given the no cover tag
    q.cancel_join_thread()

    assert True


def test_creation_and_start_errors(video_system, mock_camera, temp_directory):
    with pytest.raises(ValueError):
        VideoSystem(temp_directory, mock_camera, save_format="bad_format")

    with pytest.raises(ValueError):
        video_system.start(terminator_array_name="terminator_array12", save_format="bad_format", display_video=False)

    with pytest.raises(ValueError):
        VideoSystem(temp_directory, mock_camera, tiff_compression_level=50)

    with pytest.raises(ValueError):
        video_system.start(terminator_array_name="terminator_array13", tiff_compression_level=50, display_video=False)

    with pytest.raises(ValueError):
        VideoSystem(temp_directory, mock_camera, jpeg_quality=101)

    with pytest.raises(ValueError):
        video_system.start(terminator_array_name="terminator_array14", jpeg_quality=101, display_video=False)

    with pytest.raises(ProcessError):
        VideoSystem(temp_directory, mock_camera, num_processes=1000)

    with pytest.raises(ProcessError):
        video_system.start(terminator_array_name="terminator_array15", num_processes=1000, display_video=False)
