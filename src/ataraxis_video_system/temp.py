from PIL import Image, ImageDraw, ImageChops
from shared_memory_array import SharedMemoryArray
import numpy as np


def images_are_equal(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size != image2.size or image1.mode != image2.mode:
        return False

    diff = ImageChops.difference(image1, image2)
    return not diff.getbbox()  # Returns True if images are identical

prototype = np.array([1, 0, 1], dtype=np.int32) 
ar = SharedMemoryArray.create_array("test_array3", prototype)
ar.connect()

print(ar._array)
arr = np.ndarray(shape=1, dtype=np.int32)
arr[0] = 0
ar.write_data(2, arr)

print(ar._array)

ar.disconnect()
