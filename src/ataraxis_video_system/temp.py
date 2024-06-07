from PIL import Image, ImageDraw, ImageChops
# from multiprocessing import Queue
import multiprocessing
import queue


def images_are_equal(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size != image2.size or image1.mode != image2.mode:
        return False

    diff = ImageChops.difference(image1, image2)
    return not diff.getbbox()  # Returns True if images are identical

q = multiprocessing.Queue()

try:    
    q.get_nowait()
except queue.Empty:
    print('caught')
# print(q.qsize())
# print(q.empty())

# for i in range(4):
#     q.put(1)

# for i in range(5):
#     q.get()

# print(q.qsize())
# print(q.empty())