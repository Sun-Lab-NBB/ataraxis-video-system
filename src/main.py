# import the opencv library 
from collections import deque
import sys
import cv2 
import os 
from high_precision_timer.precision_timer import PrecisionTimer
import time as tm
from multiprocessing import Process, Queue 
import numpy as np #temp
import keyboard


d = {'ns': 10 ** 9, 'us' : 10 ** 6, 'ms' : 10 ** 3, 's': 1}

unit = 'ms'

def foo():
    print("hello")


def run_time_control(func, fps, verbose = False):
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)

    frames = 0
    
    while(True):
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            func()
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print('fps:', frames)
            check_timer.reset()
            frames = 0

def get_frame(camera, img_queue):
    ret, frame = camera.read()
    img_queue.put(frame)
    print(ret)

def input_stream(data_queue, terminator_queue, fps = None, verbose = False):
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)
    frames = 0
    vid = cv2.VideoCapture(0)
    while terminator_queue.empty():
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            get_frame(vid, data_queue)
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print('fps:', frames)
            check_timer.reset()
            frames = 0
    vid.release()

def save_frame(img_queue, img_id):
    if not img_queue.empty():
        frame = img_queue.get()
        filename = 'imgs\img' + str(img_id) + '.png'
        cv2.imwrite(filename, frame)
        return True
    return False

def save_images_loop(data_queue, fps = None, verbose = False):
    num_imgs_saved = 0
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)
    frames = 0
    while True:
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            saved = save_frame(data_queue, num_imgs_saved)
            if saved:
                num_imgs_saved += 1
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print('fps:', frames)
            check_timer.reset()
            frames = 0

if __name__ == '__main__':
    
    data_queue = Queue()
    terminator_queue = Queue()


    # p1 = Process(target=run_time_control, args=(lambda: get_frame(q), 2), daemon=True) 
    # p2 = Process(target=save_frame, args=(lambda: save_frame(q), 1), daemon=True) 

    p1 = Process(target=input_stream, args=(data_queue, terminator_queue, 5,)) 
    p2 = Process(target=save_images_loop, args=(data_queue, 1)) 

    p1.start()
    p2.start()

    keyboard.wait('q')

    terminator_queue.put(1)
    p1.join()

    keyboard.wait('q')

    p2.terminate()

    cv2.destroyAllWindows()

    print("Done") 





# run_time_control(foo, 5, verbose=True)