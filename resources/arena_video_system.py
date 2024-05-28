# arena_video_system library contains ArenaVideoSystem class, related AVSDispatcher class and some standalone functions related to avs operation and data processing
# These classes and functions are used to instantiate and interface with Arena Video Systems used to record the experimental sessions and to optimize raw video data processing
# Author: Ivan Kondratyev (ik278@cornell.edu)

import copy
import os
import shutil as sh
import sys
import time as tm
import multiprocessing as mp
from multiprocessing import shared_memory as sm
import numpy as np
import pickle as pic
import pandas as pd
from harvesters.core import Harvester as hrv
import cv2
import tifffile as tff
import queue
import threading
import subprocess
from PIL import Image
import inspect
import ataraxis_utils as axu


# This class functions as a wrapper for specific video system. Like AMC wrappers, it is a self-contained class that is instantiated for each independent video system
# It contains the functions necessary to instantiate, maintain and terminate shared-memory-based processes to enable simultaneous functioning of the PC-side components of each video system and connect to globally-instantiated sm objects
# It also contains the necessary communication, logic and data processing protocols, designed to interface with any Video system to acquire and save data as images.
# To convert images into videos, an independent, offline ffmpeg-based process is used to enable unhindered acquisition. Tested with 1 9MP camera acquiring at 87.7 fps for 25-30 minutes per session
class ArenaVideoSystem:

    # Initializes the class with user-defined parameters
    def __init__(self, name, backend, recording_path, recording_size, display_size, target_fps, global_arrays, parallel_execution=False, num_buffers=60, num_writers=5, device_key=None, async_grabber=True, debug=0, verbose=1, log_path=None):

        # General Variables
        self.name = name  # This is the user-defined name of the particular video-system. This is necessary for future system identification purposes and meaningful error callbacks
        self.fps = target_fps  # This is the fps parameter the acquisition is set to (Hz frequency of image acquisition)
        self.recording_size = recording_size  # The size of the acquired images. Harvesters acquires them as a 1-D numpy array which has to be reshaped into 2D format
        self.display_size = display_size  # The size the images are converted to before they are displayed. Allows to scale images to desired monitor-fitting size. Note, by default scaling method used is Area Interpolation, change in code if you prefer another one
        self.recording_path = f"{recording_path}/{self.name}"  # The path to the output folder. Inside the output folder, generates controller-named folder
        self.raw_path = f"{self.recording_path}/Raw"  # Generates the path to raw folder used by image saving functions
        self.device_key = device_key  # If using multiple cameras, it is imperative to pass the correct device key for the system to grab an appropriate camera. Grabbing is exclusive, so at most a single grabber cna be attached to every system (This is configurable, but you will need to do this yourself)
        self.parallel_execution = parallel_execution  # Determines whether thread-based concurrent or multiprocessing-based parallel processing is used for frame saving. Note, for most application concurrent saving is adequate, only use parallel if your pipeline requires extreme performance

        # Information FLow Variables. They control whether various messages (and which messages) are written to terminal and into a separate runtime log file
        self.debug = debug
        self.verbose = verbose
        self.log_path = log_path

        # Shared Memory Arrays. Only used during video grabbing process
        self.exp_array = global_arrays[0]  # This is the general experiment status array used throughout the project to control quit states
        self.ttl_array = global_arrays[1]  # This array is used to communicate the state of incoming frames (when they are received) to an amc process that bounces these ttl signals to the recording system
        self.save_frame_array = global_arrays[2]  # This array is used to control when the grabbed frames should be written to disk vs simply displayed to the user

        # Grabber Parameters
        self.grabber_backend = backend  # The backend contains the path to GenTL producer backend file. This enables interfacing with the camera using GenTl protocols
        self.num_buffers = num_buffers  # The number of full buffers (images) kept by harvesters backend on its end. Can be incompatible with some producer backends. Only rely on this as an emergency measure, ideally you want your pipeline to have 1:1 or faster image processing to camera acquisition ratio
        self.num_writers = num_writers  # The number of threads that write (save) the images to disk
        self.async_grabber = async_grabber  # Specifies whether to use a separate grabber thread that runs in parallel and acquires incoming camera images or whether to use a blocking dynamic API call

        # Config. Stores the value of key variables for reference.
        self.info_config = {
            'name': self.name,
            'device_key': self.device_key,
            'recording_size': self.recording_size,
            'fps': self.fps,
            'output_path': self.recording_path,
            'grabber_backend': self.grabber_backend,
            'number_of_async_buffers': self.num_buffers,
            'number_of_writer_threads': self.num_writers,
            'async_grabber': self.async_grabber,
        }  # Packages class data into a config file that is saved inside acquisition folder as reference. Only stores acquisition data

    # This is the main function of every AVS. It grabs the data from GenTL backend and pipes it to I/O writer threads. The threads then write acquired frames as uncompressed .tiff files for permanent storage
    # All thread management is carried out by the grabber module. Note, if a thread is blocked in a non-GIL-releasing fashion, this halts all submodules including this function
    def data_grabber_module(self):
        # Initializes harvesters wrapper
        h = hrv()  # Instantiates harvester object
        try:
            h.add_file(file_path=self.grabber_backend, check_existence=True, check_validity=True)  # Imports general interface cti file to enable harvester to find and work with the camera
        except FileNotFoundError:
            sys.exit(f'Warning! Error encountered in {sys.argv[0]} script at line {get_caller_line_number()} for {self.name} video system. GenTL Backend file {self.grabber_backend} does not exist. Execution aborted!')
        except OSError:
            sys.exit(f'Warning! Error encountered in {sys.argv[0]} script at line {get_caller_line_number()} for {self.name} video system. GenTL Backend file {self.grabber_backend} is not a valid GenTL backend. Execution aborted!')

        h.update()  # Updates the device list to discover cameras to grab data from

        # Creates an image acquisition object. Note, you can give it specific keys to grab the precise device you want. In case only a single camera is used, pass None as a device key to automatically select it
        if self.device_key:
            grabber = h.create(self.device_key)
        else:
            grabber = h.create()

        grabber.num_buffers = self.num_buffers  # Specifies the number of buffers to be used by the system. Only used when async_grabber is enabled. If the code runs without issue, you may not need many of these, they are mostly helpful if the processing halts for a while. In this case, the buffers do just what their name implies and prevent data loss

        # Connects to shared memory arrays
        shm_1, save_frame_arr = axu.connect_shm_arr(self.save_frame_array)  # This array is used to control when the grabbed frames should be written to disk vs simply displayed to the user
        shm_2, frame_ttl = axu.connect_shm_arr(self.ttl_array)  # This array is used to communicate when a frame has been acquired. When that happens, a dedicated amc sends a ttl pulse over to recording system to synchronize all ttl signals to the same clock
        shm_3, exp_state = axu.connect_shm_arr(self.exp_array)  # Experimental State (stage). Used to communicate the current phase of the experiment, most importantly phase 0 (shutdown)

        writer_queue = queue.Queue()  # Create a queue to pass frames to writer(s)
        display_queue = queue.Queue()  # Also create a queue to pass frames to display thread

        # Create and start writer thread(s). Each of these threads loops over save_image module code
        threads = []
        for _ in range(self.num_writers):
            t = threading.Thread(target=self.image_recording_module, args=(writer_queue,))
            t.start()
            threads.append(t)

        # Also instantiates a separate thread to display the images to the user in a video-fashion
        display_thread = threading.Thread(target=self.image_display_module, args=(display_queue,))
        display_thread.start()
        threads.append(display_thread)

        display_limiter = 0  # This is used to limit the display of certain messages
        count = 0  # Counts processed frames
        save_frames = 0  # Controls whether to save or only display acquired frames
        output_array = []  # This tracks frames and their acquisition timestamps

        # Generates a raw folder inside the output folder to store the incoming images
        # If the folder is not present, creates the folder
        os.makedirs(self.raw_path, exist_ok=True)

        device_stopped = 0  # Tracks whether image acquisition is active

        # Begins image acquisition
        # Note, make sure your GenTL is configured to software trigger beforehand. I may add this configuration later as an explicit line here
        grabber.remote_device.node_map.AcquisitionFrameRate.value = self.fps  # Uses GenTL map to pass acquisition fps value to the camera. NOTE, make sure your camera supports the requested fps
        grabber.remote_device.node_map.AcquisitionStart.execute()  # Starts image acquisition. For this to work, the camera is expected to be set to software trigger and continuous acquisition. This sends the software trigger

        if self.async_grabber:
            grabber.start(run_as_thread=True)  # Starts grabbing incoming camera frames. Uses a separate thread to do so for improved concurrency
        else:
            grabber.start(run_as_thread=False)  # Starts grabbing incoming camera frames. Does not use a separate thread. For this to work properly, your image processing should take equal or less time than your acquisition cycle does

        if (self.debug == 1) | (self.verbose == 1):
            axu.print_string(self.log_path, f'{self.name} video grabber module: initialized', 0)

        # Enters infinite loop. Note, given how this loop functions and the use of multiprocessing, this has the ability to hog the core it is assigned to for 100% of the time. To ameliorate this, you can add delays to the loop, but in my case I can sacrifice a few cores, so it is left in a less-optimized state
        # Note, this loop is designed to run until ALL filled buffers are consumed. This may result in somewhat prolonged activation compared to other libraries used in each experiment
        while (exp_state[0] != 255) | (grabber.num_holding_filled_buffers > 0):

            # Evaluates whether to save acquired frames. This code allows to use the same general pipeline to either continuously display or both display and save camera frames at user-defined timeframes
            # If frame_saving trigger is set to 1 and save_frames is set to 0, enables frame saving
            if (save_frame_arr[0] == 1) & (save_frames == 0):
                save_frames = 1
                if display_limiter != 1:
                    if (self.debug == 1) | (self.verbose == 1):
                        axu.print_string(log_path=self.log_path, string=f'{self.name} frame saving: initialized', pad=0)
                    display_limiter = 1

            # Alternatively, if frame_saving trigger is set to 0 and save_frames is enabled, disables frame saving
            elif (save_frame_arr[0] == 0) & (save_frames == 1):
                save_frames = 0
                if display_limiter != 2:
                    if (self.debug == 1) | (self.verbose == 1):
                        axu.print_string(log_path=self.log_path, string=f'{self.name} frame saving: stopped', pad=0)
                    display_limiter = 2

            # Acquires camera frame
            buffer = grabber.fetch()  # Fetches the buffer
            payload = buffer.payload  # Obtains recorded buffer payload
            component = payload.components[0]  # Retrieves the frame from payload
            timestamp = buffer.timestamp_ns  # Also retrieves GenTL-assigned nanosecond-based timestamp

            cam_frame = copy.copy(component.data.reshape(self.recording_size))  # Reshapes grabbed 1D array into 2D frame array. Also uses full copy operation to retrieve it before the buffer gets invalidated (and content-wiped) by re-queueing

            buffer.queue()  # Queues the buffer to be filled again. At this point the frame is the only container of the grabbed image data, the component container is destroyed by re-queueing the buffer

            # If saving is enabled, triggers frame ttl pulse and sends the frame to the writer threads to be saved as a lossless tif file
            if save_frames == 1:
                frame_ttl[0] = 1  # Sets ttl value to 1, triggering acquisition ttl pulse via a specialized amc process. That pulse is used to synchronize video data with photometry data
                count = count + 1  # Counts saved frames
                writer_queue.put((count, cam_frame))  # Puts the frame inside writer queue for saving
                output_array.append([count, timestamp])  # Appends frame data to storage array. Saves the image number and acquisition timestamp

            # Note, display is perpetually active to enable saving-independent monitoring of video system
            cam_frame = cv2.resize(cam_frame, self.display_size, interpolation=cv2.INTER_AREA)  # Resizes the frame to actually fit the screen when displayed with opencv

            display_queue.put((count, cam_frame))  # Puts the resized frame inside display queue

        # If state 255 is passed, the acquisition can be locked-in by the continued presence of unprocessed images. This code shuts down the acquisition, but allows the loop to run until all filled buffers are processed. This ensures there is no visual data loss
        if exp_state[0] == 255:
            grabber.remote_device.node_map.AcquisitionStop.execute()  # Stops image acquisition
            device_stopped = 1

        # This code is executed only once the while loop is left, which can only happen when

        # Generally it should never be the case that while loop is left without stopping the acquisition, but no such thing as redundant checks, so... This check ensures the camera is stopped if that happens
        if device_stopped != 1:
            grabber.remote_device.node_map.AcquisitionStop.execute()  # Stops image acquisition
        grabber.stop()  # Stops grabbing frames
        grabber.destroy()  # Destroys the grabber, releasing the device and all reserved resources

        # Using -1 for the first argument causes worker threads to break and shutdown, so sends shutdown signal to each thread
        for _ in range(self.num_writers):
            writer_queue.put((-1, -1))
        display_queue.put((-1, -1))

        # Wait for writer thread(s) and display thread to exit
        for thread in threads:
            thread.join()

        # Closes shared memory objects
        shm_1.close()
        shm_2.close()
        shm_3.close()

        return output_array  # Returns frame number and timestamp array to be saved to the output folder

    # This function obtains a frame from writer queue and saves it using predefined extension and parameters into class-specific output folder
    # Note, you can adjust the saving parameters as desired, currently configured to write uncompressed tiff files
    def image_recording_module(self, img_queue):
        # Executes until encounters a shutdown signal passed through queue
        while True:
            (frame_number, cam_frame) = img_queue.get()  # Obtains the frame object (numpy array) and it's number

            # If passed frame number is less than zero, which only happens when shutdown -1, -1 argument is passed, breaks the loop leading to thread shutdown
            if frame_number < 0:
                break

            im_path = f"{self.raw_path}{os.path.sep}{str(frame_number).rjust(8, '0')}.tif"  # Presets image filename based on preset image number

            try:
                tff.imwrite(im_path, data=cam_frame)  # Uses default zlib and compression level 0 (uncompressed) setting of tifffile library to save incoming frames as .tif files
            except TypeError:
                pass
            except PermissionError:
                pass
            except RuntimeError:
                pass

    # This function uses opencv (cv2) library to display a preprocessed frame to the user. Allows to keep track of what is being acquired (visually) and monitor the experiment using the camera system
    def image_display_module(self, img_queue):
        # Executes until encounters a shutdown signal passed through queue
        while True:
            (frame_number, cam_frame) = img_queue.get()  # Grabs the image from recorder queue

            if frame_number < 0:  # Breaks, if exit code is passed as an argument
                break

            cv2.imshow(f'{self.name}_grabbed_frames', cam_frame)  # Shows frame to user using opencv

            # This is a necessary part that enables continued display of images by the opencv module
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()  # Closes cv2 display window if the loop is escaped

    # A simple error callback that stops processing and displays the error message. Fairly uninformative, but at least indicates when something goes wrong
    @staticmethod
    def grabber_error_callback(error):
        print(f'Warning! Error encountered during AVS Video Grabber asynchronous module execution. Error: {error}')
        sys.exit('System: stopped')

    # This function packages a grabber config into a pickle file and saves it into the core recording folder (so both are saved outside the Raw folder that stores raw tiff frames)
    def save_config(self):
        config_path = f"{self.recording_path}/frame_grabber_config.pickle"
        with open(config_path, 'ab') as fl:
            pic.dump(self.info_config, protocol=pic.HIGHEST_PROTOCOL, file=fl)  # Serializes and saves the configs as pickle file

    # This function packages and saves the frame_list array returned by data_grabber_module. The array is saved as a hdf5 file
    def save_frame_list(self, frame_list):

        # First, decomposes the array into frames and respective timestamps
        frames = []
        timestamps = []
        for cam_frame in frame_list:
            frames.append(cam_frame[0])
            timestamps.append(cam_frame[1])

        # Uses arrays generated above to make a pandas dictionary
        pd_dict = {'frame': frames, 'timestamp': timestamps}  # Generates a pandas dictionary

        df = pd.DataFrame(data=pd_dict)  # Generates pandas dataframe from dictionary

        frame_list_path = f"{self.recording_path}/saved_frame_list.h5"  # Precreates the hdf5 file path

        df.to_hdf(frame_list_path, key='frame_data')  # Creates and saves the hdf 5 file

    # This is the function that is intended to be passed to a parallel process. It triggers parallel execution and post-processing of all AVS operations
    @staticmethod
    def initialize_avs(avs, debug=0, verbose=1, log_path=None):

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path=log_path, string=f'AVS {avs.name} grabber process: initialized', pad=1)

        # Note, upon completion returns a list that has frame numbers and grabbed timeframes. The timeframes are generated by the camera here, so they can be used to correct the ttl signal deviation, which is sent with respect to PC acquiring the frame
        frame_array = avs.data_grabber_module()  # Triggers image acquisition. This returns a list of frames with their timestamps

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path=log_path, string=f'AVS {avs.name} grabber module shutdown: complete. Exporting frame_list and configs...', pad=0)

        # Saves the grabber config information and the acquired frame list to the output folder
        avs.save_config()  # This saves the grabber config. avs_mode == Encoder saves encoder config
        avs.save_frame_list(frame_list=frame_array)

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path=log_path, string=f'AVS {avs.name} asynchronous execution: complete', pad=0)

        return frame_array  # Returns the frame_array to enable compatibility with standalone testing

    # This module is used to convert pre-acquired stream of raw tiff images into a compressed (and lossy) video file using ffmpeg. NOTE, you need to install ffmpeg separately
    # Deletes the raw tiff folder upon completion of the process to conserve space if this option is enabled by the user
    @staticmethod
    def video_converter_module(recorder_name, converter_params, session_data, description_data, remove_raw=False, debug=0, verbose=1, log_path=None):

        # Unpacks session data
        input_path = session_data[0]
        img_extension = session_data[1]

        # If recorder name is None, obtains recorder name from the folder tree structure. Specifically, sets it to the root folder inside which the Raw or Compressed folder are located
        if not recorder_name:
            recorder_name = input_path.split('\\')[-2]

        # Obtains the list of frame images from the Raw folder
        input_images = os.listdir(input_path)
        frame_list = []

        input_images = sorted(input_images)  # Ensures the frames are sorted in the ascending order

        # Loops over the directory and adds all files ending with correct extension to the processing list. Should really be the only contents of the folder, but checks are useful and all...
        for img in input_images:
            if img.endswith(img_extension):
                frame_list.append(f"{input_path}/{img}")

        # Generates a temporary file to serve as the image roster fed into ffmpeg
        file_list_path = f"{input_path}/file_list.txt"  # The file is saved locally, to the folder being encoded
        with open(file_list_path, "w") as fl:
            for input_frame in frame_list:
                fl.write(f"file 'file:{input_frame}'\n")  # NOTE!!! It is MANDATORY to include file: when the file_list.txt itself is located inside root source folder and each image paths is given as absolute path, as otherwise ffmpeg appends the root path to text file in addition to each image path, resulting in incompatible path
                # Also, quotation (single) marks are needed to ensure ffmpeg correctly processes special characters and spaces
                # Finally, the slash marks should be the forward slash (from my online research at least), which is conveniently handled by path-rectifier code in the main function structure

        encoding_runs = len(converter_params)  # Obtains the total dictionary size

        # Loops over each set of encoding parameters and encodes the videos
        for i, video_id in enumerate(converter_params.keys()):

            encoding_params = converter_params[video_id]  # Uses each ID (key) to retrieve processing parameters

            # Sets the output video-file name based on -crf, -preset and dictionary id
            video_file = f'{os.path.dirname(input_path)}/{recorder_name}_{encoding_params["crf"]}_{encoding_params["preset"]}_{video_id}.mp4'

            # If the video file does not exist, generates it. If video file exists, skips processing
            if not os.path.exists(video_file):

                if (verbose == 1) | (debug == 1):
                    axu.print_string(log_path, f'Encoding video {i + 1} out of {encoding_runs} requested for session {description_data[0]} out of {description_data[1]} total', 0)

                # Construct ffmpeg command
                # Note, this command is designed to generate monochrome videos. For color, edit -vf format attribute
                command = f'ffmpeg -f concat -safe 0 -r {encoding_params["fps"]} -i {file_list_path} -vcodec {encoding_params["codec"]} -crf {encoding_params["crf"]} -preset {encoding_params["preset"]} -vf {encoding_params["vf"]} -f mp4 {video_file} -thread_queue_size 128 -loglevel error -progress pipe:1'

                # Execute ffmpeg command using subprocess module
                process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, shell=True)  # Note, the current implementation prints standard ffmpeg progress data to the terminal window which enables crude progress tracking

                # Wait for the process to complete and properly close it
                stdout, stderr = process.communicate()

                if (verbose == 1) | (debug == 1):
                    axu.print_string(log_path, f'Video {i + 1} out of {encoding_runs} requested for session {description_data[0]}: Encoded. FFMPEG output ticket:', 1)

                # Print stdout and stderr if needed
                axu.print_string(log_path, f"STDERR: {stderr}", 0)

        # Delete temporary file list
        os.remove(file_list_path)

        # Clears raw files if enabled by the user. Default is False
        if remove_raw:
            sh.rmtree(input_path)  # Deletes raw tiff files once they have been written as a video file

    # This function is used to compress lossless images produced by image_recording_module for improved transportability
    # Used to assist in transferring captured folders to processing Workstation for subsequent HEVC encoding, probably not helpful for general users
    @staticmethod
    def compress_images(input_directory, compression, remove_raw=True, debug=0, verbose=1, log_path=None):
        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'Image compression module: initialized', 0)

        # These arrays count processed images and failed (error-triggering) processing attempts
        compressed_arr = axu.create_sm_array([0], 'int64', 'compressed_array')
        failed_arr = axu.create_sm_array([0], 'int64', 'failed_arr')

        shm_1 = sm.SharedMemory(name=compressed_arr[0])
        processed_files = np.ndarray(shape=compressed_arr[1], dtype=compressed_arr[2], buffer=shm_1.buf)

        shm_2 = sm.SharedMemory(name=failed_arr[0])
        failed_files = np.ndarray(shape=failed_arr[1], dtype=failed_arr[2], buffer=shm_2.buf)

        # Sets input and output directories
        output_directory = f"{input_directory}/Compressed"
        output_file = f"{input_directory}/compressed_frame_list.h5"  # Also generates a file path for frame processing data file
        input_directory = f"{input_directory}/Raw"

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'Shared memory array setup: complete', 0)

        # Ensures output directory exists. If it does, there is no error (that is why exist_ok is used)
        os.makedirs(output_directory, exist_ok=True)

        # Generates image paths
        directory_files = os.listdir(input_directory)

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'Generating image list...', 0)

        image_list = []  # Presets image list variable
        for image_file in directory_files:
            if image_file.endswith('tiff') | image_file.endswith('tif'):
                image_list.append(f"{input_directory}/{image_file}")  # Appends each tiff image to processing list

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'Image list generation: complete', 0)
            axu.print_string(log_path, 'Generating mp pool argument list...', 0)

        input_args = []  # A mega-list where input arguments for the compression module are packaged as lists (so a list of lists)
        for image in image_list:
            # For each image, sets the output path
            image_string = image.split('/')  # Splits the directory
            image_string = image_string[-1]  # Keeps image name and extension only

            # Removes the extension
            image_split = image_string.split('.')  # Splits around the comma. Should result in the whole part + extension. Selecting index 0 in the resultant array returns the whole path part without extension

            image_string = f"{image_split[0]}.png"  # Adds .png extension

            # Generates image output path
            output_image = f"{output_directory}/{image_string}"  # In the end, this is the path to the image with the same name as the original image, but inside Compressed folder and with png extension

            # Builds the argument list for each image and appends it to arg list
            input_args.append([image, output_image, compression, compressed_arr, failed_arr])  # path to input, path to output, desired compression level, compressed and failed shm arrays

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'MP pool argument list: generated', 0)
            axu.print_string(log_path, 'Instantiating MP pool...', 0)

        processing_pool = mp.Pool(axu.get_max_mp_processes())  # Sets it to num logical cpus, so physical * 2 unless you are on the intel train

        # Runs the mp pool
        start_time = tm.monotonic_ns()
        compression_result = processing_pool.starmap_async(ArenaVideoSystem.image_compression_module, input_args)

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, 'Pool: initialized. Conversion in progress...', 1)

        # Waits for the compressio to complete. Prints status images
        elapsed_prev = 0
        while not compression_result.ready():
            complete_percent = round(((processed_files[0] / len(image_list)) * 100), 0)
            elapsed_current = round((((tm.monotonic_ns() - start_time) / 1000000000) / 60), 0)  # Converts to minutes
            if elapsed_current > elapsed_prev + 1:  # Sets to display progress every minute
                if (verbose == 1) | (debug == 1):
                    axu.print_string(log_path, f'Compressing... {complete_percent} % complete. {processed_files[0]} files out of {len(image_list)} processed. {failed_files[0]} file compressions failed. Elapsed time: {elapsed_current} minutes', 0)
                elapsed_prev = elapsed_current

        end_time = tm.monotonic_ns()

        delta_t = (end_time - start_time) / 1000000000  # Converts delta to seconds

        if (verbose == 1) | (debug == 1):
            axu.print_string(log_path, f"Compression: complete. Compressed {len(image_list)} files, which took {round(delta_t / 60, 2)} minutes", 1)

        # Closes mp pool
        processing_pool.close()
        processing_pool.join()

        result = compression_result.get()  # Retrieves image processing data

        # Decomposes result into image name and status arrays
        image_names = []
        image_statuses = []
        for image_data in result:
            image_names.append(image_data[0])
            image_statuses.append(image_data[1])

        # If some compressions failed, determines which files failed and tells the user
        if failed_files[0] > 0:

            # Loops over image_statuses and if any compression status is 0 (failure), appends corresponding image name to a failure sub-array
            failed_list = []
            for num, image_status in enumerate(image_statuses):
                if image_status == 0:
                    failed_list.append(image_names[num])

            # Prints failed files
            if (verbose == 1) | (debug == 1):
                axu.print_string(log_path, f"However, {failed_arr[0]} files failed to compress. The files were:", 0)
            for failure in failed_list:
                print(failure)
        else:  # Note, only removes raw files if compression went without failure
            # If raw file removal is enabled, deletes the raw folder, leaving only the compressed folder behind
            if remove_raw:
                sh.rmtree(input_directory)
                if (verbose == 1) | (debug == 1):
                    axu.print_string(log_path, f"Raw folder: removed", 0)

        # Packages the image processing data files into a h5 file using pandas dataframe tools
        output_dict = {'image': image_names, 'status': image_statuses}
        df = pd.DataFrame(data=output_dict)
        df.to_hdf(path_or_buf=output_file, key='processed_frame_data')

    # This module is used to compress the raw images for transportation in raw format. I developed this simple script to transfer files from recording PC to my workstation which has a maxed TR to encode video at better preset and lower crf to improve the quality and processing times
    # The code replaces the Raw folder with Compressed folder. The encoder has been updated to work with compressed folder
    # This code is designed for massively parallel execution
    @staticmethod
    def image_compression_module(image_file, output_path, compression, processed_array, failed_array):
        # Connects to shared memory arrays
        shm_1 = sm.SharedMemory(name=processed_array[0])
        processed_files = np.ndarray(shape=processed_array[1], dtype=processed_array[2], buffer=shm_1.buf)

        shm_2 = sm.SharedMemory(name=failed_array[0])
        failed_files = np.ndarray(shape=failed_array[1], dtype=failed_array[2], buffer=shm_2.buf)

        image_name = image_file.split('/')[-1]  # Extracts image name out of the input path

        try:
            image = Image.open(image_file)  # Opens the image to be converted
            image.save(output_path, "PNG", compress_level=compression)  # max compression is 9
            processed_files[0] = processed_files[0] + 1  # Increments succeeded counter by 1
            image_status = 1  # 1 is processed
        except OSError:
            failed_files[0] = failed_files[0] + 1  # Increments failed counter by 1
            image_status = 0  # 0 is failed

        return [image_name, image_status]  # returns the image name and its processing status
