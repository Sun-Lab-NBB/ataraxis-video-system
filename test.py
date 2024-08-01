# import cv2
# from harvesters.core import Harvester
# from pathlib import Path
# import time as tm
# from queue import Queue
# from threading import Thread
# import copy
#
#
# class CameraInterface:
#     def __init__(self, name):
#         self.name = name
#         self.harvester = Harvester()
#         self.camera = None
#         self.display_queue = Queue()
#         self.display_thread = None
#
#     def image_display_module(self):
#         while True:
#             frame_number, cam_frame = self.display_queue.get()
#
#             if frame_number < 0:
#                 break
#
#             cv2.imshow(f"{self.name}_grabbed_frames", cam_frame)
#
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#
#         cv2.destroyAllWindows()
#
#     def initialize_camera(self, cti_path):
#         try:
#             self.harvester.add_cti_file(str(cti_path))
#         except (FileNotFoundError, OSError):
#             print(f"Failed to add CTI file: {cti_path}")
#             return False
#
#         self.harvester.update_device_info_list()
#
#         if not self.harvester.device_info_list:
#             print("No devices found.")
#             return False
#
#         self.camera = self.harvester.create_image_acquirer(0)
#         return True
#
#     def start_acquisition(self, duration=30):
#         if not self.camera:
#             print("Camera not initialized.")
#             return
#
#         self.display_thread = Thread(target=self.image_display_module)
#         self.display_thread.start()
#
#         self.camera.start_image_acquisition()
#
#         start = tm.time()
#         num = 0
#
#         try:
#             while tm.time() - start < duration:
#                 buffer = self.camera.fetch_buffer()
#                 component = buffer.payload.components[0]
#                 cam_frame = copy.copy(component.data.reshape(2840, 2840))
#                 num += 1
#                 self.display_queue.put((num, cam_frame))
#                 buffer.queue()
#         except Exception as e:
#             print(f"Error during acquisition: {e}")
#         finally:
#             self.camera.stop_image_acquisition()
#             self.display_queue.put((-1, None))  # Signal to stop display thread
#             self.display_thread.join()
#
#     def cleanup(self):
#         if self.camera:
#             self.camera.destroy()
#         self.harvester.reset()
#
#
# # Usage
# if __name__ == "__main__":
#     cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
#
#     cam_interface = CameraInterface("TestCamera")
#     if cam_interface.initialize_camera(cti_path):
#         cam_interface.start_acquisition(duration=600)
#         cam_interface.cleanup()
#     else:
#         print("Failed to initialize camera.")

from src.ataraxis_video_system.camera import MockCamera, HarvestersCamera, OpenCVCamera
from pathlib import Path
import sys
import time as tm
from src.ataraxis_video_system.saver import VideoSaver

print(VideoSaver.supported_video_formats)

sys.exit()

cti_path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")

# camera = HarvestersCamera(name='harvey', cti_path=cti_path, fps=30, height=1024, width=1280)
camera = MockCamera(name="mock", fps=100, height=1024, width=1200)
camera.connect()

start = tm.time()

print(camera.fps)
print(camera.width)
print(camera.height)

frames = 0
while tm.time() - start < 20:
    frame = camera.grab_frame()
    frames += 1

camera.disconnect()
print(frames)
