import cv2


class Camera:
    """A wrapper clase for an opencv VideoCapture object"""

    def __init__(self):
        self.vid = None

    def __del__(self):
        self.disconnect()

    def connect(self):
        self.vid = cv2.VideoCapture(0)

    def disconnect(self):
        if self.vid:
            self.vid.release()
            self.vid = None

    @property
    def isConnected(self):
        return self.vid == None

    def grab_frame(self):
        if self.vid:
            ret, frame = self.vid.read()
            print(ret)
            return frame
        else:
            print("Exception raised")  # to be deleted
            raise Exception("camera not connected")
