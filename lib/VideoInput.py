import cv2
import os
import numpy as np
import imutils

class VideoInput:
    #vs
    def __init__(self, filename):
        self.filename = filename
        # initialize the video stream, pointer to output video file, and frame dimensions
        self.vs = cv2.VideoCapture(filename)
        self._total = -1

    def start(self):
        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            self._total = int(self.vs.get(prop))
            print("[INFO] {} total frames in video".format(self._total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            _total = -1
        return self

    def getNextFrame(self):
        (grabbed, frame) = self.vs.read()
        if not grabbed:
            return None
        return frame

    def getTotalFrames(self):
        return self._total

    def release(self):
        self.vs.release()




