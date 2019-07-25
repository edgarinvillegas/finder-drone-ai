from abc import ABCMeta, abstractmethod
import cv2

class BaseVideoOutput(metaclass = ABCMeta):
    def __init__(self, source = 0):
        ...

    def setFrameSize(self, frameSize):
        self._frameSize = frameSize

    @abstractmethod
    def write(self, frame):
        ...

    def release(self):
        ...