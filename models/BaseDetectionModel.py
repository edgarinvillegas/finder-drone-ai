# Placeholder class, TODO: still need implementations!
from abc import ABCMeta, abstractmethod

class BaseDetectionModel(metaclass=ABCMeta):
    #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    def __init__(self, confidence, threshold = 0):
        ...

    def getFrameProcessTime(self):
        ...

    def detect(self, frame):
        ...

    # TODO: move this to another class
    def drawDetections(self, frame, detections):
        ...






