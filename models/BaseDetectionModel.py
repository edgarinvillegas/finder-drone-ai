# Placeholder class, TODO: still needs implementations!
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np

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
        return -1

    # @must_override
    def detect(self, frame):
        ...

    # @overridable
    # TODO: Consider moving this to another class
    def drawDetections(self, frame, detections):
        LABELS = self.LABELS
        COLORS = self.COLORS
        for detection in detections:
            box = detection['box']
            classID = detection["classID"]
            confidence = detection["confidence"]
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            # display the prediction
            label = "{}: {:.2f}%".format(LABELS[classID], confidence * 100)
            print("[INFO] {}".format(label))

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[detection["classID"]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame






