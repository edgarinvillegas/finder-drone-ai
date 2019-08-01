import cv2
import numpy as np
import time
from . import BaseDetectionModel

class FaceDetectionModel(BaseDetectionModel):    #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    def __init__(self, confidence):
        super().__init__(confidence)
        self.confidence = confidence
        self.LABELS = ['']
        self.COLORS = [(0, 0, 255)]
        self.net = cv2.dnn.readNetFromCaffe("models/face/deploy.prototxt.txt", "models/face/res10_300x300_ssd_iter_140000.caffemodel")


    def detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        raw_detections = self.net.forward()

        detections = []

        for i in range(0, raw_detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = raw_detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            detections.append({
                'box': [startX, startY, int(endX - startX), int(endY - startY)],
                'confidence': float(confidence),
                'classID': 0
            })

        return detections
