
import numpy as np
from . import BaseDetectionModel
from . import CustomClassifier
import time
import cv2
import matplotlib.pyplot as plt

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

class SlidingWindowDetectionModel(BaseDetectionModel):    #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    def __init__(self, confidence):
        super().__init__(confidence)
        self.confidence = confidence
        # self.LABELS = ['bear', 'chimp', 'giraffe', 'gorilla', 'llama', 'ostrich', 'porcupine', 'skunk', 'triceratops', 'zebra']
        self.LABELS = ['moose', 'no-moose']
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        self.net = CustomClassifier()

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        # stepSize = int(winH / 2)
        (winW, winH) = (400, 400)
        stepSize = int(winH / 2)
        window_detections = []
        for (x, y, window) in sliding_window(frame, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            (prediction, score, ps) = self.net.predict(window)
            # print('prediction: ', prediction)
            startX, startY, endX, endY = (x, y, x + winW, y + winH)
            if(prediction == 0 and score > self.confidence):
                print('Predicted: ', prediction, 'ps: ', ps)
                window_detections.append({
                    'box': [startX, startY, int(endX - startX), int(endY - startY)],
                    'confidence': float(score),
                    'classID': prediction
                })
            if(len(window_detections) == 0):
                detections = []
            else:
                detections = [max(window_detections, key=lambda d: d['confidence'])]
        return detections
