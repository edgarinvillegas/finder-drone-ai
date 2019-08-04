
import numpy as np
from . import BaseDetectionModel
from . import CustomClassifier
import time
import matplotlib.pyplot as plt

class CustomDetectionModel(BaseDetectionModel):    #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    def __init__(self, confidence):
        super().__init__(confidence)
        self.confidence = confidence
        self.LABELS = ['bear', 'chimp', 'giraffe', 'gorilla', 'llama', 'ostrich', 'porcupine',
                        'skunk', 'triceratops', 'zebra']
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        self.net = CustomClassifier()

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        (prediction, score) = self.net.predict(frame)
        print('prediction: ', prediction)
        startX, startY, endX, endY = (10, 10, w-20, h-20)
        best_detection = {
            'box': [startX, startY, int(endX - startX), int(endY - startY)],
            'confidence': float(score),
            'classID': prediction
        }
        detections = [best_detection]
        return detections
