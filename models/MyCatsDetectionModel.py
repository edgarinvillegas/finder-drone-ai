# import necessary libraries
import numpy as np
from . import CatDetectionModel, CustomClassifier
import cv2

class MyCatsDetectionModel(CatDetectionModel):  #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    # Confidence is called thershold in FasterRCNN
    def __init__(self, confidence = 0.5, threshold = 0.1):
        super().__init__(confidence, threshold)
        self.LABELS = ['juana', 'lily', 'whisky', 'yayo']
        self.COLORS = [(163, 237, 73), (255, 255, 255), (130, 130, 130), (0, 0, 0) ]
        #self.classifier = CustomClassifier('train_results/mycats_model_3.pt')
        self.classifier = CustomClassifier('train4_results/mycats_model_7.pt')
        #self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))

    def detect(self, frame):
        orig_detections = super().detect(frame)
        detections = []
        for orig_detection in orig_detections:
            (x, y, w, h) = orig_detection['box']
            subframe = frame[y:y+h, x:x+w]
            (catId, score, ps) = self.classifier.predict(subframe)
            print('Probabilities: ', ps)
            detections.append({
                'box': orig_detection['box'],
                'confidence': float(score),
                'classID': catId
            })
        return detections
