import cv2
import numpy as np
import time
from . import BaseDetectionModel

class SsdDetectionModel(BaseDetectionModel):
    #LABELS
    #COLORS
    #net
    #confidence
    #frameSize
    def __init__(self, confidence):
        super().__init__(confidence)
        self.confidence = confidence
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.LABELS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        #TODO: use OS separators
        modelPath = "models/ssd"
        self.net = cv2.dnn.readNetFromCaffe(modelPath + "/MobileNetSSD_deploy.prototxt.txt", modelPath + "/MobileNetSSD_deploy.caffemodel")

    def getFrameProcessTime(self):
        (start, end) = self._frameProcessTime
        return end - start

    def detect(self, image):
        #image = imutils.resize(image, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        ##
        # pass the blob through the network and obtain the detections and predictions
        start = time.time()
        self.net.setInput(blob)
        raw_detections = self.net.forward()
        end = time.time()
        self._frameProcessTime = (start, end)

        # loop over the detections
        # pass the blob through the network and obtain the detections and predictions
        # print("[INFO] computing object detections...")

        # loop over the detections
        detections = []

        for i in np.arange(0, raw_detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = raw_detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for the object
                idx = int(raw_detections[0, 0, i, 1])
                box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                detections.append({
                    'box': [startX, startY, int(endX - startX), int(endY - startY)],
                    'confidence': float(confidence),
                    'classID': idx
                })
        return detections





