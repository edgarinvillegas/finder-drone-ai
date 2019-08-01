import cv2
import os
import numpy as np
import time
from . import BaseDetectionModel

#TODO: extend from BaseDetectionModel
class YoloDetectionModel(BaseDetectionModel):
    #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    def __init__(self, confidence, threshold):
        super().__init__(confidence, threshold)
        modelPath = "models/yolo-coco"
        labelsPath = os.path.sep.join([modelPath, "coco.names"])
        # load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([modelPath, "yolov3.weights"])
        configPath = os.path.sep.join([modelPath, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)  #Most important line
        self._ln = self.net.getUnconnectedOutLayersNames()
        self.confidence = confidence
        self.threshold = threshold
        self.frameSize = (None, None)

    def getFrameProcessTime(self):
        (start, end) = self._frameProcessTime
        return end - start

    def detect(self, frame):
        (H, W) = self.frameSize
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            self.frameSize = (H, W)

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self._ln)
        end = time.time()
        self._frameProcessTime = (start, end)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        detections = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    # boxes.append([x, y, int(width), int(height)])
                    # confidences.append(float(confidence))
                    # classIDs.append(classID)

                    detections.append({
                        'box': [x, y, int(width), int(height)],
                        'confidence': float(confidence),
                        'classID': classID
                    })

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        boxes = list(map(lambda d: d['box'], detections))
        confidences = list(map(lambda d: d['confidence'], detections))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        idxs = idxs.flatten()
        #return boxes[idxs], confidences[idxs], classIDs[idxs]
        return [detections[i] for i in idxs]






