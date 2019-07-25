import cv2
import os
import numpy as np

class VideoDetectionModel:
    #LABELS
    #COLORS
    #net
    def __init__(self):
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
        self.ln = self.net.getUnconnectedOutLayersNames()






