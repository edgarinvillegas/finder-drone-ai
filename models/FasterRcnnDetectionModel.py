# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
from . import BaseDetectionModel

class FasterRcnnDetectionModel(BaseDetectionModel):  #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    # Confidence is called thershold in FasterRCNN
    def __init__(self, threshold):
        super().__init__(threshold)
        self.threshold = threshold
        # Class labels from official PyTorch documentation for the pretrained model
        # Note that there are some N/A's
        # for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        # we will use the same list for this notebook
        self.LABELS = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.useCuda = torch.cuda.is_available()
        if self.useCuda:
            self.net.cuda()
        self.net.eval() # To use model for inference
        self.transform = T.Compose([T.ToTensor()])

    def detect(self, frame):
        #(h, w) = frame.shape[:2]
        img = Image.fromarray(frame)
        img = self.transform(img)
        if self.useCuda: img = img.cuda()
        pred = self.net.forward([img])
        if self.useCuda:
            pred = np.array(pred)
            pred[0]['labels'] = pred[0]['labels'].cpu()
            pred[0]['boxes'] = pred[0]['boxes'].cpu()
            pred[0]['scores'] = pred[0]['scores'].cpu()

        pred_classes = [i for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pre_pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold]        
        pred_t = 0 if len(pre_pred_t) == 0 else pre_pred_t[-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_classes = pred_classes[:pred_t + 1]
        # return pred_boxes, pred_class
        detections = []
        for i in range(0, len(pred_classes)):
            pred_box = pred_boxes[i]
            confidence = pred_score[i]
            classID = pred_classes[i]
            startX, startY = int(pred_box[0][0]), int(pred_box[0][1])
            endX, endY = pred_box[1][0], pred_box[1][1]

            detections.append({
                'box': [startX, startY, int(endX - startX), int(endY - startY)],
                'confidence': float(confidence),
                'classID': classID
            })
        return detections
