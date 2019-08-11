# import necessary libraries
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from . import BaseDetectionModel

class FasterRcnnDetectionModel(BaseDetectionModel):  #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    # Confidence is called thershold in FasterRCNN
    def __init__(self, confidence, threshold=0.3):
        super().__init__(confidence, threshold)
        self.confidence = confidence
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
        print('Loading FasterRcnnDetectionModel...')
        self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        print('FasterRcnnDetectionModel loaded')
        self.useCuda = torch.cuda.is_available()
        if self.useCuda:
            self.net.cuda()
        self.net.eval() # To use model for inference
        self.transform = T.Compose([T.ToTensor()])

    # This is intended to be overriden by derived classes that need single class detection
    # TODO: Change this to a proper filter that gets a detection as parameter
    def detect_single_class(self):
        return -1

    def detect(self, frame):
        #(h, w) = frame.shape[:2]
        # img = Image.fromarray(frame)  # Not needed anymore
        # img90 = self.transform(frame.rot90())
        img = self.transform(frame)
        if self.useCuda: img = img.cuda()

        preds = self.net.forward([img])
        pred = preds[0] # Because it can predict several images at a time, we're only doing for 1

        # If the model is for single class, pre-filter it
        if self.detect_single_class() > 0:
            filtered_indexes = pred['labels'] == self.detect_single_class()
            print('filtered_indexes', filtered_indexes)
            pred['labels'] = pred['labels'][filtered_indexes]
            pred['boxes'] = pred['boxes'][filtered_indexes]
            pred['scores'] = pred['scores'][filtered_indexes]

        # Apply non-maximum-supression (with cuda if available)
        nms_indexes = torchvision.ops.nms(pred['boxes'], pred['scores'], self.threshold)
        pred['labels'] = torch.index_select(pred['labels'], 0, nms_indexes)
        pred['boxes'] = torch.index_select(pred['boxes'], 0, nms_indexes)
        pred['scores'] = torch.index_select(pred['scores'], 0, nms_indexes)

        if self.useCuda:
            preds = np.array(preds)
            pred['labels'] = pred['labels'].cpu()
            pred['boxes'] = pred['boxes'].cpu()
            pred['scores'] = pred['scores'].cpu()

        pred_classes = [i for i in list(pred['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().numpy())]
        pred_score = list(pred['scores'].detach().numpy())
        pre_pred_t = [pred_score.index(x) for x in pred_score if x > self.confidence]
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
