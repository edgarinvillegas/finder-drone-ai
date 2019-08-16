# import necessary libraries
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from . import BaseDetectionModel

class FasterRcnnFlipDetectionModel(BaseDetectionModel):  #LABELS
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
        # There are 91 classes in the pretrained model
        self.LABELS = list(map(lambda cid: 'Class #{}'.format(cid), range(91)))
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
        original_shape = img.shape
        if self.useCuda:
            img = img.cuda()
        #print('img shape', img.shape)
        imgVflip = torch.flip(img, [1])  # Vertical flip ([1] because it's dimension 1)
        # We take advantage of predicting for the original image and the vertical flip in parallel
        preds = self.net.forward([img, imgVflip])

        predNormal, predFlip = preds

        # Now we have to flip the y coordinates of boxes
        h = img.shape[1]
        vflipBoxes =  predFlip['boxes']             # aux tensor
        startYOrigCol = vflipBoxes[:, 1].clone()
        vflipBoxes[:, 1] = h - vflipBoxes[:, 3]     # startY column will become flipped endY
        vflipBoxes[:, 3] = h - startYOrigCol        # endY column will become flipped startY

        pred = {
            'labels': torch.cat((predNormal['labels'], predFlip['labels'])),
            'boxes': torch.cat((predNormal['boxes'], vflipBoxes)),
            'scores': torch.cat((predNormal['scores'], predFlip['scores'])),
        }

        detections = []
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


        for i in range(0, len(pred_classes)):
            pred_box = pred_boxes[i]
            confidence = pred_score[i]
            classID = pred_classes[i]
            startX, startY = int(pred_box[0][0]), int(pred_box[0][1])
            endX, endY = pred_box[1][0], pred_box[1][1]
            if confidence > self.confidence:  # Added because I got a detection with 6% confidence
                detections.append({
                    'box': [startX, startY, int(endX - startX), int(endY - startY)],
                    'confidence': float(confidence),
                    'classID': classID
                })
        detections.sort(key = lambda d: d['confidence'], reverse = True)
        return detections
