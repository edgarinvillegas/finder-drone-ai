import cv2
import numpy as np
import time
import torch
from . import BaseDetectionModel
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from PIL import Image

# Applying Transforms to the Data
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def predict(model, frame):
    transform = image_transforms['test']
    test_image = Image.fromarray(frame)
    # plt.imshow(test_image)

    test_image_tensor = transform(test_image)
    print('cuda available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        prediction = topclass.cpu().numpy()[0][0]
        score = topk.cpu().numpy()[0][0]
        return (prediction, score)
        # for i in range(3):
        #     print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
        #           topk.cpu().numpy()[0][i])

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
        print('Loading model...')
        self.net = model = torch.load('models/custom/custom.pt')
        print('Model loaded.')

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        (prediction, score) = predict(self.net, frame)
        print('prediction: ', prediction)
        startX, startY, endX, endY = (10, 10, w-20, h-20)
        detections = [{
            'box': [startX, startY, int(endX - startX), int(endY - startY)],
            'confidence': float(score),
            'classID': prediction
        }]
        return detections
