# import necessary libraries
from . import FasterRcnnFlipDetectionModel

class CatDetectionModel(FasterRcnnFlipDetectionModel):  #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    # Confidence is called thershold in FasterRCNN
    def __init__(self, confidence, threshold = 0.1):
        super().__init__(confidence, threshold)
        self.LABELS = ['cat']
        self.COLORS = [(0, 0, 255)]

    def detect(self, frame):
        orig_detections = super().detect(frame)
        detections = []
        for orig_detection in orig_detections:
            cid = orig_detection['classID']
            # Classes that belong to a cat
            # This filter is just a shortcut trick for the prototype
            # In production, we would have a FasterRCNN trained with aerial cat photos, without the need of this filter
            if cid == 17 or cid == 18:  # Living
            # if cid == 16 or cid == 17 or cid == 18:     # Lilas
            # if  cid == 16 or cid == 17 or cid == 18 or cid == 88:     # deprecated
                print('Class detected: ', cid, 'Confidence: ', orig_detection['confidence'])
                orig_detection['classID'] = 0
                detections.append(orig_detection)
        return detections
