# import necessary libraries
from . import FasterRcnnDetectionModel

class CatDetectionModel(FasterRcnnDetectionModel):  #LABELS
    #COLORS
    #net
    #confidence
    #threshold?
    #frameSize
    # Confidence is called thershold in FasterRCNN
    def __init__(self, threshold):
        super().__init__(threshold)
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
            if  cid == 16 or cid == 17 or cid == 18:
                orig_detection['classID'] = 0
                detections.append(orig_detection)
        return detections
