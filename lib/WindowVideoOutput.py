import cv2
from . import BaseVideoOutput

class WindowVideoOutput(BaseVideoOutput):
    def __init__(self, winName = "output"):
        super().__init__(winName)
        self.winName = winName
        self._writer = None

    def write(self, frame):
        #(H, W) = self._frameSize
        cv2.imshow(self.winName, frame)
        key = cv2.waitKey(1) & 0xFF

    def release(self):
        super().release()
        #cv2.destroyWindow(self.winName)
