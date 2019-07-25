import cv2
from . import BaseVideoOutput

class FileVideoOutput(BaseVideoOutput):
    def __init__(self, filePath):
        super().__init__(filePath)
        # initialize our video writer
        self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.filePath = filePath
        self._writer = None

    def write(self, frame):
        (H, W) = self._frameSize
        if self._writer is None:
            self._writer = cv2.VideoWriter(self.filePath, self._fourcc, 30, (W, H), True)
        self._writer.write(frame)

    def release(self):
        super().release()
        self._writer.release()