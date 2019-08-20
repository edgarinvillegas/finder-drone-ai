# USAGE
# python test_detector_frame.py
# (See README.md)

from models import YoloDetectionModel, SsdDetectionModel, FaceDetectionModel, SlidingWindowDetectionModel, \
    FasterRcnnDetectionModel, FasterRcnnFlipDetectionModel, CatDetectionModel, MyCatsDetectionModel
import cv2
import time

## --- MODEL SELECTION ---
# model = YoloDetectionModel(0.5, 0.3)	# Yolo. [Make sure you have model file. See README.md]
# model = YoloDetectionModel(0.3, 0.3)	# Yolo. [Make sure you have model file. See README.md]
# model = SsdDetectionModel(0.5)		# SSD
# model = SlidingWindowDetectionModel(confidence=0.5)		# Face
# model = FasterRcnnFlipDetectionModel(0.5, 0.1)
# model = FasterRcnnDetectionModel(0.5, 0.1)
model = CatDetectionModel(0.5)
# model = MyCatsDetectionModel()


#frame = cv2.imread('images/moose-in-room.jpg')
#frame = cv2.imread('images/landscape-with-giraffe.jpg')
#frame = cv2.imread('images/TelloPhoto/zorrino-gato-jeep-recortado.png')
#frame = cv2.imread('images/TelloPhoto/1565125920104.png')

# frame = cv2.imread('images/TelloPhoto/juanis-noche-3.png')
## frame = cv2.imread('images/juanis-noche-3-rect.png')         #GRAVE!!
frame = cv2.imread('images/cat-above-2.jpg')
# frame = cv2.imread('images/lily-juanis-jardin-frame.png')
# frame = cv2.imread('images/glass-cat.jpg')


#frame = cv2.flip(frame, 0)  # Flip Vertical
# frame = cv2.flip(frame, 1)  # Flip Horizontal
# frame = cv2.GaussianBlur(frame, (11, 11), 0)

start = time.time()
detections = model.detect(frame)

print('Took {} s'.format(time.time()-start))
model.drawDetections(frame, detections)
print(detections)
#cv2.imwrite('images/real-and-toy-cats_output.jpg', frame)
#cv2.imwrite('images/output.jpg', frame)
# cv2.imshow("output", cv2.resize(frame, (1296, 968)))
cv2.imshow("output", frame)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()