# USAGE
# python test_detector_frame.py
# (See README.md)

from models import YoloDetectionModel, SsdDetectionModel, FaceDetectionModel, CustomDetectionModel
import cv2

## --- MODEL SELECTION ---
# model = YoloDetectionModel(args["confidence"], args["threshold"])	# Yolo. [Make sure you have model file. See README.md]
# model = SsdDetectionModel(args["confidence"])		# SSD
model = CustomDetectionModel(confidence=0.5)		# Face

frame = cv2.imread('images/084_0072.jpg')
detections = model.detect(frame)
model.drawDetections(frame, detections)
cv2.imshow("output", frame)
key = cv2.waitKey(0) & 0xFF