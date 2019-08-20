# USAGE
# python create_dataset.py
# (See README.md)

# import the necessary packages
import argparse
import numpy as np

from models import YoloDetectionModel, SsdDetectionModel, FaceDetectionModel, CatDetectionModel
from lib import VideoInput
from lib import FileVideoOutput, WindowVideoOutput
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

## --- VIDEO INPUT SELECTION ---
currentClass = "whisky"
vIn = VideoInput("dataset/{}.mp4".format(currentClass)).start()  # File

## --- MODEL SELECTION ---
# model = YoloDetectionModel(args["confidence"], args["threshold"])	# Yolo. [Make sure you have model file. See README.md]
# model = SsdDetectionModel(args["confidence"])		# SSD
# model = FaceDetectionModel(args["confidence"])		# Face
model = CatDetectionModel(args["confidence"], args["threshold"])

## --- VIDEO OUTPUT SELECTION ---
vOut = WindowVideoOutput()			     # Show results on window
# vOut = FileVideoOutput("output/output.avi")	 # Creates an output video file

isFirstFrame = True

#def processCatDetection()

i = 4000

def is_cat(detection):
	label = model.LABELS[detection['classID']];
	return label == 'cat'

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = vIn.getNextFrame()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if frame is None:
		break

	i += 1
	if i % 10 != 0:
		continue

	detections = model.detect(frame)
	catDetection = next(filter(is_cat, detections), None)
	if not (catDetection is None):
		x, y, w, h = catDetection['box']
		subframe = frame[y:y+h, x:x+w]
		r = np.random.rand()
		if r < 0.6:
			set = 'train'
		elif r < 0.8:
			set = 'valid'
		else:
			set = 'test'

		# for example: dataset/train/juana/juana1.jpg
		cv2.imwrite('dataset/{}/{}/{}{}.jpg'.format(set, currentClass, currentClass, i), subframe)

	model.drawDetections(frame, detections)

	(h, w) = frame.shape[:2]
	frame = cv2.resize(frame, (w // 2, h // 2))
	# check if it's the first frame
	if isFirstFrame:
		vOut.setFrameSize((h, w))
		# some information on processing single frame
		total = vIn.getTotalFrames()
		if total > 0:
			elap = model.getFrameProcessTime()
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
		isFirstFrame = False
	# write the output frame
	vOut.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
vOut.release()
vIn.release()