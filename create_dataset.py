# USAGE
# python create_dataset.py
# (See README.md)

# import the necessary packages
import argparse
import numpy as np

from models import YoloDetectionModel, SsdDetectionModel, FaceDetectionModel, CustomDetectionModel
from lib import VideoInput
from lib import FileVideoOutput, WindowVideoOutput
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

## --- VIDEO INPUT SELECTION ---
currentClass = "whisky"
vIn = VideoInput("dataset/{}.mp4".format(currentClass)).start()  # File

## --- MODEL SELECTION ---
# model = YoloDetectionModel(args["confidence"], args["threshold"])	# Yolo. [Make sure you have model file. See README.md]
model = SsdDetectionModel(args["confidence"])		# SSD
# model = FaceDetectionModel(args["confidence"])		# Face
# model = CustomDetectionModel(confidence=0.3)

## --- VIDEO OUTPUT SELECTION ---
vOut = WindowVideoOutput()			     # Show results on window
# vOut = FileVideoOutput("output/output.avi")	 # Creates an output video file

isFirstFrame = True

#def processCatDetection()

i = 1

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
	catDetection = next(filter(lambda d: d['classID'] == 8, detections), None)
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

		# for example: dataset/train/juana1.jpg
		cv2.imwrite('dataset/{}/{}/{}{}.jpg'.format(set, currentClass, currentClass, i), subframe)

	model.drawDetections(frame, detections)

	# check if it's the first frame
	if isFirstFrame:
		vOut.setFrameSize((frame.shape[0], frame.shape[1]))
		# some information on processing single frame
		total = vIn.getTotalFrames()
		if total > 0:
			elap = model.getFrameProcessTime()
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
		isFirstFrame = False
	# write the output frame to disk
	vOut.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
vOut.release()
vIn.release()