# USAGE
# python test_video.py
# (See README.md)

# import the necessary packages
import argparse

from models import YoloDetectionModel, SsdDetectionModel, FaceDetectionModel, CustomDetectionModel, FasterRcnnDetectionModel
from lib import VideoInput
from lib import FileVideoOutput, WindowVideoOutput
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

## --- VIDEO INPUT SELECTION ---
#vIn = VideoInput().start()	# Webcam
# vIn = VideoInput("rtsp://user:pass@192.168.1.100/11").start()	# Webcam with RTSP
vIn = VideoInput("videos/dog-kid.mp4").start()  # File
# vIn = VideoInput("D:/edgarin/mycode/(COURSES)/fbook-ia/drone-find-anything/cats-raw/raw/Juanis/VID_20190805_233242.mp4").start()  # File

## --- MODEL SELECTION ---
# model = YoloDetectionModel(args["confidence"], args["threshold"])	# Yolo. [Make sure you have model file. See README.md]
# model = SsdDetectionModel(args["confidence"])		# SSD
# model = FaceDetectionModel(args["confidence"])		# Face
model = FasterRcnnDetectionModel(args["confidence"])
# model = CustomDetectionModel(confidence=0.3)

## --- VIDEO OUTPUT SELECTION ---
vOut = WindowVideoOutput()			     # Show results on window
# vOut = FileVideoOutput("output/output.avi")	 # Creates an output video file

isFirstFrame = True

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = vIn.getNextFrame()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if frame is None:
		break

	detections = model.detect(frame)
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