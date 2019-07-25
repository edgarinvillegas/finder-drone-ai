# USAGE
# python main.py --input videos/dog-kid-small.mp4 --output output/dog-kid-small.avi

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

#from models import VideoDetectionModel
#import models.VideoDetectionModel as VideoDetectionModel
from models.VideoDetectionModel import VideoDetectionModel
from lib.VideoInput import VideoInput

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
# ap.add_argument("-y", "--type", required=False, help="network type")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


np.random.seed(42)

#model = VideoDetectionModel.VideoDetectionModel()
model = VideoDetectionModel(args["confidence"], args["threshold"])
net = model.net

vi = VideoInput(args["input"]).start()
total = vi.getTotalFrames()

writer = None


# def processFrame(self, frame):


# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = vi.getNextFrame()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if frame is None:
		break

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	#(boxes,  confidences, classIDs) = model.detect(frame)
	#model.drawDetections(frame, boxes, confidences, classIDs)
	detections = model.detect(frame)
	model.drawDetections(frame, detections)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = model.getFrameProcessTime()
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vi.release()