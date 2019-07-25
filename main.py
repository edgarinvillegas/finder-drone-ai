# USAGE
# python main.py --input videos/dog-kid-small.mp4 --output output/dog-kid-small.avi

# import the necessary packages
import numpy as np
import argparse

from models import VideoDetectionModel
from lib import VideoInput
from lib import FileVideoOutput, WindowVideoOutput

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
# ap.add_argument("-y", "--type", required=False, help="network type")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


np.random.seed(42)

model = VideoDetectionModel(args["confidence"], args["threshold"])
vIn = VideoInput(args["input"]).start()

#vOut = FileVideoOutput("output/dog-kid-small.avi")
vOut = WindowVideoOutput()

isFirstFrame = True

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = vIn.getNextFrame()

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