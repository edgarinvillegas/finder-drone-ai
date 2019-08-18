# USAGE
# python test_video_cat.py
# (See README.md)

# import the necessary packages
import argparse

from models import CatDetectionModel, MyCatsDetectionModel
from lib import VideoInput
from lib import FileVideoOutput, WindowVideoOutput
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

## --- VIDEO INPUT SELECTION ---
# vIn = VideoInput().start()	# Webcam
# vIn = VideoInput("rtsp://user:pass@192.168.1.100/11").start()	# Webcam with RTSP
# vIn = VideoInput("videos/dog-kid.mp4").start()  # File
vIn = VideoInput("videos/lil-jua-whi-garden-horiz.mp4").start()  # File
#vIn = VideoInput("videos/drone-tests/jardin02-lily_juanis_bicho.mp4").start()  # File

# model = CatDetectionModel(args["confidence"], args["threshold"])
model = MyCatsDetectionModel(args["confidence"], args["threshold"])

## --- VIDEO OUTPUT SELECTION ---
vOut = WindowVideoOutput()			     # Show results on window
# vOut = FileVideoOutput("output/output.avi")	 # Creates an output video file

f = 0
process_every = 1 # Number of frames
# loop over frames from the video file stream
total_time = 0
n_frames_processed = 0
while True:
	# read the next frame from the file
	frame = vIn.getNextFrame()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if frame is None:
		break
	f += 1
	if f % process_every != 0: continue

	detection_start = time.time()
	detections = model.detect(frame)
	detection_end = time.time()
	total_time += detection_end - detection_start
	n_frames_processed += 1

	model.drawDetections(frame, detections)

	# check if it's the first frame
	if f == 1:
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
	print("Avg frame detection time: {:.4f}s".format(total_time / n_frames_processed))
	print("FPS: {:.4f}".format(n_frames_processed / total_time ))

# release the file pointers
print("[INFO] cleaning up...")
vOut.release()
vIn.release()

