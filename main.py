from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse

from enum import Enum
from models import FaceDetectionModel, CatDetectionModel
from utils import missionStepToKeyFramesObj, mission_from_str, get_next_auto_key_fn

# standard argparse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=3,
    help='use -d to change the distance of the drone. Range 0-6')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
    help='use -sx to change the saftey bound on the x axis . Range 0-480')
parser.add_argument('-sy', '--saftey_y', type=int, default=100,
    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

# Speed of the drone (cm/s)
S = 20
# Step size (cm)
step_size = 40

S2 = 5

#UDOffset = 150
# These are the values in which kicks in speed up mode, as of now, this hasn't been finalized or fine tuned so be careful
# Tested are 3, 4, 5
#acc = [500,250,250,150,110,70,50]

# Frames per second of the window display
FPS = 3

# Frames needed per step
frames_step = step_size / S * FPS

old_mission = [
    {'direction': 'forward', 'steps': 1},
    {'direction': 'right', 'steps': 1},
    {'direction': 'back', 'steps': 2},
    {'direction': 'left', 'steps': 2},

    {'direction': 'forward', 'steps': 3},
    {'direction': 'right', 'steps': 3},
    {'direction': 'back', 'steps': 4},
    {'direction': 'left', 'steps': 4},
]

#mission = mission_from_str('fbfb')
#mission = mission_from_str('fff-bbb-fff-bbb')
mission = mission_from_str('ff-l-bb')
#print(mission)

# This transforms the mission in a set of simulated keys to be pressed.
# next_auto_key will be a function that returns the new key to press every time
next_auto_key = get_next_auto_key_fn(mission, frames_step)

# If we are to save our sessions, we need to make sure the proper directories exist
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

PMode = Enum('PilotMode', 'NONE SPIRAL FOLLOW')

class DroneUI(object):
    
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.mode = PMode.NONE  # Can be '', 'SPIRAL', 'OVERRIDE' or 'FOLLOW'

        self.send_rc_control = False

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        self.model = CatDetectionModel(0.5)

        frame_read = self.tello.get_frame_read()

        should_stop = False
        imgCount = 0

        OVERRIDE = False
        DETECT_ENABLED = False  # Set to true to automatically start in follow mode
        self.mode = PMode.NONE

        # oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()
        
        # Safety Zone X
        szX = args.saftey_x

        # Safety Zone Y
        szY = args.saftey_y
        
        if args.debug:
            print("DEBUG MODE ENABLED!")

        while not should_stop:
            frame_time_start = time.time()
            self.update()

            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame

            vid = self.tello.get_video_capture()

            if args.save_session:
                cv2.imwrite("{}/tellocap{}.jpg".format(ddir,imgCount),frameRet)
            
            frame = np.rot90(frame)
            imgCount+=1

            #time.sleep(1 / FPS)

            # Listen for key presses
            k = cv2.waitKey(20)

            try:
                if chr(k) in 'ikjluoyhp': OVERRIDE = True
            except:
                ...

            # Press T to take off
            if k == ord('t'):
                if not args.debug:
                    print("Taking Off")
                    self.tello.takeoff()
                    self.tello.get_battery()
                self.send_rc_control = True

            if k == ord('s') and self.send_rc_control == True:
                self.mode = PMode.SPIRAL
                # DETECT_ENABLED = True   # To start following with spiral
                OVERRIDE = False
                print('Switch to spiral mode')

            # This is temporary, follow mode should start automatically
            if k == ord('f') and self.send_rc_control == True:
                DETECT_ENABLED = True
                OVERRIDE = False
                print('Switch to follow mode')

            # Press L to land
            if k == ord('g'):
                if not args.debug:
                    print("Landing")
                    self.tello.land()
                self.send_rc_control = False
                self.mode = PMode.NONE  # TODO: Consider calling reset

            # Press Backspace for controls override
            if k == 8:
                if not OVERRIDE:
                    OVERRIDE = True
                    print("OVERRIDE ENABLED")
                else:
                    OVERRIDE = False
                    print("OVERRIDE DISABLED")

            # if k == ord('1'):
            #     oSpeed = 1
            # # Press 2 to set speed to 2
            # if k == ord('2'):
            #     oSpeed = 2
            # # Press 3 to set speedee to 3
            # if k == ord('3'):
            #     oSpeed = 3

            # Quit the software
            if k == 27:
                should_stop = True
                break

            autoK = -1
            if k == -1 and self.mode == PMode.SPIRAL:
                if not OVERRIDE:
                    autoK = next_auto_key()
                    if autoK == -1:
                        self.mode = PMode.NONE
                    else:
                        print('Automatically pressing ', chr(autoK))

            key_to_process = autoK if k == -1 and self.mode == PMode.SPIRAL and OVERRIDE == False else k
            self.process_move_key(key_to_process)

            tDistance = 3

            dCol = (0, 255, 255)
            if not OVERRIDE and self.send_rc_control and DETECT_ENABLED:
                dCol = self.track_object(OVERRIDE, frameRet, szX, szY, tDistance)

            if OVERRIDE:
                show = "OVERRIDE"
                dCol = (255,255,255)
            else:
                show = "AI: {}".format(str(tDistance))

            mode_label = ' Mode: {}'.format(self.mode)

            # Draw the distance choosen
            cv2.putText(frameRet, mode_label, (32, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frameRet,show,(32,664),cv2.FONT_HERSHEY_SIMPLEX,1,dCol,2)

            # Display the resulting frame
            cv2.imshow(f'Tello Tracking...',frameRet)

            frame_time = time.time() - frame_time_start
            sleep_time = 1 / FPS - frame_time
            if sleep_time < 0:
                sleep_time = 0
                print('SLEEEP TIME NEGATIVE FOR FRAME {} ({}s).. TURNING IT 0'.format(imgCount, frame_time))
            time.sleep(sleep_time)

        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()
        
        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def process_move_key(self, k):
        # S & W to fly forward & back
        if k == ord('i'):
            self.for_back_velocity = int(S)
        elif k == ord('k'):
            self.for_back_velocity = -int(S)
        else:
            self.for_back_velocity = 0
        # a & d to pan left & right
        if k == ord('o'):
            self.yaw_velocity = int(S)
        elif k == ord('u'):
            self.yaw_velocity = -int(S)
        else:
            self.yaw_velocity = 0
        # Q & E to fly up & down
        if k == ord('y'):
            self.up_down_velocity = int(S)
        elif k == ord('h'):
            self.up_down_velocity = -int(S)
        else:
            self.up_down_velocity = 0
        # c & z to fly left & right
        if k == ord('l'):
            self.left_right_velocity = int(S)
        elif k == ord('j'):
            self.left_right_velocity = -int(S)
        else:
            self.left_right_velocity = 0

    def track_object(self, OVERRIDE, frameRet, szX, szY, tDistance):
        detections = self.model.detect(frameRet)
        # These are our center dimensions
        (frame_h, frame_w) = frameRet.shape[:2]
        cWidth = int(frame_w / 2)
        cHeight = int(frame_h / 2)
        noDetections = len(detections) == 0
        if len(detections) > 0:
            self.mode = PMode.FOLLOW
        # if we've given rc controls & get object coords returned
        #if self.send_rc_control and not OVERRIDE:
        if self.mode == PMode.FOLLOW and not OVERRIDE:
            for det in detections:
                (x, y, w, h) = det['box']
                # setting Object Box properties
                obCol = (255, 0, 0)  # BGR 0-255
                obStroke = 2

                # end coords are the end of the bounding box x & y
                end_cord_x = x + w
                end_cord_y = y + h
                end_size = w * 2

                # This is not face detection so we don't need offset
                UDOffset = 0

                # these are our target coordinates
                targ_cord_x = int((end_cord_x + x) / 2)
                targ_cord_y = int((end_cord_y + y) / 2) + UDOffset

                # This calculates the vector from the object to the center of the screen
                vTrue = np.array((cWidth, cHeight))
                vTarget = np.array((targ_cord_x, targ_cord_y))
                vDistance = vTrue - vTarget

                #
                if not args.debug:
                    # for turning
                    if vDistance[0] < -szX:
                        self.yaw_velocity = S
                        # self.left_right_velocity = S2
                    elif vDistance[0] > szX:
                        self.yaw_velocity = -S
                        # self.left_right_velocity = -S2
                    else:
                        self.yaw_velocity = 0

                    # for up & down
                    if vDistance[1] > szY:
                        self.up_down_velocity = S
                    elif vDistance[1] < -szY:
                        self.up_down_velocity = -S
                    else:
                        self.up_down_velocity = 0

                # Draw the object bounding box
                cv2.rectangle(frameRet, (x, y), (end_cord_x, end_cord_y), obCol, obStroke)

                # Draw the target as a circle
                cv2.circle(frameRet, (targ_cord_x, targ_cord_y), 10, (0, 255, 0), 2)

                # Draw the safety zone
                cv2.rectangle(frameRet, (targ_cord_x - szX, targ_cord_y - szY), (targ_cord_x + szX, targ_cord_y + szY),
                              (0, 255, 0), obStroke)

                # Draw the estimated drone vector position in relation to object bounding box
                cv2.putText(frameRet, str(vDistance), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # if there are no objects detected, don't do anything
            if noDetections:
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.for_back_velocity = 0
                print("NO TARGET")
        # Draw the center of screen circle, this is what the drone tries to match with the target coords
        cv2.circle(frameRet, (cWidth, cHeight), 10, (0, 0, 255), 2)
        dCol = lerp(np.array((0, 0, 255)), np.array((255, 255, 255)), tDistance + 1 / 7)
        return dCol

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

def lerp(a,b,c):
    return a + c*(b-a)

def main():
    # for i in range(8):
    #     key = next_auto_key()
    #     print(chr(key))
    frontend = DroneUI()

    # run frontend
    frontend.run()
    pass


if __name__ == '__main__':
    main()
