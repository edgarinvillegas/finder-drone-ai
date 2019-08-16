from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse
import imutils
import globals.mission
from threading import Timer

from enum import Enum
from models import FaceDetectionModel, CatDetectionModel
from utils import missionStepToKeyFramesObj, mission_from_str, get_next_auto_key_fn
# from utils import get_squares_coords, should_block_boundaries
from utils import get_squares_push_directions

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
FPS = 3  # 3 is appropiate

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
#mission = mission_from_str('fff-bbb-ffgf-bbb')
#mission = mission_from_str('ffff-bbbb')
#mission = mission_from_str('fffff-rrrrr-ll-llbbb-fffff-rrrrr-ll-llbbb-')
#mission = mission_from_str('-frlllr-b--llrrbrfll--ffr--fbf-rrf-r-l-rb--b-r-b-l')
#mission = mission_from_str('fffff-l-bbbb-l-fffff-l-bbbb-l-fffff-l-bbbb-l-fffff-l-bbbb-l-fffff-l-bbbb')
mission = mission_from_str('fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb-l-fffff-l-bbbbb')
print(mission)


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

PMode = Enum('PilotMode', 'NONE FIND FOLLOW FLIP')

onFoundAction = PMode.FLIP  # Can be FOLLOW or FLIP

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
        self.mode = PMode.NONE  # Can be '', 'FIND', 'OVERRIDE' or 'FOLLOW'

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

        self.tello.get_battery()
        
        # Safety Zone X
        szX = args.saftey_x

        # Safety Zone Y
        szY = args.saftey_y
        
        if args.debug:
            print("DEBUG MODE ENABLED!")

        while not should_stop:
            frame_time_start = time.time()
            # self.update() # Moved to the end before sleep to have more accuracy

            if frame_read.stopped:
                frame_read.stop()
                self.update() ## Just in case
                break


            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            print('---')
            # TODO: Analize if colors have to be tweaked
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = cv2.flip(frame_read.frame, 0)   # Vertical flip due to the mirror
            # frameRet = frame_read.frame

            vid = self.tello.get_video_capture()
            
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
                self.mode = PMode.FIND
                DETECT_ENABLED = True   # To start following with autopilot
                OVERRIDE = False
                print('Switch to spiral mode')

            # This is temporary, follow mode should start automatically
            if k == ord('f') and self.send_rc_control == True:
                DETECT_ENABLED = True
                OVERRIDE = False
                print('Switch to follow mode')

            # Press L to land
            if k == ord('g'):
                self.land_and_set_none()
                # self.update()  ## Just in case
                # break

            # Press Backspace for controls override
            if k == 8:
                if not OVERRIDE:
                    OVERRIDE = True
                    print("OVERRIDE ENABLED")
                else:
                    OVERRIDE = False
                    print("OVERRIDE DISABLED")

            # Quit the software
            if k == 27:
                should_stop = True
                self.update()  ## Just in case
                break

            autoK = -1
            if k == -1 and self.mode == PMode.FIND:
                if not OVERRIDE:
                    autoK = next_auto_key()
                    if autoK == -1:
                        self.mode = PMode.NONE
                        print('Queue empty! no more autokeys')
                    else:
                        print('Automatically pressing ', chr(autoK))

            key_to_process = autoK if k == -1 and self.mode == PMode.FIND and OVERRIDE == False else k

            if self.mode == PMode.FIND and not OVERRIDE:
                #frame ret will get the squares drawn after this operation
                if self.process_move_key_andor_square_bounce(key_to_process, frameRet) == False:
                    # If the queue is empty and the object hasn't been found, land and finish
                    self.land_and_set_none()
                    #self.update()  # Just in case
                    break
            else:
                self.process_move_key(key_to_process)

            dCol = (0, 255, 255)
            #detected = False
            if not OVERRIDE and self.send_rc_control and DETECT_ENABLED:
                self.track_object(OVERRIDE, frameRet, szX, szY)

            show = ""
            if OVERRIDE:
                show = "MANUAL"
                dCol = (255,255,255)
            elif self.mode == PMode.FOLLOW or self.mode == PMode.FLIP:
                show = "FOUND!!!"
            elif self.mode == PMode.FIND:
                show = "Finding.."

            mode_label = 'Mode: {}'.format(self.mode)

            # Draw the distance choosen
            cv2.putText(frameRet, mode_label, (32, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frameRet, show, (32,600), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol, 2)

            # Display the resulting frame
            cv2.imshow('FINDER DRONE', frameRet)
            if (self.mode == PMode.FLIP):
                self.flip()
                # OVERRIDE = True

            self.update()  # Moved here instead of beginning of loop to have better accuracy

            frame_time = time.time() - frame_time_start
            sleep_time = 1 / FPS - frame_time
            if sleep_time < 0:
                sleep_time = 0
                print('SLEEEP TIME NEGATIVE FOR FRAME {} ({}s).. TURNING IT 0'.format(imgCount, frame_time))
            if args.save_session and self.send_rc_control == True:  # To avoid recording before takeoff
                output_filename = "{}/tellocap{}.jpg".format(ddir,imgCount)
                print('Created {}'.format(output_filename))
                cv2.imwrite(output_filename, frameRet)
            time.sleep(sleep_time)


        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        # cv2.destroyWindow('FINDER DRONE')
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def flip(self):
        print('Flip!')
        self.left_right_velocity = self.for_back_velocity = 0
        self.update()
        time.sleep(self.tello.TIME_BTW_COMMANDS)
        self.tello.flip_left()
        #self.tello.flip_right()
        # The following 2 lines allow going back to follow mode
        self.mode = PMode.FOLLOW
        global onFoundAction
        onFoundAction = PMode.FOLLOW # So it doesn't flip over and over

    def land_and_set_none(self):
        if not args.debug:
            print("------------------Landing--------------------")
            self.tello.land()
        self.send_rc_control = False
        self.mode = PMode.NONE  # TODO: Consider calling reset


    def oq_discard_keys(self, keys_to_pop):
        oq = globals.mission.operations_queue
        keys_to_pop += 'p'
        while len(oq) > 0:
            oqkey = oq[0]['key']
            if oqkey in keys_to_pop:
                print('Removing {} from queue'.format(oqkey))
                oq.popleft()
            else:
                break

    def process_move_key_andor_square_bounce(self, k, frameRet):
        self.process_move_key(k)  # By default use key direction
        (hor_dir, ver_dir) = get_squares_push_directions(frameRet)
        print('(hor_dir, ver_dir): ({}, {})'.format(hor_dir, ver_dir))
        oq = globals.mission.operations_queue
        print('operations_queue len: ', len(oq))
        keys_to_pop = ''
        if ver_dir == 'forward':
            self.for_back_velocity = int(S)
            if k != ord('i'): print('Square pushing forward')
            keys_to_pop += 'k'
        elif ver_dir == 'back':
            self.for_back_velocity = -int(S)
            if k != ord('k'): print('Square pushing back')
            keys_to_pop += 'i'
        if hor_dir == 'right':
            self.left_right_velocity = int(S)
            if k != ord('l'): print('Square pushing right')
            keys_to_pop += 'j'
        elif hor_dir == 'left':
            self.left_right_velocity = -int(S)
            if k != ord('j'): print('Square pushing left')
            keys_to_pop += 'l'
        if(len(keys_to_pop) > 0):
            self.oq_discard_keys(keys_to_pop)
        return (len(oq) > 0)

    def process_move_key(self, k):
        # i & k to fly forward & back
        if k == ord('i'):
            self.for_back_velocity = int(S)
        elif k == ord('k'):
            self.for_back_velocity = -int(S)
        else:
            self.for_back_velocity = 0
        # o & u to pan left & right
        if k == ord('o'):
            self.yaw_velocity = int(S)
        elif k == ord('u'):
            self.yaw_velocity = -int(S)
        else:
            self.yaw_velocity = 0
        # y & h to fly up & down
        if k == ord('y'):
            self.up_down_velocity = int(S)
        elif k == ord('h'):
            self.up_down_velocity = -int(S)
        else:
            self.up_down_velocity = 0
        # l & j to fly left & right
        if k == ord('l'):
            self.left_right_velocity = int(S)
        elif k == ord('j'):
            self.left_right_velocity = -int(S)
        else:
            self.left_right_velocity = 0
        # p to keep still
        if k == ord('p'):
            print('pressing p')

    def save_detection(self, frameRet, firstDetection):
        output_filename_det_full = "{}/detected_full.jpg".format(ddir)
        cv2.imwrite(output_filename_det_full, frameRet)
        print('Created {}'.format(output_filename_det_full))
        (x, y, w, h) = firstDetection['box']
        add_to_borders = 100
        (xt, yt) = (x + w + add_to_borders, y + h + add_to_borders)
        (x, y) = (max(0, x - add_to_borders), max(0, y - add_to_borders))

        subframe = frameRet[y:yt, x:xt].copy()
        def show_detection():
            output_filename_det_sub = "{}/detected_sub.jpg".format(ddir)
            cv2.imwrite(output_filename_det_sub, subframe)
            print('Created {}'.format(output_filename_det_sub))
            cv2.imshow('Detected', subframe)
            cv2.waitKey(0)
        Timer(1.0, show_detection).start()

    def track_object(self, OVERRIDE, frameRet, szX, szY):
        detections = self.model.detect(frameRet)
        # These are our center dimensions
        (frame_h, frame_w) = frameRet.shape[:2]
        cWidth = int(frame_w / 2)
        cHeight = int(frame_h / 2)
        noDetections = len(detections) == 0
        if len(detections) > 0:
            print('CAT FOUND!!!!!!!!!!')
            if self.mode != onFoundAction:  # To create it only the first time
                self.save_detection(frameRet, detections[0])
            self.mode = onFoundAction
        # if we've given rc controls & get object coords returned
        #if self.send_rc_control and not OVERRIDE:
        if self.mode == PMode.FOLLOW and not OVERRIDE:
            print('Following...')
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

                if not args.debug:
                    if vDistance[0] < -szX:
                        # Right
                        self.left_right_velocity = S
                    elif vDistance[0] > szX:
                        # Left
                        self.left_right_velocity = -S
                    else:
                        self.left_right_velocity = 0

                    # for up & down
                    if vDistance[1] > szY:
                        self.for_back_velocity = S
                    elif vDistance[1] < -szY:
                        self.for_back_velocity = -S
                    else:
                        self.for_back_velocity = 0

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
                print("CAT NOT DETECTED NOW")
        # Draw the center of screen circle, this is what the drone tries to match with the target coords
        # cv2.circle(frameRet, (cWidth, cHeight), 10, (0, 0, 255), 2)

        return len(detections) > 0

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            print('Sending speeds to tello. H: {} V: {}'.format(self.left_right_velocity, self.for_back_velocity) )
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


if __name__ == '__main__':
    main()
