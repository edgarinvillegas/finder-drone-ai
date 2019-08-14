import cv2
import numpy as np
import imutils
from pyimagesearch.transform import four_point_transform

def get_square_direction(full_frame, approx):
    # First we get the warped square. Gets the perspective and flattens it
    # to be a perfect square with vertical and horizontal edges
    warped = four_point_transform(full_frame, approx.reshape(4, 2))
    #cv2.imshow('warped', warped)
    #cv2.waitKey(0)
    # Now we search for the darkest spot
    spot_coords = get_spot_coords(warped)
    if not spot_coords is None:
        cv2.circle(warped, spot_coords, 20, (255, 0, 0), 2)
        x, y = spot_coords
        (h, w) = warped.shape[:2]
        directions = ('right', 'forward', 'left', 'back')
        distances_to_edges = (w-x, y, x, h-y)
        # print('distances_to_edges', distances_to_edges)
        min_index = np.argmin(distances_to_edges)
        direction = directions[min_index]
        # print(direction)
        # cv2.imshow('warped', warped)
        # cv2.waitKey(0)
        return direction
    else:
        return None

# byref frame (will be drawn)
def get_squares_push_directions(frame):
    hor_dir, ver_dir = ("", "")
    squares_coords = get_squares_coords_and_dirs(frame)
    dir_index = 2   # Because the tuple is (x, y, dir)
    some_forward = any(sq[dir_index] == 'forward' for sq in squares_coords)
    some_back = any(sq[dir_index] == 'back' for sq in squares_coords)
    some_right = any(sq[dir_index] == 'right' for sq in squares_coords)
    some_left = any(sq[dir_index] == 'left' for sq in squares_coords)
    if some_right and (not some_left): hor_dir = 'right'
    if some_left and (not some_right): hor_dir = 'left'
    if some_forward and (not some_back): ver_dir = 'forward'
    if some_back and (not some_forward): ver_dir = 'back'
    return (hor_dir, ver_dir)

# byref frame (will be drawn)
def get_spot_coords(frame):
    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    spot_coords = list()
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is "roughly" rectangular
        if True: #or len(approx) >= 4 and len(approx) <= 7:
            # compute the bounding box of the approximated contour and
            # use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)

            # compute the solidity of the original contour
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)

            # compute whether or not the width and height, solidity, and
            # aspect ratio of the contour falls within appropriate bounds
            keepDims = w > 10 and h > 10
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.3 and aspectRatio <= 3
            #print('size: ({}, {})'.format(w, h), 'solidity: ', solidity, 'aspectRatio: ', aspectRatio)

            # ensure that the contour passes all our tests
            if keepDims and keepSolidity and keepAspectRatio:
                # draw an outline around the target and update the status
                # text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)

                # compute the center of the contour region and draw the
                # crosshairs
                M = cv2.moments(approx)
                (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
                spot_coords.append((cX, cY))
                break   #Because we're only interested in the first one for now

    if(len(spot_coords) > 0):
        return spot_coords[0]
    else:
        return None

# byref frame (will be drawn)
def get_squares_coords_and_dirs(frame):
    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    square_coords = list()
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is "roughly" rectangular
        if len(approx) == 4:
            # compute the bounding box of the approximated contour and
            # use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)

            # compute the solidity of the original contour
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)

            # compute whether or not the width and height, solidity, and
            # aspect ratio of the contour falls within appropriate bounds
            keepDims = w > 25 and h > 25
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
            # print('size: ({}, {})'.format(w, h), 'solidity: ', solidity, 'aspectRatio: ', aspectRatio)

            # ensure that the contour passes all our tests
            if keepDims and keepSolidity and keepAspectRatio:
                # Calculate directions before drawing
                dir = get_square_direction(frame, approx)
                print('SQUARE FOUND size: ({}, {})'.format(w, h), 'solidity: ', solidity, 'aspectRatio: ', aspectRatio, 'dir: ', dir)
                # draw an outline around the target and update the status
                # text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)

                # compute the center of the contour region and draw the
                # crosshairs
                M = cv2.moments(approx)
                (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
                (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
                (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
                cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
                cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)

                # Draw direction indicators (a circle on the appropiate cross tip
                dir_circle_color = (255, 0, 0)
                if dir == 'left':
                    #left - blue
                    cv2.circle(frame, (startX, cY), 10, dir_circle_color, -2)
                elif dir == 'right':
                    #right - light blue
                    cv2.circle(frame, (endX, cY), 10, dir_circle_color, -2)
                elif dir == 'forward':
                    #forward - red
                    cv2.circle(frame, (cX, startY), 10, dir_circle_color, -2)
                elif dir == 'back':
                    # back - black
                    cv2.circle(frame, (cX, endY), 10, dir_circle_color, -2)

                square_coords.append((cX, cY, dir))

    return square_coords