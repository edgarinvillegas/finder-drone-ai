import cv2
import imutils

def get_squares_coords(frame):
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
        if len(approx) >= 4 and len(approx) <= 6:
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
                square_coords.append((cX, cY))

    return square_coords

# Checks if there are squares in the boundaries
# @param frameRet the drone frame
# Returns Clockwise tuple of booleans, true if it should be blocked
def should_block_boundaries(frameRet):
    squares_coords = get_squares_coords(frameRet)
    (h, w) = frameRet.shape[:2]
    block_right, block_forward, block_left, block_back = (False, False, False, False)
    # percentage of width to have on left and right boundaries to detect squares. Max 0.5
    w_perc = 0.4
    # percentage of height to have on top and bottom boundaries to detect squares. Max 0.5
    h_perc = 0.4
    for x, y in squares_coords:
        print('Square found on ({}, {})'.format(x, y))
        # Square to the left, block j
        if x < w * w_perc:
            #print('Square detected to the left, cancelling key {}'.format(chr(autoK)))
            block_left = True
        # Square to the right, block l
        if x > (w - w * w_perc):
            block_right = True
        # Square forward, block i
        if y < h * (h_perc):
            block_forward = True
        # Square back, block k
        if y > (h - h * h_perc):
            block_back = True

    if block_right: print('Must block right')
    if block_forward: print('Must block forward')
    if block_left: print('Must block left')
    if block_back: print('Must block back')

    return (block_right, block_forward, block_left, block_back)