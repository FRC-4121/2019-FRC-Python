# import the necessary packages
from collections import deque
#from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
#import imutils
import time
import math
import cscore as cs

from cscore import CameraServer

#logging

#image variables
ballRadius = 6.5 #fixed value
imgwidth = 640 #fixed value
imgheight = 480
FOV_in_degrees = 27.3 #fixed value
minRadius = 50
targetX = 0
targetY = 0
targetRadius = 0
distanceToBall = -1
angleToBall = -1

#program flags

#Set up a camera server
camserv = CameraServer.getInstance()
camserv.enableLogging

#Start capturing webcam video
camera = camserv.startAutomaticCapture(dev=0, name="MainPICamera")
camera.setResolution(imgwidth, imgheight)

#Define video sink
cvsink = camserv.getVideo()

#Create blank image
img = np.zeros(shape=(imgwidth, imgheight, 3), dtype=np.uint8)

#contour detection function (basically the below)

# keep looping
while True:

    # grab the current frame
    video_timestamp, img = cvsink.grabFrame(img)

    # resize the frame, blur it, and convert it to the HSV
    # color space
    #frame = imutils.resize(frame, width=imgwidth)
    blurred = cv2.GaussianBlur(img.copy(), (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space
    targetMin = (0, 50, 235)
    targetMax = (61, 255, 255)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, targetMin, targetMax)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:

      #draw all contours found in mask with red outline
      for contour in cnts:
        
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > minRadius:
          cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 2)
          cv2.circle(img, center, 5, (0, 0, 255), -1)


      #draw the largest circle with a green outline
      largestContour = max(cnts, key=cv2.contourArea)
      ((x, y), radius) = cv2.minEnclosingCircle(largestContour)
      M = cv2.moments(largestContour)
      center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

      if radius > minRadius:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        targetRadius = radius
        targetX = x
        targetY = y

      #distance and angle offset calculations
      if targetRadius != 0:
          inches_per_pixel = ballRadius/targetRadius #set up a general conversion factor
          distanceToBall = inches_per_pixel * (imgwidth / (2 * math.tan(math.radians(FOV_in_degrees))))
          angleOffsetInInches = inches_per_pixel * (targetX - imgwidth / 2)
          angleToBall = math.degrees(math.atan((angleOffsetInInches / distanceToBall)))
      else:
          distanceToBall = 0
          angleToBall = 0
    
    cv2.putText(img, 'Target X: %.2f' %targetX, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
    cv2.putText(img, 'Target Y: %.2f' %targetY, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
    cv2.putText(img, 'Target Radius: %.2f' %targetRadius, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
    cv2.putText(img, 'Target Distance: %.2f' %distanceToBall, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
    cv2.putText(img, 'Target Angle: %.2f' %angleToBall, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
    
    # show the frame to our screen
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# stop grabbing video feed
#vs.stop()

# close all windows
cv2.destroyAllWindows()
