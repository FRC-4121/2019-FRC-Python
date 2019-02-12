#!/usr/bin/env python3

###Hybridization work-in-progress.  Do not use.



#System imports
import sys
import imp

#Module imports
import cv2 as cv
import numpy as np
import datetime
import time
import logging
import argparse
import math
import cscore as cs
from cscore import CameraServer

#Set up basic logging
logging.basicConfig(level=logging.DEBUG)

#Initialize operating constants
global imgWidth = 640  
global imgHeight = 480
global imgBrightness = .5
global cameraFieldOfView = 27.3

visionTargetWidth = 3.31
visionTargetHeight = 5.82
minArea = 100 #for other detection functions

#Initialize variables to return through network tables
#What all should go here?

#Initialize variables for general purpose use
targetX = 0
targetY = 0
targetW = 0
targetH = 0

#Define program control flags
writeVideo = False

#Define image processing method
def process_image(imgRaw, hsvMin, hsvMax):
    
    #Blur image to remove noise
    blur = cv.GaussianBlur(imgRaw.copy(),(7,7),0)
        
    #Convert from BGR to HSV colorspace
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    #Set pixels to white if in target
    #HSV range, else set to black
    mask = cv.inRange(hsv, hsvMin, hsvMax)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    #Find contours in mask
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    return contours


#Define processing class
def detect_ball_target(imgRaw):

    #Define constraints for ball detection
    ballRadius = 6.5 #in inches
    minRadius = 50 #in pixels, this can be tweaked as needed

    #Define the lower and upper boundaries of the "green"
    #ball in the HSV color space
    ballHSVMin = (63, 0, 87)
    ballHSVMax = (108, 255, 255)

    #Values to be returned
    targetRadius = 0 #px
    targetX = -1 #px
    targetY = -1 #px
    distanceToBall = -1 #inches
    angleToBall = -1 #degrees
    foundBall = False;

    #Find contours in the mask and clean up the return style from OpenCV
    ballContours = process_image(imgRaw, ballHSVMin, ballHSVMax)

    if len(ballContours) == 2:
        ballContours = ballContours[0]
    elif len(ballContours) == 3:
        ballContours = ballContours[1]

    #Only proceed if at least one contour was found
    if len(ballContours) > 0:
        
        largestContour = max(ballContours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(largestContour)

        if radius > minRadius:

            targetRadius = radius
            targetX = x
            targetY = y
            foundBall = True

        #Distance and angle offset calculations
        if targetRadius != 0:
            
            inches_per_pixel = ballRadius/targetRadius #set up a general conversion factor
            distanceToBall = inches_per_pixel * (imgwidth / (2 * math.tan(math.radians(FOV_in_degrees))))
            angleOffsetInInches = inches_per_pixel * (targetX - imgwidth / 2)
            angleToBall = math.degrees(math.atan((angleOffsetInInches / distanceToBall)))
          
        else:
            
            distanceToBall = 0
            angleToBall = 0

    #cv.circle(img, (int(targetX), int(targetY)), int(targetRadius), (0, 255, 0), 2)

    return targetX, targetY, targetRadius, distanceToBall, angleToBall, foundBall

    
#Define floor alignment tape detection method
def detect_floor_tape(imgRaw):
    
    #Define constraints for detecting floor tape
    floorTapeWidth = 2.0
    floorTapeLength = 18.0
    minTapeArea = 100

    #Define HSV range for white alignment tape
    tapeHSVMin = (0, 0, 68)
    tapeHSVMax = (192, 100, 255)

    #Values to be returned
    targetX = -1
    targetY = -1
    targetW = -1
    targetH = -1
    foundTape = False
    
    #Find alignment tape in image
    tapeContours = process_image(imgRaw, tapeHSVMin, tapeHSVMax)
  
    #Continue with processing if alignment tape found
    if len(tapeContours) > 0:

        #find the largest contour and check it against the mininum tape area
        largestContour = max(tapeContours, key=cv.contourArea)

        if largestContour.area > minTapeArea:
            
            targetX, targetY, targetW, targetH = cv.boundingRect(largestContour)
            foundTape = True

    #cv.rectangle(img,(targetX,targetY),(targetX+targetW,targetY+targetH),(100,0,255),1)

    return targetX, targetY, targetW, targetH, foundTape
    

#Define contour detector function
def detect_vision_targets(imgRaw):

    #Set constraints for detecting vision targets
    visionTargetWidth = 3.313
    visionTargetHeight = 5.826
    minArea = 0

    #Define HSV range for cargo ship vision targets
    visionTargetHSVMin = (79, 91, 38)
    visionTargetHSVMax = (96, 255, 255)
    
    #Values to be returned
    targetH = -1
    targetW = -1
    targetX = -1
    targetY = -1
    distanceToVisionTarget = -1
    angleToVisionTarget = -1
    foundVisionTarget = False

    #Find contours in mask
    visionTargetContours = process_image(imgRaw, visionTargetHSVMin, visionTargetHSVMax)
    
    inchesPerPixel = 0
    
    #Loop over all contours to find valid targets based on color range 1
    for testContour1 in contours:

        #Get bounding rectangle
        x1, y1, w1, h1 = cv.boundingRect(testContour1)

        inchesPerPixel = visionTargetHeight/h1

        #Compare contour with other contours to find ones that are 
        for testContour2 in contours:

            #Get bounding rectangle
            x2, y2, w2, h2 = cv.boundingRect(testContour2)

            diffTargets = x2 - (x1 + w1)
            if diffTargets * inchesPerPixel > 7.9 and diffTargets * inchesPerPixel < 8.1:
                targetX = x1
                targetY = y1
                targetW = w1 + w2 + diffTargets
                targetH = max(h1, h2)

        if targetW * targetH > minarea:
    
            #Draw rectangle on image
            img_contours = cv.rectangle(img_raw,(targetX,targetY),(targetX+targetW,targetY+targetH),(0,0,255),1)

            

    ##Work in progress, needs to be cleaned somewhat.


    #cv.rectangle(imgRaw,(targetX,targetY),(targetX+targetW,targetY+targetH),(0,0,255),1)

    #return results
    return targetX, targetY, targetW, targetH, distanceToVisionTarget, angleToVisionTarget, foundVisionTarget


#define main processing function
def main():

    #Set up a camera server
    camserv = CameraServer.getInstance()
    camserv.enableLogging

    #Start capturing webcam video
    camera = camserv.startAutomaticCapture(dev=0, name="MainPICamera")
    camera.setResolution(imgWidth, imgHeight)
    camera.setBrightness(imgBrightness)

    #Define video sink
    cvsink = camserv.getVideo()

    #Create network table things here
##    ballX
##    ballY
##    ballRadius
##    ballDistance
##    ballAngle
##    foundBall
##
##    tapeX
##    tapeY
##    tapeW
##    tapeH
##    foundTape
##
##    visionTargetX
##    visionTargetY
##    visionTargetW
##    visionTargetH
##    visionTargetDistance
##    visionTargetAngle
##    foundVisionTarget

    #Create blank image
    img = np.zeros(shape=(imgWidth, imgHeight, 3), dtype=np.uint8)

    #Start main processing loop
    while (True):

        #Read in an image from 2019 Vision Images
        #img = '2019VisionImages\CargoLine36in.jpg'

        #OR grab an image from a video feed
        img = cvsink.grabFrame(img)

        ballX, ballY, ballRadius, ballDistance, ballAngle, foundBall = detect_ball_target(img)
        tapeX, tapeY, tapeW, tapeH, foundTape = detect_floor_tape(img)
        visionTargetX, visionTargetY, visionTargetW, visionTargetH, visionTargetDistance, visionTargetAngle, foundVisionTarget = detect_vision_targets(img)

        #Network table updating to go here

        #Draw various contours on the image
        cv.circle(img, (int(ballX), int(ballY)), int(ballRadius), (0, 255, 0), 2) #ball
        cv.rectangle(img,(tapeX,tapeY),(tapeX+tapeW,tapeY+tapeH),(100,0,255),1) #floor tape
        cv.rectangle(img,(visionTargetX,visionTargetY),(visionTargetX+visionTargetW,visionTargetY+visionTargetH),(100,0,255),1) #vision targets

        #Check for stop code from robot
        if cv.waitKey(1) == 27:
            break
        #robotStop = visionTable.getNumber("RobotStop", 0)
        #if robotStop == 1:
        #    break

    #Close all open windows
    cv.destroyAllWindows()
    

#define main function
if __name__ == '__main__':
    main()

