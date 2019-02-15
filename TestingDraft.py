#!/usr/bin/env python3

###Hybridization work-in-progress.

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
from operator import itemgetter
import math
import cscore as cs
from cscore import CameraServer

#Set up basic logging
logging.basicConfig(level=logging.DEBUG)

#Initialize operating constants
imgWidth = 640  
imgHeight = 480
imgBrightness = .5
cameraFieldOfView = 27.3

#Initialize variables to return through network tables
#What all should go here?

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
    _, contours, _ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    return contours


#Define processing class
def detect_ball_target(imgRaw):

    #Define constraints for ball detection
    ballRadius = 6.5 #in inches
    minRadius = 50 #in pixels, this can be tweaked as needed

    #Define the lower and upper boundaries of the "green"
    #ball in the HSV color space
    #ballHSVMin = (0, 99, 191)
    #ballHSVMax = (21, 255, 255)
    ballHSVMin = (0, 151, 19)
    ballHSVMax = (9, 255, 255)
   
    #Values to be returned
    targetRadius = -1 #px
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

##        for contour in ballContours:
##            ((x, y), radius) = cv.minEnclosingCircle(contour)
##            if radius > minRadius:
##                cv.circle(imgRaw, (int(x), int(y)), int(radius), (0, 0, 255), 2)

        
        largestContour = max(ballContours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(largestContour)

        if radius > minRadius:

            targetRadius = radius
            targetX = x
            targetY = y
            foundBall = True

        #Distance and angle offset calculations
        if targetRadius > 0:
            
            inches_per_pixel = ballRadius/targetRadius #set up a general conversion factor
            distanceToBall = inches_per_pixel * (imgWidth / (2 * math.tan(math.radians(cameraFieldOfView))))
            offsetInInches = inches_per_pixel * (targetX - imgWidth / 2)
            angleToBall = math.degrees(math.atan((offsetInInches / distanceToBall)))
          
        else:
            
            distanceToBall = -1
            angleToBall = -1

    cv.circle(imgRaw, (int(targetX), int(targetY)), int(targetRadius), (0, 255, 0), 2)

    return targetX, targetY, targetRadius, distanceToBall, angleToBall, foundBall

    
#Define floor alignment tape detection method
def detect_floor_tape(imgRaw):
    
    #Define constraints for detecting floor tape
    floorTapeWidth = 2.0 #in inches
    floorTapeLength = 18.0 #in inches
    minTapeArea = 100 #in square px, can be tweaked if needed

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

        if cv.contourArea(largestContour) > minTapeArea:
            
            targetX, targetY, targetW, targetH = cv.boundingRect(largestContour)
            foundTape = True

    cv.rectangle(imgRaw,(targetX,targetY),(targetX+targetW,targetY+targetH),(100,0,255),1)

    return targetX, targetY, targetW, targetH, foundTape


#Define contour detector function
def detect_vision_targets(imgRaw):

    #Set constraints for detecting vision targets
    visionTargetWidth = 3.313 #in inches
    visionTargetHeight = 5.826 #in inches
    minTargetArea = 750 #in square px, for individual pieces of tape, calculated for viewing from ~4ft
    minRegionArea = 3200 #in square px, for paired pieces of tape, calculated for viewing from ~4ft

    #Define HSV range for cargo ship vision targets
    #values with light in Fab Lab
    visionTargetHSVMin = (90, 0, 86)
    visionTargetHSVMax = (124, 150, 248)
    #values from image testing
    #visionTargetHSVMin = (63, 0, 87)
    #visionTargetHSVMax = (108, 255, 255)

    #List to collect datapoints of all contours located
    #Append tuples in form (x, y, w, h)
    visionTargetValues = []

    #List to collect datapoints and area of all paired contours calculated
    #Append tuples in form (regionArea, x, y, w, h)
    visionRegionValues = []

    #Other processing values
    inchesPerPixel = -1
    diffTargets = -1
    
    #Values to be returned
    targetX = -1
    targetY = -1
    targetW = -1
    targetH = -1
    #targetArea = -1
    distanceToVisionTarget = -1
    angleToVisionTarget = float('nan') #default set to not-a-number
    foundVisionTarget = False

    #Find contours in mask
    visionTargetContours = process_image(imgRaw, visionTargetHSVMin, visionTargetHSVMax)
    
    #only continue if contours are found
    if len(visionTargetContours) > 0:
        
        #Loop over all contours
        for testContour in visionTargetContours:

            #Get bounding rectangle dimensions
            x, y, w, h = cv.boundingRect(testContour)

            #If large enough, draw a rectangle and store the values in the list
            if cv.contourArea(testContour) > minTargetArea:

                cv.rectangle(imgRaw,(x,y),(x+w,y+h),(0,0,255),2)

                visionTargetTuple = (x, y, w, h)
                visionTargetValues.append(visionTargetTuple)

##                #Compare contour with other contours to find ones that are 8 inches apart 
##                for testContour2 in visionTargetContours:
##
##                    #Get bounding rectangle
##                    x2, y2, w2, h2 = cv.boundingRect(testContour2)
##
##                    #Calculate the distance between the contours
##                    if x1 < x2:
##                        diffTargets = x2 - (x1 + w1)
##                    elif x2 < x1:
##                        diffTargets = x1 - (x2 + w2)
##                    else:
##                        diffTargets = -1
##                   
##                    #Check within a tolerance of the 8-inch known and set rectangle values properly
##                    if diffTargets * inchesPerPixel > 7.6 and diffTargets * inchesPerPixel < 8.4:
##
##                        if x2 <= x1:
##                            targetX = x2
##                        else:
##                            targetX = x1
##                        
##                        targetY = min(y1, y2)
##                        targetW = w1 + w2 + diffTargets
##                        targetH = min(h1, h2)
##                        targetArea = targetW * targetH
                        #visionTarget

        #Sort the contours found into a left-to-right order (sorting by x-value)
        visionTargetValues.sort(key=itemgetter(0))

        #Not sure if this will work properly, but an attempt to get the maximum height of the contours
        maxHeight = max(visionTargetValues, key=itemgetter(3))

        #Create a conversion factor between inches and pixels with a known value (the target height)
        #and the height of the tallest contour found
        inchesPerPixel = visionTargetHeight/maxHeight

        #Compare each contour to the next-right-most contour to determine distance between them
        for i in range(len(visionTargetValues) - 1):

            #Create a conversion factor between inches and pixels with a known value (the target height)
            #and the height of the left-most contour found
            #if i == 0:
            #    inchesPerPixel = visionTargetHeight/visionTargetValues[i][3]

            #Calculate the pixel difference between contours (right x - (left x + left width))
            diffTargets = visionTargetValues[i + 1][0] - (visionTargetValues[i][0] + visionTargetValues[i][2])
            print(diffTargets)

            #Check the distance against 8 inches with a tolerance, check the area, and store 
            #the matched pairs in the indices list
            if diffTargets * inchesPerPixel > 7.5 and diffTargets * inchesPerPixel < 8.5:

                #Calculate area of region found (height * (left width + right width + diffTargets))
                regionHeight = visionTargetValues[i][3] #using left height
                regionWidth = visionTargetValues[i][2] + visionTargetValues[i + 1][2] + diffTargets
                regionArea = regionWidth * regionHeight

                #Check area and draw rectangle (for testing)
                if regionArea > minRegionArea:

                    x = visionTargetValues[i][0]
                    y = visionTargetValues[i][1]
                    w = regionWidth
                    h = regionHeight
                    cv.rectangle(imgRaw,(x,y),(x+w,y+h),(0,0,255),2) 
                    
                    visionRegionValues.append(regionArea, x, y, w, h)
                    print ('Region found')

        #Sort the collected paired regions from largest area to smallest area (largest area is index 0)
        visionRegionValues.sort(key=itemgetter(0), reverse = True)

        #Assign final values to be returned
        targetX = visionRegionValues[0][1]
        targetY = visionRegionValues[0][2]
        targetW = visionRegionValues[0][3]
        targetH = visionRegionValues[0][4]

        foundVisionTarget = True
                        
        distanceToVisionTarget = inchesPerPixel * (imgWidth / (2 * math.tan(math.radians(cameraFieldOfView))))
        offsetInInches = inchesPerPixel * ((targetX + targetW/2) - imgWidth / 2)
        angleToVisionTarget = math.degrees(math.atan((offsetInInches / distanceToVisionTarget)))

        #Draw rectangle on image (for testing purposes)
        cv.rectangle(imgRaw,(targetX,targetY),(targetX+targetW+diffTargets,targetY+targetH),(0,255,0),2)  
                
    #Work in progress, needs to be cleaned somewhat.

    #Return results
    return targetX, targetY, targetW, targetH, distanceToVisionTarget, angleToVisionTarget, foundVisionTarget


#Define main processing function
def main():

    #Set up a camera server
    camserv = CameraServer.getInstance()
    camserv.enableLogging

    #Start capturing webcam video
    camera = camserv.startAutomaticCapture(dev=0, name="MainPICamera")
    camera.setResolution(imgWidth, imgHeight)

    #for prop in camera.enumerateProperties():
        #print(prop.getName())

    #print(camera.getBrightness())
    camera.setBrightness(0)

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
        #img = cv.imread('RetroreflectiveTapeImages2019/CargoStraightDark90in.jpg')
        #if img is None:
        #    break

        #OR grab an image from a video feed
        _, img = cvsink.grabFrame(img)

        #ballX, ballY, ballRadius, ballDistance, ballAngle, foundBall = detect_ball_target(img)
        #tapeX, tapeY, tapeW, tapeH, foundTape = detect_floor_tape(img)
        visionTargetX, visionTargetY, visionTargetW, visionTargetH, visionTargetDistance, visionTargetAngle, foundVisionTarget = detect_vision_targets(img)

        #Network table updating to go here

        #Draw various contours on the image
        #cv.circle(img, (int(ballX), int(ballY)), int(ballRadius), (0, 255, 0), 2) #ball
        #cv.rectangle(img,(tapeX,tapeY),(tapeX+tapeW,tapeY+tapeH),(100,0,255),1) #floor tape
        cv.rectangle(img,(visionTargetX,visionTargetY),(visionTargetX+visionTargetW,visionTargetY+visionTargetH),(0,255,0),2) #vision targets
        cv.putText(img, 'Distance to Vision: %.2f' %visionTargetDistance, (10, 400), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
        cv.putText(img, 'Angle to Vision: %.2f' %visionTargetAngle, (10, 440), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
        #cv.putText(img, 'Distance to Ball: %.2f' %ballDistance, (10, 50), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)

        #Check for stop code from robot or keyboard
        cv.imshow("Frame", img)
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
