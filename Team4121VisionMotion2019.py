#--------------------------------------------------#
#         North Canton Hoover High School          #
#                                                  #
#          Team 4121 - Norsemen Robotics           #
#                                                  #
#         Vision & Motion Processing Code          #
#--------------------------------------------------#
#                                                  #
#  This code continuously analyzes images from     #
#  one or more USB cameras to identify on field    #
#  game pieces and vision targets.  For game       #
#  pieces, the code will identify all game         #
#  pieces within the camera's field of view and    #
#  determine the closest one.  The distance and    #
#  angle to the closest game piece is calculated   #
#  and made available to the main robot code       #
#  through network tables.  The closest game piece #
#  is highlighted with a green box while all other #
#  found game pieces are highlighted with a red    #
#  box.  The annotated video is streamed to the    #
#  driver station for display.  The annotated      #
#  video is also saved to a file for post game     #
#  review and analysis.  For vision targets, the   #
#  code will identify all vision targets and       #
#  calculate the angle and distance to each one.   #
#  Vision target information is made available to  #
#  the main robot code through network tables.     #
#                                                  #
#  This code also continuously interrogates a      #
#  VMX-Pi board to determine linear and angular    #
#  motion in all three axes.  This information is  #
#  made available to the main robot code through   #
#  network tables.                                 #
#                                                  #
#--------------------------------------------------#
#                                                  #
#  Authors:  Jonas Muhlenkamp                      #
#            Ricky Park                            #
#            Tresor Nshimiye                       #
#            Tim Fuller                            #
#                                                  #
#  Creation Date: 3/1/2018                         #
#                                                  #
#  Revision: 3.0                                   #
#                                                  #
#  Revision Date: 2/18/2019                        #
#                                                  #
#--------------------------------------------------#

#!/usr/bin/env python3

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
from networktables import NetworkTables
from time import sleep

#Set up basic logging
logging.basicConfig(level=logging.DEBUG)

#Initialize operating constants
imgWidthVision = 640  
imgHeightVision = 480
imgWidthDriver = 160
imgHeightDriver = 120
cameraFieldOfView = 27.3
framesPerSecond = 30

#Define program control flags
writeVideo = True

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
    ballHSVMin = (0, 115, 220)
    ballHSVMax = (18, 255, 255)
   
    #Values to be returned
    targetRadius = 0 #px
    targetX = -1 #px
    targetY = -1 #px
    distanceToBall = -1 #inches
    angleToBall = -1 #degrees
    screenPercent = -1
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
        if targetRadius > 0:
            
            inches_per_pixel = ballRadius/targetRadius #set up a general conversion factor
            distanceToBall = inches_per_pixel * (imgWidthVision / (2 * math.tan(math.radians(cameraFieldOfView))))
            offsetInInches = inches_per_pixel * (targetX - imgWidthVision / 2)
            angleToBall = math.degrees(math.atan((offsetInInches / distanceToBall)))
            screenPercent = cv.contourArea(largestContour) / (imgWidthVision * imgHeightVision)
          
        else:
            
            distanceToBall = -1
            angleToBall = float('nan')

    return targetX, targetY, targetRadius, distanceToBall, angleToBall, screenPercent, foundBall

    
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
    centerOffset = -1
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

            #calculate center offset of tape
            centerOffset = (imgWidthVision / 2) - (tapeX + (tapeW / 2))

    return targetX, targetY, targetW, targetH, centerOffset, foundTape


#Define contour detector function
def detect_vision_targets(imgRaw):

    #Set constraints for detecting vision targets
    visionTargetWidth = 3.313 #in inches
    visionTargetHeight = 5.826 #in inches
    minTargetArea = 750 #in square px, for individual pieces of tape, calculated for viewing from ~4ft
    minRegionArea = 3200 #in square px, for paired pieces of tape, calculated for viewing from ~4ft

    #Define HSV range for cargo ship vision targets
    #values with light in Fab Lab
    visionTargetHSVMin = (93, 63, 70)
    visionTargetHSVMax = (255, 255, 255)
    #values from image testing
    #visionTargetHSVMin = (63, 0, 87)
    #visionTargetHSVMax = (108, 255, 255)

    #List to collect datapoints of all contours located
    #Append tuples in form (x, y, w, h, a)
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
            rect = cv.minAreaRect(testContour)
            a = rect[2]
            #print (rect[2])
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(imgRaw,[box],0,(0,0,255),2)            

            #If large enough, draw a rectangle and store the values in the list
            if cv.contourArea(testContour) > minTargetArea:

                #cv.rectangle(imgRaw,(x,y),(x+w,y+h),(0,0,255),2)

                visionTargetTuple = (x, y, w, h, a)
                visionTargetValues.append(visionTargetTuple)

        #Only continue if two appropriately sized contours were found
        if len(visionTargetValues) > 1:

            #Sort the contours found into a left-to-right order (sorting by x-value)
            visionTargetValues.sort(key=itemgetter(0))

            #Compare each contour to the next-right-most contour to determine distance between them
            for i in range(len(visionTargetValues) - 1):

                #Create a conversion factor between inches and pixels with a known value (the target height)
                #and the height of the left-most contour found
                inchesPerPixel = visionTargetHeight/visionTargetValues[i][3]
                
                #Calculate the pixel difference between contours (right x - (left x + left width))
                diffTargets = visionTargetValues[i + 1][0] - (visionTargetValues[i][0] + visionTargetValues[i][2])
                
                #Check the distance against the expected angle with a tolerance, check the area, and store 
                #the matched pairs in the indices list
                if visionTargetValues[i][4] < -65 and visionTargetValues[i+1][4] > -25:

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
                        cv.rectangle(imgRaw,(x,y),(x+w,y+h),(0,0,255),1) 
                        
                        visionRegionTuple = (regionArea, x, y, w, h)
                        visionRegionValues.append(visionRegionTuple)
                        
            #Only proceed if an appropriately sized merged region is found
            if len(visionRegionValues) > 0:

                #Sort the collected paired regions from largest area to smallest area (largest area is index 0)
                visionRegionValues.sort(key=itemgetter(0), reverse = True)

                #Assign final values to be returned
                targetX = visionRegionValues[0][1]
                targetY = visionRegionValues[0][2]
                targetW = visionRegionValues[0][3]
                targetH = visionRegionValues[0][4]

                foundVisionTarget = True
                                
                distanceToVisionTarget = inchesPerPixel * (imgWidthVision / (2 * math.tan(math.radians(cameraFieldOfView))))
                offsetInInches = inchesPerPixel * ((targetX + targetW/2) - imgWidthVision / 2)
                angleToVisionTarget = math.degrees(math.atan((offsetInInches / distanceToVisionTarget)))

    #Return results
    return targetX, targetY, targetW, targetH, distanceToVisionTarget, angleToVisionTarget, foundVisionTarget


#Define main processing function
def main():

    global imgWidthDriver
    global imgHeightDriver
    global imgWidthVision
    global imgHeightVision
    global framesPerSecond

    #define local variables
    driverCameraBrightness = 50
    visionCameraBrightness = 25

    #Get current time as a string
    currentTime = time.localtime(time.time())
    timeString = str(currentTime.tm_year) + str(currentTime.tm_mon) + str(currentTime.tm_mday) + str(currentTime.tm_hour) + str(currentTime.tm_min)

    #open a log file
    logFilename = '/data/Logs/Run_Log_' + timeString + '.txt'
    log_file = open(logFilename, 'w')
    log_file.write('run started on %s.\n' % datetime.datetime.now())
    log_file.write('')

    #Connect NetworkTables
    #NetworkTables.initialize(server='10.41.21.2')
    #visionTable = NetworkTables.getTable("vision")
    #smartDash = NetworkTables.getTable("SmartDashboard")

    #Set up a camera server
    camserv = CameraServer.getInstance()
    camserv.enableLogging

    #Start capturing webcam video
    driverCamera = camserv.startAutomaticCapture(dev=0, name = "DriverCamera")
    driverCamera.setResolution(imgWidthDriver, imgHeightDriver)
    driverCamera.setBrightness(driverCameraBrightness)
    visionCamera = camserv.startAutomaticCapture(dev=1, name="VisionCamera")
    visionCamera.setResolution(imgWidthVision, imgHeightVision)
    visionCamera.setBrightness(visionCameraBrightness)

    #Define video sink
    driverSink = camserv.getVideo(name = 'DriverCamera')
    visionSink = camserv.getVideo(name = 'VisionCamera')

    #Create an output video stream
    driverOutputStream = camserv.putVideo("MainCamera", imgWidthDriver, imgHeightDriver)

    #Create blank image
    imgDriver = np.zeros(shape=(imgWidthDriver, imgHeightDriver, 3), dtype = np.uint8)
    imgVision = np.zeros(shape=(imgWidthVision, imgHeightVision, 3), dtype = np.uint8)

    #Set video codec and create VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    videoFilename = '/data/Match_Videos/RobotVisionCam-' + timeString + '.avi'
    visionImageOut = cv.VideoWriter(videoFilename,fourcc,20.0,(imgWidthVision,imgHeightVision))

    #Create network table entries here
##    ballX
##    ballY
##    ballRadius
##    ballDistance
##    ballAngle
##    ballScreenPercent
##    foundBall
##
##    tapeX
##    tapeY
##    tapeW
##    tapeH
##    tapeOffset
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
    imgDriver= np.zeros(shape=(imgWidthDriver, imgHeightDriver, 3), dtype=np.uint8)
    imgVision= np.zeros(shape=(imgWidthVision, imgHeightVision, 3), dtype=np.uint8)

    #Start main processing loop
    while (True):

        #Read in an image from 2019 Vision Images (for testing)
        #img = cv.imread('RetroreflectiveTapeImages2019/CargoStraightDark90in.jpg')
        #if img is None:
        #    break

        #Grab frames from the web cameras
        driverVideoTimestamp, imgDriver = driverSink.grabFrame(imgDriver)
        visionVideoTimestamp, imgVision = visionSink.grabFrame(imgVision)

        #Check for frame errors
        visionFrameGood = True
        if (driverVideoTimestamp == 0) or (visionVideoTimestamp == 0):
            if driverVideoTimestamp == 0:
                log_file.write('Driver video error: \n')
                log_file.write(driverSink.getError())
                log_file.write('\n')
            else:
                log_file.write('Vision video error: \n')
                log_file.write(visionSink.getError())
                log_file.write('\n')
                visionFrameGood = False
            sleep (float(framesPerSecond * 2) / 1000.0)
            continue    
        
        if (visionFrameGood == True):
            ballX, ballY, ballRadius, ballDistance, ballAngle, ballScreenPercent, foundBall = detect_ball_target(imgVision)
            #tapeX, tapeY, tapeW, tapeH, tapeOffset, foundTape = detect_floor_tape(img)
            #visionTargetX, visionTargetY, visionTargetW, visionTargetH, visionTargetDistance, visionTargetAngle, foundVisionTarget = detect_vision_targets(imgVision)

            #Vision network table updating to go here

            #Draw various contours on the image
            cv.circle(imgVision, (int(ballX), int(ballY)), int(ballRadius), (0, 255, 0), 2) #ball
            #cv.rectangle(img,(tapeX,tapeY),(tapeX+tapeW,tapeY+tapeH),(100,0,255),1) #floor tape
            #cv.rectangle(imgVision,(visionTargetX,visionTargetY),(visionTargetX+visionTargetW,visionTargetY+visionTargetH),(0,255,0),2) #vision targets
            #cv.putText(imgVision, 'Distance to Vision: %.2f' %visionTargetDistance, (10, 400), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
            #cv.putText(imgVision, 'Angle to Vision: %.2f' %visionTargetAngle, (10, 440), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 255, 0), 2)
            cv.putText(imgVision, 'Distance to Ball: %.2f' %ballDistance, (320, 400), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 0, 255), 2)
            cv.putText(imgVision, 'Angle to Ball: %.2f' %ballAngle, (320, 440), cv.FONT_HERSHEY_SIMPLEX, .75,(0, 0, 255), 2)
            cv.putText(imgVision, str(datetime.datetime.now()), (10, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

        #Update navx network table here

        #Send driver camera to dashboard
        driverOutputStream.putFrame(imgDriver)

        #Write processed image to file
        if writeVideo:
            visionImageOut.write(imgVision)

        #Display the two camera streams (for testing only)
        cv.imshow("Vision", imgVision)
        cv.imshow("Driver", imgDriver)
        
        #Check for stop code from robot or keyboard (for testing)
        if cv.waitKey(1) == 27:
            break
        #robotStop = visionTable.getNumber("RobotStop", 0)
        #if robotStop == 1:
        #    break


    #Close all open windows (for testing)
    cv.destroyAllWindows()

    #Close video file
    visionImageOut.release()

    #Close the log file
    log_file.write('Run stopped on %s.' % datetime.datetime.now())
    log_file.close()
    

#define main function
if __name__ == '__main__':
    main()
