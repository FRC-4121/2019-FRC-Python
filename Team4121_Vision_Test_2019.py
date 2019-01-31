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

from time import sleep

#Set up basic logging
logging.basicConfig(level=logging.DEBUG)

#Set image variables
imgwidth = 320
imgheight = 240
fov_v = 42


#Define contour detector function
def detect_contours(img_raw):

    #Set global variables
    global imgwidth
    global imgheight
    
    #Set known values
    FOV_angle_in_degrees = 27.3
    width_of_target = 3.188
    
    #Set object test values
    minarea = 200


    #Initialize some processing variables
    targetH = 0
    targetW = 0
    targetX = 0
    targetY = 0
    img_contours = cv.rectangle(img_raw,(0,0),(0,0),(0,0,255),2)
    #n Target color in HSV
    targetMin = (63, 0, 87)
    targetMax = (108, 255, 255)
    #Blur image to remove noise
    blur = cv.GaussianBlur(img_raw.copy(),(5,5),0)
        
    #Convert from BGR to HSV colorspace
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    #Define first range of Visio

    #Set pixels to white if in target HSV range, else set to black
    mask = cv.inRange(hsv, targetMin, targetMax)

    #Find contours in mask
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #Loop over all contours to find valid targets based on color range 1
    for testContour in contours:

        #Get area of contour
        area = cv.contourArea(testContour)

        #Check area before further processing
        if area > minarea:

            #Get bounding rectangle
            targetX, targetY, targetW, targetH = cv.boundingRect(testContour)
            
            #Draw rectangle on image
            img_contours = cv.rectangle(img_raw,(targetX,targetY),(targetX+targetW,targetY+targetH),(0,0,255),1)

    #return results
    return img_contours


#define main processing function
def mainloop():

    #Read in an image from 2019 Vision Images
    imgFilename = '2019VisionImages\LoadingAngle36in.jpg'
    img = cv.imread(imgFilename)

    #Start main processing loop
    while (True):

        #Find contours in image
        img_contours = detect_contours(img)
       
        #Show image
        cv.imshow('Target Image', img_contours)

        #Check for stop code from robot
        if cv.waitKey(1) == 27:
            break

    #Close all open windows
    cv.destroyAllWindows()
    

#define main function
def main():
    mainloop()

if __name__ == '__main__':
    main()
