# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:58:28 2020

@author: julia
"""

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
redBajo1 = np.array([135, 100, 0], np.uint8)
redAlto1 = np.array([150, 255, 255], np.uint8)
redBajo2=np.array([151, 100, 0], np.uint8)
redAlto2=np.array([165, 255, 255], np.uint8)

bBajo1 = np.array([85, 100, 20], np.uint8)
bAlto1 = np.array([86, 255, 255], np.uint8)
bBajo2=np.array([87, 100, 20], np.uint8)
bAlto2=np.array([95, 255, 255], np.uint8)
while True:
  ret,frame = cap.read()
  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)
    maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed) 

    maskBlue1= cv2.inRange(frameHSV,bBajo1, bAlto1)
    maskBlue2= cv2.inRange(frameHSV,bBajo2, bAlto2)
    maskBlue= cv2.add(maskBlue1,maskBlue2)
    maskBluevis = cv2.bitwise_and(frame, frame, mask= maskBlue) 

    bordes = cv2.Canny(maskRed, 100, 200) 
    _, ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctns, -1, (0,0,255), 2)
    lines1=cv2.HoughLinesP(bordes,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines1:
        x1, y1, x2, y2 = line[0]
        cv2.line(maskRedvis, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
    bordes2 = cv2.Canny(maskBlue, 100, 200) 
    lines2=cv2.HoughLinesP(bordes2,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines2:
        x1, y1, x2, y2 = line[0]
        cv2.line(maskBluevis, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
    _, ctns2, _ = cv2.findContours(bordes2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctns2, -1, (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('maskRedvis', maskRedvis)
    cv2.imshow('maskBlue', maskBlue)
    cv2.imshow('maskBluevis', maskBluevis)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()


