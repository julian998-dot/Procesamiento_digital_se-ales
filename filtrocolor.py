# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:58:28 2020

@author: julia
"""

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([179, 255, 255], np.uint8)
while True:
  ret,frame = cap.read()
  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)
    maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed) 
    src = maskRed
    circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20,param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # dibujar circulo 
        cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
        # dibujar centro
        cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)       
    cv2.imshow('frame', frame)
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('maskRedvis', maskRedvis)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()
