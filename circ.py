












"""
Created on Thu Nov  5 21:15:06 2020

@author: julia
"""
import cv2
import numpy as np

img = cv2.imread('prueba.jpg')
src = cv2.medianBlur(img, 5)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # dibujar circulo 
    cv2.circle(img, (i[0], i[1]), i[2], (0,255,0), 2)
    # dibujar centro
    cv2.circle(img, (i[0], i[1]), 2, (0,0,255), 3)

cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

