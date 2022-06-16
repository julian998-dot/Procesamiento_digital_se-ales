# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:47:15 2020

@author: Julian
"""
import numpy as np
import cv2
 
imgpr2 = cv2.imread('C:\Users\Julian\Documents\UMNG\VII-VIII\Señales\Tercer Corte\sample1.png')
clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

lab = cv2.cvtColor(imgpr2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv2.split(lab)  # split on 3 different channels

l2 = clahe.apply(l)  # apply CLAHE to the L-channel

lab = cv2.merge((l2,a,b))  # merge channels
img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
#cv2.imshow('Increased contrast', img2)
##cv2.imwrite('sunset_modified.jpg', img2)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
dst = cv2.fastNlMeansDenoisingColored(img2,None,10,10,7,21)
newfiltr = cv2.GaussianBlur(img2,(11,11),7)

# Cargamos la imagen
# Detectamos los bordes con Canny
canny = cv2.Canny(newfiltr,50,150)
canny2 = cv2.Canny(dst,50,500)
# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(contornos2,_) = cv2.findContours(canny2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))
print("He encontrado {} objetos".format(len(contornos2)))
 
borde = cv2.drawContours(newfiltr,contornos,-1,(0,0,255),5)
borde2 = cv2.drawContours(dst,contornos2,-1,(0,255,0),5)
cv2.imwrite('pb2.png',borde)
cv2.imwrite('pb3.png',borde2)

