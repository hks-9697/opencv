import cv2
import matplotlib.pyplot as plot
import numpy as num
img1=cv2.imread('image1.jpeg',0)
img2=cv2.imread('image2.jpg',0)
ret,mask=cv2.threshold(img2,150,255,cv2.THRESH_BINARY_INV)
ret1,mask1=cv2.threshold(img2,70,255,cv2.THRESH_BINARY_INV)
img3=cv2.addWeighted(mask,0.7,mask1,0.3,0)
cv2.imshow('mask',mask)
cv2.imshow('image2',img2)
cv2.imshow('merge',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
		
