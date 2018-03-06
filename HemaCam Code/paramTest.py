# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 18:08:45 2018

@author: eshikasaxena
"""
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage



img = cv2.imread("C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\0073.jpg")
clean = img.copy()

#for sp in range(0, 50, 5):
#    for sr in range(0, 50, 5):
contours = []
img = cv2.pyrMeanShiftFiltering(img, 0, 0)
#        filtered = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\poster\\paramtest\\gray.jpg", gray)
#        blur = cv2.GaussianBlur(gray, (5,5),0)
#    blur = cv2.bilateralFilter(gray,5,75,75)
#    cv2.imshow("!", blur)
#    cv2.imwrite(f, blur)
threshVal = 200 #127
ret, thresh = cv2.threshold(gray, threshVal, 255,
                               cv2.THRESH_BINARY_INV
                               +cv2.THRESH_OTSU)
cv2.imwrite("C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\poster\\paramtest\\thresh.jpg", thresh)

#    cv2.imwrite(f2, threshold)
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
count = 1
for label in np.unique(labels):
    	# if the label is zero, we are examining the 'background'
    	# so simply ignore it
    if label == 0:
    		continue
    
    	# otherwise, allocate memory for the label region and draw
    	# it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
     
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    if c.shape[0] >= 5:
        contours.append(c)

cv2.drawContours(clean, contours, -1, (255,0,0), 3)
cv2.imwrite("C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\poster\\paramtest\\watershed_7.jpg", clean)
