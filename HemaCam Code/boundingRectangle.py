# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:11:01 2018

@author: eshikasaxena

bounding rectangle
"""
import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

global rootpath, imgnum
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgnum = 58



def watershed_segmentation(img, threshold, gray):
    contours = []
    clean = img.copy()
    D = ndimage.distance_transform_edt(threshold)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=threshold)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=threshold)
    count = 1
    for label in np.unique(labels):
        if label == 0:
        		continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255         
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        if c.shape[0] >= 5:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(clean, (x,y), (x+w, y+h), (0,0,0), 2)
            roi = clean[y:y+h, x:x+w]
            contours.append(c)    
        count += 1
    cv2.imshow("3", clean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#    print num, count

if __name__ == "__main__":
    img, gray = img_load(imgnum)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    watershed_segmentation(clean, imgthresh, gray)

