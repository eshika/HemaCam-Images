# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:29:41 2018

@author: eshikasaxena
"""
import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

global rootpath, imgnum
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgnum = 1000



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
        cv2.drawContours(clean, cnts, -1, (255,0,0), 3)
        cv2.imshow("2", clean)
        c = max(cnts, key=cv2.contourArea)
        if c.shape[0] >= 5:
            x,y,w,h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])            
#            cv2.circle(clean, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(clean, "{}".format(count), (cX-5, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            roi = clean[y:y+h, x:x+w]
            contours.append(c)    
        count += 1

#    print num, count

if __name__ == "__main__":
    img, gray = img_load(imgnum)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    cv2.imshow("1", imgthresh)
    watershed_segmentation(clean, imgthresh, gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
