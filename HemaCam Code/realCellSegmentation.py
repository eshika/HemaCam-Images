# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:39:22 2017

@author: eshikasaxena

0: normal
1: sickle
2: cluster
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:29:41 2018

@author: eshikasaxena

labels with area
"""
import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from color import calcHist

global rootpath, imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgname = "0057"



def cellSegmentation(img, threshold, gray, filepath):
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
            roi = clean[y:y+h, x:x+w]
            cv2.imwrite(rootpath + filepath + "_{}.png".format(count), roi)
            calcHist(roi, filepath, count)
            contours.append(c)    
        count += 1

#    print num, count

if __name__ == "__main__":
    img, gray = img_load(imgname)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    cellSegmentation(clean, imgthresh, gray)

