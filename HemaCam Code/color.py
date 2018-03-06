# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:38:56 2018

@author: eshikasaxena

color histograms for each cell
"""
import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot as plt

global rootpath, imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgname = "0"

def calcHist(roi, filepath, count):
    roi_copy = roi.copy()
    cellgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cellthresh = thresh(roi, cellgray)
    cellmask = cellthresh
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([roi_copy],[i],cellmask,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    fig = plt.gcf()
    fig.savefig(rootpath + filepath + "_hist_{}.jpg".format(count))
    plt.clf()

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
        c = max(cnts, key=cv2.contourArea)
        if c.shape[0] >= 5:
            x,y,w,h = cv2.boundingRect(c)
            roi = clean[y:y+h, x:x+w]
            calcHist(roi)
#            cv2.imshow("p", cellthresh)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            contours.append(c)    
        count += 1

if __name__ == "__main__":
#    img, gray = img_load(imgnum)
#    clean = img.copy()
#    imgthresh = thresh(img, gray)
#    watershed_segmentation(clean, imgthresh, gray)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    img = cv2.imread('c40.png')
    calcHist(img)


