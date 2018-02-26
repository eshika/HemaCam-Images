# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:26:42 2018

@author: eshikasaxena

write features to csv
"""
import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh, extractFeatures
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import csv

global rootpath, imgnum
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgnum = 100

def write_csv(data, filename):
    with open(rootpath + filename + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
#        for x in data:
#            writer.writerow(x)

def append_csv(data, filename):
    with open(rootpath + filename + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
#        for x in data:
#            writer.writerow(x)

def watershed_segmentation(img, threshold, gray):
    contours = []
    clean = img.copy()
    D = ndimage.distance_transform_edt(threshold)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=threshold)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=threshold)
    write_csv(['Cell Number', 'Perimeter', 'Area', 'Circularity', 'Major Axis', 'Minor Axis', 'Ratio'], 'data_{}'.format(imgnum))
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
            features = extractFeatures(c)
            features = [count] + features
            append_csv(features, 'data_{}'.format(imgnum))
            roi = clean[y:y+h, x:x+w]
            contours.append(c)    
            count += 1
            
        
#    print num, count

if __name__ == "__main__":
    img, gray = img_load(imgnum)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    watershed_segmentation(clean, imgthresh, gray)