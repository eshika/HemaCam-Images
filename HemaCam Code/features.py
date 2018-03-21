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

global rootpath, imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"

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


def calcFeatures(img, threshold, gray, filepath, imgname):
    contours = []
    clean = img.copy()
    D = ndimage.distance_transform_edt(threshold)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=threshold)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=threshold)
    write_csv(['File Name', 'Cell Image', 'Perimeter', 'Area', 'Circularity'], filepath + "_data")
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
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area > 100 and perimeter > 150 and area < 6000 and perimeter < 300:
                features = extractFeatures(c)           
# Original            
#            features = [imgname + "_{}".format(count) + ".png"] + [count] + features
 # MODIFIED           
#            features = [imgname + "_{}".format(count) + ".png"] + [ '<img src=' + '"..\\Cell_Images\\' + imgname + "_{}".format(count) + ".png" + '">'] + [count] + features
                features = [imgname + "_{}".format(count) + ".png"] + [ '<img src="' + imgname + "_{}".format(count) + ".png" + '">'] + features

# End Modified
            
                append_csv(features, filepath + "_data")
                roi = clean[y:y+h, x:x+w]
                contours.append(c)    
                count += 1
            
        
#    print num, count

if __name__ == "__main__":
    img, gray = img_load("0073")
    clean = img.copy()
    imgthresh = thresh(img, gray)
    calcFeatures(clean, imgthresh, gray)
