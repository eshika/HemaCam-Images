# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:39:22 2017

@author: eshikasaxena

9 image dataset:
4,5,6: normal train
40,42,352: sickle train
31,32: normal test
300: sickle test

0: normal
1: sickle
2: other\
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import csv
import math

global rootpath, imgnum
#rootpath = "C:\Users\eshikasaxena\OneDrive\Documents\Github\Sickle Cell"
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgnum = 2

def img_load(num):
#    filepath = rootpath + "\Images\\new\s{}.jpg".format(num)
    filepath = rootpath + "{}.jpg".format(num)
    img = cv2.imread(filepath)
    img = cv2.pyrMeanShiftFiltering(img, 21, 25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def thresh(img, gray):
    filepath = rootpath  
    blur = cv2.GaussianBlur(gray, (5,5),0)
#    f = filepath + "\Images/blur{}.png".format(imgnum)
#    cv2.imwrite(f, blur)
    threshVal = 200 #127
    
    ret, threshold = cv2.threshold(blur, threshVal, 255,
                                   cv2.THRESH_BINARY_INV
                                   +cv2.THRESH_OTSU)
#    f2 = filepath + "\Images/thresh{}.png".format(imgnum)                                   
#    cv2.imwrite(f2, threshold)
    x,contours, hierarchy = cv2.findContours(threshold.copy(),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(img, contours, -1, (255,0,0), cv2.FILLED)
#    f3 = filepath + "\Images\contour{}.png".format(imgnum)                                   
#    cv2.imwrite(f3, img)
    new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new = cv2.bitwise_not(new)
    ret2, new = cv2.threshold(new, threshVal, 255,
                                   cv2.THRESH_BINARY
                                   +cv2.THRESH_OTSU)                       
#    f4 = filepath + "\Images/newthresh{}.png".format(imgnum)                                   
#    cv2.imwrite(f4, new)  
    thresh = new
    return thresh

def extractFeatures(c):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    compactness = (perimeter**2/area) - 1.0
    ellipse = cv2.fitEllipse(c)
#    cv2.ellipse(out, ellipse, (0,255,0),2)
    x = int(ellipse[0][0])
    y = int(ellipse[0][1])
    center = (x,y)
    width = int(ellipse[1][0])
    height = int(ellipse[1][1])
    r1 = width/2
    r2 = height/2
#    cv2.imshow("mask", new)    
#    mean, stddev = cv2.meanStdDev(gray, new)
#    print stddev
    features = [r1, r2]
    return features

def manual_label(c, clean, target):
    x,y,w,h = cv2.boundingRect(c)
    roi = clean[y:y+h, x:x+w] 
    cv2.imshow("cell", roi)
    cv2.waitKey(50)
    tag = input("label: ")
    if tag == 0 or tag == 1 or tag == 2:
        target.append(tag)                       
        cv2.destroyAllWindows()

def watershed_segmentation(num, img, thresh, gray, manual, outline):
    master_data = []
    target = []
    clean = img.copy()
    cleanContours = np.ones((1000,1000, 3)) * 255
#    cv2.imshow('clean', clean)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=thresh)
    
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
        cleanContours = cv2.drawContours(cleanContours, cnts, -1, (255,0,0), 1)
        c = max(cnts, key=cv2.contourArea)
        if c.shape[0] >= 5:
            if manual:
               manual_label(c, clean,target)
#            features = extractFeatures(c)
#            master_data.append(features)
            x,y,w,h = cv2.boundingRect(c)
            if outline:
                roi = cleanContours[y:y+h, x:x+w]
                height = np.size(roi, 0)
                width = np.size(roi, 1)
    #            cv2.fillPoly(img, pts =[contours], color=(255,255,255))
                print (width, height)
                print ((128-w)/2., (128-h)/2.)
                top = int(math.ceil((128-h)/2.))
                bottom = int(math.floor((128-h)/2.))
                left = int(math.ceil((128-w)/2.))
                right = int(math.floor((128-w)/2.))
                roi = cv2.copyMakeBorder(roi,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(255,255,255))
                cv2.imwrite(rootpath + "\Images\cells_new\{0}_c{1}.png".format(num, count), roi)
            else:
                roi = clean[y:y+h, x:x+w]
#                cv2.imwrite(rootpath + "\Images\cells_out\{0}_c{1}.png".format(num, count), roi)
                cv2.imwrite(rootpath + "\output\{0}_c{1}.png".format(num, count), roi)
        count += 1
#    print num, count
    return master_data, target

def write_csv(data, target, filename):
    with open(rootpath + filename + '.csv', 'w') as f:
        writer = csv.writer(f)
        for x in data:
            writer.writerow(x)
        writer.writerow(target)

def append_csv(data, target, filename):
    with open(rootpath + filename + '.csv', 'a') as f:
        writer = csv.writer(f)
        for x in data:
            writer.writerow(x)
        writer.writerow(target)
   
def read_csv(filename):    
    with open(rootpath + filename + '.csv', 'r') as f:
        reader = csv.reader(f)
        datalist = []
        datalist = list(reader)
        datalist = [list(map(int, row)) for row in datalist]

        target = []
        target = datalist[len(datalist)-1]
        target = [int(x) for x in target]
        
        del datalist[len(datalist)-1]
        return datalist, target

def load_data(imgnum, filename):
    img, gray = img_load(imgnum)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    master_data, target = watershed_segmentation(clean, imgthresh, gray, True)
    write_csv(master_data, target, filename)

def read_data(filename): 
    datalist, target = read_csv(filename)
    return datalist, target

def machine_learning(datalist, target):
    train, test, train_labels, test_labels = train_test_split(datalist,
                                                              target,
                                                              test_size=0.50,
                                                              random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train, train_labels)
    
    print (test_labels)
    print (clf.predict(test))
    print (clf.score(train, train_labels))
    print (clf.score(test, test_labels))           

if __name__ == "__main__":
    num = 1
    img, gray = img_load(num)
    clean = img.copy()
    threshold = thresh(img, gray)
    watershed_segmentation(num, clean, threshold, gray, False, False)
    
#    write_filename = 'newtest2'
#    load_data(imgnum, write_filename)
    
#    read_filename = 'data'
#    master_data, target = read_data(read_filename)
#    machine_learning(master_data, target)
#    img, gray = img_load(imgnum)
#    clean = img.copy()
#    imgthresh = thresh(img, gray)
    