# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:39:22 2017

@author: eshikasaxena

0: normal
1: sickle
2: cluster
"""
#import relevant libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import csv

global rootpath, imgnum
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
imgnum = 2

#loads and filters image from libary containing microscope images
def img_load(num):
#    filepath = rootpath + "Images\sickle{}.jpg".format(num)
    filepath = "01.jpg"
    img = cv2.imread(filepath)
    img = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

#thresholds image into binary to find contours
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
#returns features of each individual cell such as area, perimeter, axis, etc.
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

#method for manually labeling images
def manual_label(c, clean, target):
    x,y,w,h = cv2.boundingRect(c)
    roi = clean[y:y+h, x:x+w] 
    cv2.imshow("cell", roi)
    cv2.waitKey(50)
    tag = input("label: ")
    if tag == 0 or tag == 1 or tag == 2:
        target.append(tag)                       
        cv2.destroyAllWindows()

#uses watershed algorithm to find cells
def watershed_segmentation(img, thresh, gray, manual):
    master_data = []
    target = []
    clean = img.copy()
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=thresh)
    
    # perform a connected component analysis on the local peaks,
	# then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    
    for label in np.unique(labels):
        	# if the label is zero, it is the 'background'
        	# so we ignore it
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
            if manual:
               manual_label(c, clean,target)
            features = extractFeatures(c)
            master_data.append(features)
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

#initial test method using knn algorithm for machine learning
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
    
#    write_filename = 'newtest2'
#    load_data(imgnum, write_filename)
    
#    read_filename = 'data'
#    master_data, target = read_data(read_filename)
#    machine_learning(master_data, target)
    imgnum = 2
    img, gray = img_load(imgnum)
    clean = img.copy()
    imgthresh = thresh(img, gray)