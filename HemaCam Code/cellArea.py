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

global rootpath, imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\HemaCam-Data\\"
imgname = "0057"



def cellArea(img, threshold, gray, filepath):
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
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])   
            area = M["m00"]
            perimeter = cv2.arcLength(c, True)
            if area > 1500 and perimeter > 150 and area < 6000 and perimeter < 300:
#            cv2.circle(clean, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(clean, "{}".format(area), (cX-10, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
                roi = clean[y:y+h, x:x+w]
                contours.append(c)    
                count += 1
#    cv2.imshow("3", clean)
    cv2.drawContours(clean, contours, -1, (255,0,0), 3)
    cv2.imwrite(rootpath + filepath + "_area.jpg", clean)
    return clean
#    cv2.imshow("2", clean)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    print num, count

if __name__ == "__main__":
    img, gray = img_load(imgname)
    clean = img.copy()
    imgthresh = thresh(img, gray)
    cellArea(clean, imgthresh, gray)
