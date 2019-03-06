# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 18:34:43 2018

@author: eshikasaxena

cellSegmentation
cellCount
cellArea
boundingEllipse
boundingRect
color
features

"""

import numpy as np
import cv2
from HemaCamSegmentation import img_load, thresh
from realCellSegmentation import cellSegmentation
from cellArea import cellArea
from boundingEllipse import boundingEllipse
from cellCount import cellCount
from features import calcFeatures
from boundingRectangle import boundingRectangle

imgname = 'Sickle2'
filepath = "demo_out" + "\\" + imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\HemaCam-Data\\"
img = cv2.imread(rootpath + imgname + ".jpg")
cv2.imwrite(rootpath+filepath+".jpg", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imwrite(rootpath+filepath+"_gray.jpg", gray)
threshold = thresh(img, gray)
cv2.imshow("thresh", threshold)
cv2.imwrite(rootpath+filepath+"_thresh.jpg", threshold)
area = cellArea(img, threshold, gray, filepath)
ellipse = boundingEllipse(img, threshold, gray, filepath)
count = cellCount(img, threshold, gray, filepath)
cv2.imwrite(rootpath+filepath+"_count.jpg", count)
rect = boundingRectangle(img, threshold, gray, filepath)
cellSegmentation(img, threshold, gray, filepath)
#calcFeatures(img, threshold, gray, filepath, imgname)
cv2.imshow("out", np.hstack([ellipse, area, rect]))
cv2.waitKey(0)
cv2.destroyAllWindows()