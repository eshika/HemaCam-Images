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

imgname = 'Best3'
filepath = "poster\\" + imgname + "\\" + imgname
rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
print (filepath)
img, gray = img_load(imgname)
cv2.imwrite(rootpath + filepath + ".jpg", img)
cv2.imwrite(rootpath + filepath + "_gray.jpg", gray)
threshold = thresh(img, gray)
cv2.imwrite(rootpath + filepath + "_thresh.jpg", threshold)
cellArea(img, threshold, gray, filepath)
boundingEllipse(img, threshold, gray, filepath)
cellCount(img, threshold, gray, filepath)
boundingRectangle(img, threshold, gray, filepath)
cellSegmentation(img, threshold, gray, filepath)
calcFeatures(img, threshold, gray, filepath)
