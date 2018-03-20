# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:32:51 2018

@author: eshikasaxena
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
import glob
import os
import pandas as pd




filepath = '..\\HemaCam-Data\\Segmented_Cells\\Cell_Images\\' 
filepath2 = '..\\HemaCam-Data\\Segmented_Cells\\Cell_Properties\\' 


def pipeline(): 
    for img in glob.glob('..\\HemaCam-Data\\Input_Images\\*.jpg'):
        image = cv2.imread(img)
        imgname, extension = os.path.basename(img).split(".")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = thresh(image, gray)
        cellSegmentation(image, threshold, gray, filepath + imgname)
        calcFeatures(image, threshold, gray, filepath2 + imgname, imgname)
        
#        csvInpfile = filepath2 + imgname + "_data.csv"
#        htmlOutfile = filepath2 + imgname + "_data.html"
#        df = pd.read_csv(csvInpfile)
#        file = open(htmlOutfile, 'w')
#        file.write(df.to_html(justify='center', escape = False))


if __name__ == "__main__":
    pipeline()

#imgname = '..\\HemaCam-Data\\Input_Images'
#filepath = "demo\\" + imgname + "\\" + imgname
#rootpath = "C:\\Users\\eshikasaxena\\Desktop\\HemaCam Project\\Code\\"
#img, gray = img_load(imgname)
#cv2.imwrite(rootpath + filepath + ".jpg", img)
#cv2.imwrite(rootpath + filepath + "_gray.jpg", gray)
#threshold = thresh(img, gray)
#cv2.imwrite(rootpath + filepath + "_thresh.jpg", threshold)
#area = cellArea(img, threshold, gray, filepath)
#boundingEllipse(img, threshold, gray, filepath)
#cellCount(img, threshold, gray, filepath)
#boundingRectangle(img, threshold, gray, filepath)
#cellSegmentation(img, threshold, gray, filepath)
#calcFeatures(img, threshold, gray, filepath)
#cv2.imshow("out", np.hstack((img, area)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()