# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:32:04 2018

@author: eshikasaxena
"""

import sys
import numpy as np
import cv2



def converter(blue,green,red): 
     
    color = np.uint8([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
     
    hue = hsv_color[0][0][0]
     
    print("Lower bound is :"),
    print("[" + str(hue-10) + ", 100, 100]\n")
     
    print("Upper bound is :"),
    print("[" + str(hue + 10) + ", 255, 255]")
    
    lower_bound = [(hue - 10), 100, 100]
    upper_bound = [(hue + 10), 255, 255]
    
    return lower_bound, upper_bound


def filter_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_bound, upper_bound = converter(36,28,237)
    
    lower_range = np.array(lower_bound, dtype=np.uint8)
    upper_range = np.array(upper_bound, dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
     
    cv2.imshow('mask',mask)
    cv2.imshow('image', img)
     
if __name__ == "__main__":
    img = cv2.imread('circles.png', 1)
    filter_color(img)
    while(1):
      k = cv2.waitKey(0)
      if(k == 0):
        break
     
    cv2.destroyAllWindows()

