#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:55:06 2017

@author: DK
"""

### Intersection of 2 images ###

# CASE 1: when the images are in arrays
import numpy as np

def intersection_v1(array1, array2):
    a = np.array(array1)
    b = np.array(array2)
    
    c = a&b     #intersection
    d = a^b     #non intersection
    
    return c
    

# CASE2: when coordinates of the images are known
import math
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def intersection_v2(image1, image2):
    dx = min(image1.xmax, image2.xmax) - max(image1.xmin, image2.xmin)
    dy = min(image1.ymax, image2.ymax) - max(image1.ymin, image2.ymin)
    
    if dx>0 and dy>0:
        print(dx*dy)