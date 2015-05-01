#!/user/bin/env python
from __future__ import absolute_import
import cv2

def resize_image(im, desired_size=1000):
    biggest_dim = max(im.shape)
    scale = float(desired_size) / float(biggest_dim)
    return cv2.resize(im, (0,0), fx=scale, fy=scale), scale
