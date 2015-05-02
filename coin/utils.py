#!/user/bin/env python
import cv2

import numpy as np
import matlab
import os.path as op
import matlab
import matplotlib.pyplot as plt
import coin.segmenter
from matplotlib.patches import Rectangle
from matlab import engine

def load_matlab_util(matlab_engine):
    ml_path = op.join(op.dirname(op.realpath(__file__)), 'matlab')
    matlab_engine.addpath(matlab_engine.genpath(ml_path))

def resize_image(im, desired_size=1000):
    biggest_dim = max(im.shape)
    scale = float(desired_size) / float(biggest_dim)
    return cv2.resize(im, (0,0), fx=scale, fy=scale), scale

def create_pad_mask(im, padding=50):
    h, w = im.shape
    np.zeros((h, w), )
    return np.pad(np.zeros((h - (padding * 2), w - (padding * 2))), padding, mode='constant', constant_values=1)

def prepare_mask_for_matlab(mask):
    if np.max(mask) == 1:
        mask = mask*255
    ml_mask = matlab.uint8(mask.flatten('F').astype('uint8').tolist())
    ml_mask.reshape(mask.shape)
    return ml_mask

def draw_bounding_boxes(img, centers, radii):
    _, subplt = plt.subplots(1,1)
    subplt.imshow(img)
    subplt.scatter(centers[:,0], centers[:,1])
    for i in range(radii.shape[0]):
        x = centers[i][0]
        y = centers[i][1]
        r = radii[i][0]
        subplt.add_patch(Rectangle((x - r, y - r), r * 2, r * 2, facecolor='green', edgecolor='green', alpha=0.3))

def doall(fileName, eng):
    img = cv2.imread(fileName)
    mask, scale = coin.segmenter.create_better_mask(fileName, 500)
    ml_mask = prepare_mask_for_matlab(mask)
    cr = eng.findCircles(ml_mask, scale)
    centers = np.array(cr['centers'])
    radii = np.array(cr['radii'])
    draw_bounding_boxes(img, centers, radii)
