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
import time

def getEngine():
    eng = engine.start_matlab()
    load_matlab_util(eng)
    return eng

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
    fig, subplt = plt.subplots(1,1)
    plt.axis('off')
    if (centers.shape[0] > 0):
        subplt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        subplt.scatter(centers[:,0], centers[:,1])
        for i in range(radii.shape[0]):
            x = centers[i][0]
            y = centers[i][1]
            r = radii[i][0]
            subplt.add_patch(Rectangle((x - r, y - r), r * 2, r * 2, facecolor='green', edgecolor='green', alpha=0.3))
    else:
        print("No coins found")
    extent = subplt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('figure.png', bbox_inches=extent)

