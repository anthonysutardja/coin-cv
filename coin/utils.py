#!/user/bin/env python
import cv2

def resize_image(im, desired_size=1000):
    biggest_dim = max(im.shape)
    scale = float(desired_size) / float(biggest_dim)
    return cv2.resize(im, (0,0), fx=scale, fy=scale), scale

def create_pad_mask(im, padding=50):
    h, w = im.shape
    np.zeros((h, w), )
    return np.pad(np.zeros((h - (padding * 2), w - (padding * 2))), padding, mode='constant', constant_values=1)

