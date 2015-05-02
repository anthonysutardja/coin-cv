#!/user/bin/env python
import cv2

import numpy as np
import matlab


def resize_image(im, desired_size=1000):
    biggest_dim = max(im.shape)
    scale = float(desired_size) / float(biggest_dim)
    return cv2.resize(im, (0,0), fx=scale, fy=scale), scale

def create_pad_mask(im, padding=50):
    h, w = im.shape
    np.zeros((h, w), )
    return np.pad(np.zeros((h - (padding * 2), w - (padding * 2))), padding, mode='constant', constant_values=1)

numpy2matlab_type = {
    np.int64         : matlab.int64,
    np.bool_         : matlab.int8,
    np.int8          : matlab.int8,
    np.int16         : matlab.int16,
    np.int32         : matlab.int32,
    np.int64         : matlab.int64,
    np.uint8         : matlab.uint8,
    np.uint16        : matlab.uint16,
    np.uint32        : matlab.uint32,
    np.uint64        : matlab.uint64,
    np.float16       : matlab.single,
    np.float32       : matlab.single,
    np.float64       : matlab.double
}

def numpy2matlab(np_arr):
    np_type = np_arr.dtype.type
    ml_arr_klass = numpy2matlab_type.get(np_type, None)
    if not ml_arr_klass:
        raise ValueError('Cannot convert numpy type {0} to matlab array.'.format(np_type))
    ml_arr = ml_arr_klass(np_arr.flatten().tolist())
    ml_arr.reshape(np_arr.shape)
    return ml_arr
