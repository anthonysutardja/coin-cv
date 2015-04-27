#!/usr/bin/env python
import cv2
import numpy as np


def create_coin_mask(bgr_image):
    """Returns a mask for the coins in the image.

    Example usage:

        bgr_im = cv2.imread('path/to/img.jpg')
        mask = create_coin_mask(bgr_im)
        masked_im = bgr_im * mask[:,:,np.newaxis]
        plt.imshow(masked_im)

    :params: bgr_image must be a BGR (not RGB) image.
    :return: 2D binary array with original image's size
    """
    # Setup
    kernel = np.ones((3, 3), np.uint8)
    gray_im = cv2.cvtColor(bgr_image, cv2.cv.CV_BGR2GRAY)

    _, thresh = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=6)

    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Prepare mask for graph cuts
    mask = np.zeros(gray_im.shape[:2], np.uint8)
    mask[sure_fg == 0] = cv2.GC_PR_BGD
    mask[sure_fg == 255] = cv2.GC_FGD

    # Parameters for graph cuts
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # mask will be modified
    cv2.grabCut(bgr_image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    overlay_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    return overlay_mask
