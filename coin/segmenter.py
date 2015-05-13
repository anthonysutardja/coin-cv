#!/usr/bin/env python
import cv2
import numpy as np
import coin.utils
import time
import matplotlib.pyplot as plt

from coin.predictor import predict_bgr_uint8_images

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

    thresh1 = cv2.adaptiveThreshold(
        gray_im, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 10
    )
    _, thresh2 = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # thresh = thresh1 + thresh2
    thresh = thresh2

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

    # Clean up the mask
    kernel = np.ones((4, 4), np.uint8)
    refined_overlay_mask = cv2.morphologyEx(overlay_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return refined_overlay_mask

def create_better_mask(imgPath, desiredSize=500):
    """Returns a mask for the coins in the image
    Applies an adaptive threshold to a blurred value space of the image to get an initial mask guess
    Refines this mask using the grabCut algorithm
    Then, holes are closed using dilation followed by erosion morphological transformation
    :params: an image path
    :return: 2D binary array with original image's size
    """
    img, scale = coin.utils.resize_image(cv2.imread(imgPath), desiredSize)
    v = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]
    if (np.mean(h[1:10, :]) + np.mean(h[:, 1:10]) + np.mean(h[:,-10:-1]) + np.mean(h[-10:-1,:]) > 480):
        v = h
    v = cv2.medianBlur(v, 11)
    thresh = cv2.adaptiveThreshold(v , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 401, 10);
    h, w = img.shape[:2]
    mask = thresh/255*3

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,np.shape(img)[1]-100,np.shape(img)[0]-100)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    kernel = np.ones((4,4),np.uint8)
    refined_mm = cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernel, iterations = 2)
    ones = sum((refined_mm == 1).astype('int').flatten())
    zeros = sum((refined_mm == 0).astype('int').flatten())
    if (ones > zeros):
        return (refined_mm + 1) % 2, scale
    return refined_mm, scale 

def segmentImage(img, centers, radii):
    """
    :params: image matrix, located centers, located radii
    :return: a list of nxmx3 matricies which are detected coins
    """
    results = []
    for i in xrange(radii.shape[0]):
        x = centers[i][0]
        y = centers[i][1]
        r = radii[i][0]
        results.append(img[y-(r*1.2):y+r,x-(r*1.2):x+r,:])
    return results

def computeClassifications(coins):
    """
    :params: list of coin images
    """
    return predict_bgr_uint8_images([cv2.resize(c, (256, 256)) for c in coins])

def processImgBounds(fileName, eng):
    """
    finds the coins in the image and graphs it with bounding boxes around coins
    :params: file path, matlab engine
    """
    now = time.time()
    img = cv2.imread(fileName)
    mask, scale = coin.segmenter.create_better_mask(fileName, 1000)
    ml_mask = coin.utils.prepare_mask_for_matlab(mask)
    cr = eng.findCircles(ml_mask, scale)
    centers = np.array(cr['centers'])
    radii = np.array(cr['radii'])
    coin.utils.draw_bounding_boxes(img, centers, radii)
    print(time.time()-now)

def processImg(fileName, eng):
    """
    :params: file path, matlab engine
    :return: a list of nxmx3 matricies which are detected coins
    """
    img = cv2.imread(fileName)
    mask, scale = create_better_mask(fileName, 1000)
    ml_mask = coin.utils.prepare_mask_for_matlab(mask)
    cr = eng.findCircles(ml_mask, scale)
    centers = np.array(cr['centers'])
    radii = np.array(cr['radii'])
    return segmentImage(img, centers, radii)

LABEL = {
    'Q': 'Quarter',
    'P': 'Penny',
    'N': 'Nickel',
    'D': 'Dime',
}

def processImageforJSON(fileName, eng):
    '''
    Use this one for the webapp
    :params: file path, matlab engine
    :return: JSON representing all the center, radii, classification of the coins found
    '''
    img = cv2.imread(fileName)
    mask, scale = create_better_mask(fileName, 1000)
    ml_mask = coin.utils.prepare_mask_for_matlab(mask)
    cr = eng.findCircles(ml_mask, scale)
    centers = np.array(cr['centers'])
    radii = np.array(cr['radii'])
    coins = segmentImage(img, centers, radii)
    cl = computeClassifications(coins)
    detections = []
    for i in xrange(radii.shape[0]):
        if cl[i] != 'BAD':
            detections.append({"x":centers[i][0], "y":centers[i][1], "r":radii[i][0], "cl":LABEL[cl[i]]})
    return {"size": {"h": img.shape[0], "w": img.shape[1]}, "detections": detections}
