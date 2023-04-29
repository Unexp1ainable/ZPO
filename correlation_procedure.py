import cv2 as cv
import numpy as np
from kernels import *

def wiggle(img, width, height):
    min_width = int(width*0.9)
    max_width = int(width*1.1)

    min_height = int(height * 0.9)
    max_height = int(height * 1.0)

    best_score = -np.inf
    best_center = None
    best_width = None
    best_height = None

    for width in range(min_width, max_width):
        for height in range(min_height, max_height):
            kernel = lower20(width, height)
            mask = cv.filter2D(img, cv.CV_32F, kernel)
            amin = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
            center = (amin[1], amin[0])
            if mask[amin] > best_score:
                best_score = mask[amin]
                best_center = center
                best_width = width
                best_height = height

    return best_center, (best_width, best_height)


def fitEllipse(img, height=None):
    img_width, _ = img.shape
    min_width = int(img_width*0.1)
    max_width = int(img_width*0.9)

    best_score = -np.inf
    for width in range(min_width, max_width):
        kernel = half_empty_norm(width, height)
        mask = cv.filter2D(img, cv.CV_32F, kernel)
        amin = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        if mask[amin] > best_score:
            best_score = mask[amin]
            best_width = width

    if not height:
        width, height = get_ellipse_size(best_width)

  
    return wiggle(img, best_width, height)
