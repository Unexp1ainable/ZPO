"""
=========================================================================
Brief: FIB spot measurement using correlation
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: correlation_procedure.py
Date: April 2023
=========================================================================
"""

from typing import Tuple, Union
import cv2 as cv
import numpy as np
from kernels import *


def wiggle(img: np.ndarray, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Attempt to fit an ellipse with parameters similar to given width and height to the lower edge of the spot.

    Args:
        img (np.ndarray): Input image
        width (int): Approximate width of the spot
        height (int): Approximate height of the spot

    Returns:
        Tuple[Tuple[int,int], Tuple[int,int]]:((xpos, ypos), (width, height)) of the fitted ellipse
    """
    # determine ranges
    min_width = int(width*0.9)
    max_width = int(width*1.1)
    min_height = int(height * 0.9)
    max_height = int(height * 1.0)

    # will hold best parameters
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

            # if better kernel respone, save parameters
            if mask[amin] > best_score:
                best_score = mask[amin]
                best_center = center
                best_width = width
                best_height = height

    return best_center, (best_width, best_height)


def fitEllipse(img: np.ndarray, height: Union[int, None] = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Fit ellpise to the middle part of the spot

    Args:
        img (np.ndarray): Input image
        height (Union[int, None], optional): Height if known. If not, sin(55)*width will be used. Defaults to None.

    Returns:
        Tuple[Tuple[int,int], Tuple[int,int]]: ((xpos, ypos), (width, height)) of the fitted ellipse
    """
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
