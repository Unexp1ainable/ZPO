"""
=========================================================================
Brief: Height measurement using histogram analysis
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: histogram_analysis.py
Date: April 2023
=========================================================================
"""

import cv2 as cv
import numpy as np


def find_height(img: np.ndarray) -> int:
    """Find height of the spot on the thresholded image.

    Args:
        img (np.ndarray): Thresholded image

    Returns:
        int: Height
    """

    # Iterate over the columns
    max_count = 0
    for j in range(img.shape[1]):
        col = img[:, j]
        start = False
        c = count = 0
        for i in range(img.shape[0]):
            if col[i] == 0:
                if not start:
                    start = True
                c += 1
            elif start:
                start = False
                if c > count:
                    count = c
                c = 0

        if count > max_count:
            max_count = count

    return max_count


def determineHeight(img: np.ndarray) -> int:
    """Determine height of the middle part of the spot using image thresholding.

    Args:
        img (np.ndarray): Inut image

    Returns:
        int: Height
    """

    # Get histogram
    r = img.ravel()
    counts, _ = np.histogram(r, bins=256, range=(0, 255))

    # Smooth the values
    counts = np.convolve(counts, [1, 1, 1, 1, 1, 1, 1], 'same')

    # find first peak
    firstPeak = 0
    # 0.1 is empirical value that is scaled according to image size
    expectedPeakRising = (img.shape[0]*0.1) * (img.shape[1]*0.1)
    for i, item in enumerate(counts):
        if item > expectedPeakRising:
            firstPeak = i
            break

    minMin = 9999999999
    minI = 0
    rising = 0
    # find first valley to the left of first peak
    for i in range(firstPeak, 0, -1):
        if minMin > counts[i]:
            minMin = counts[i]
            minI = i
            rising = 0
        else:
            rising += 1
            if rising > 5:
                break

    # hard minimum necessary for some ugly images
    minI = max(minI, 45)

    _, out = cv.threshold(img, minI, 255, cv.THRESH_BINARY)

    return find_height(out)
