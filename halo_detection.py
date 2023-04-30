"""
=========================================================================
Brief: FIB spot halo detection
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: halo_detection.py
Date: April 2023
=========================================================================
"""

from typing import Tuple
import cv2 as cv
import numpy as np
from ellipse import LsqEllipse


def findSeed(img: np.ndarray) -> Tuple[int, int]:
    """Find place where it is best to begin floodfill

    Args:
        img (np.ndarray): Input image

    Returns:
        Tuple[int,int]: (posx, posy)
    """
    # blur the image with a chonky kernel
    ksize = min(img.shape)//5
    kernel = np.ones((ksize, ksize), np.float32)/(ksize**2)
    im = img.astype(np.float32)
    dst = cv.filter2D(im, -1, kernel)
    # find the minimum
    _, _, minLoc, _ = cv.minMaxLoc(dst)
    return minLoc


def detectHalo(img: np.ndarray) -> Tuple[Tuple[float, float], float, float, float]:
    """Perform halo detection using background removal

    Args:
        img (np.ndarray): Input image

    Returns:
        Tuple[(float, float), float, float, float]: Parameters of fitted ellipse ((posx, posy),width, height, phi)
    """
    center = (img.shape[1]//2, img.shape[0]//2)

    # filter part of the image, where only background should be
    ellipseMask = np.zeros_like(img, dtype=np.uint8)
    ellipseMask = cv.ellipse(ellipseMask, center, center, 0, 0, 360, 255, -1)
    ellipseMaskInv = np.bitwise_not(ellipseMask).astype(np.uint8)
    background = np.bitwise_and(img, ellipseMaskInv)

    # find background mean and standard deviation
    counts, _ = np.histogram(background, 256, [0, 256])
    counts[0] = 0
    thresh = max(counts) / 10
    counts[counts < thresh] = 0
    probs = counts / np.sum(counts)
    vals = list(range(256))
    mean = np.sum(probs * vals)
    sd = np.sqrt(np.sum(probs * (vals - mean)**2))

    # mask out where the background should be
    mask = np.logical_or(img > (mean+sd*3), img < mean-sd*3)
    mask = mask.astype(np.uint8)

    # filter out background to reduce noise
    amask = np.logical_and(mask, ellipseMask).astype(np.uint8)

    # close to create a filled region if possible
    exmask = cv.morphologyEx(amask, cv.MORPH_CLOSE, np.ones((7, 7)))

    # remove noise
    mmask = cv.medianBlur(exmask, 5)

    # make sure, that floodfill gets everywhere
    mmask[0] = 0
    mmask[-1] = 0
    mmask[:, 0] = 0
    mmask[:, -1] = 0

    # close and double floodfill to hopefuly extract only the spot
    ffmask = np.zeros_like(mmask, dtype=np.uint8)
    ffmask = np.zeros((mmask.shape[0]+2, mmask.shape[1]+2), dtype=np.uint8)

    cv.floodFill(mmask, ffmask, (0, 0), 1, flags=cv.FLOODFILL_MASK_ONLY)
    fmask2 = np.zeros_like(ffmask)
    seed = findSeed(ffmask)

    ksize = round(img.shape[1]/512 * 21)
    exmask2 = cv.morphologyEx(ffmask[1:-1, 1:-1], cv.MORPH_CLOSE, np.ones((ksize, ksize)))

    cv.floodFill(exmask2, fmask2, seed, 1, flags=cv.FLOODFILL_MASK_ONLY)
    fmask2 = fmask2[1:-1, 1:-1]

    # extract the edge
    dilated = cv.morphologyEx(fmask2, cv.MORPH_DILATE, np.ones((3, 3)))
    edge = dilated - fmask2

    a, b = np.nonzero(edge)
    params = fitEllipse(b, a)

    # sometimes, artifacts are present inside of the filtered region. This is an attempt to remove them and improved fit precision.
    filtered = edge.copy()
    center, width, height, phi = params
    c = np.rint(center).astype(int)
    a = np.rint((width-10, height-10)).astype(int)

    cv.ellipse(filtered, c, a, np.rad2deg(phi), 0, 360, 0, -1)

    a, b = np.nonzero(filtered)
    params = fitEllipse(b, a)

    return params


def fitEllipse(xs: np.ndarray, ys: np.ndarray) -> Tuple[Tuple[float, float], float, float, float]:
    """Fit points to an ellipse
    Args:
        xs (np.ndarray): List of x coordinates of points
        ys (np.ndarray): List of y coordinates of points

    Returns:
        Tuple[(float, float), float, float, float]: Parameters of fitted ellipse ((posx, posy),width, height, phi)
    """

    X = np.array(list(zip(xs, ys)))
    if X.size == 0:
        raise ValueError("Unable to detect spot.")
    reg = LsqEllipse().fit(X)
    return reg.as_parameters()
