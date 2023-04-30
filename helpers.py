"""
=========================================================================
Brief: Helper functions
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: helpers.py
Date: April 2023
=========================================================================
"""

import configparser
from typing import Tuple

import cv2 as cv
import numpy as np


def loadImage(path: str) -> np.ndarray:
    """Attempts to load an image and remove info strip if .hdr file is available.

    Args:
        path (str): Path to file. Header file is expected to be in the same folder and have unusal format 
        (.png replaced with -png.hdr)

    Raises:
        Exception: If path to file is invalid

    Returns:
        np.ndarray: Loaded image [with cropped info strip].
    """
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise NameError(f"Invalid image path: {path}")
    inipath = path[:-4] + "-png.hdr"
    config = configparser.ConfigParser()
    try:
        config.read(inipath)
        ssize = int(config["MAIN"]["ImageStripSize"])
        img = img[:-ssize]

    except KeyError:
        pass
    return img


def loadPixelsize(path: str) -> Tuple[float, float]:
    """Attempt to load pixelsize from .hdr file associated with the image

    Args:
        path (str): Path to the image

    Returns:
        Tuple[float, float]: PixelSizeX, PixelSizeY
    """
    inipath = path[:-4] + "-png.hdr"
    pxsx = pxsy = 1.

    config = configparser.ConfigParser()
    try:
        config.read(inipath)
        xs = float(config["MAIN"]["PixelSizeX"])
        ys = float(config["MAIN"]["PixelSizeY"])
        pxsx = xs
        pxsy = ys

    except KeyError:
        pass

    return pxsx, pxsy
