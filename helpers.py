"""_summary_

Raises:
    Exception: _description_

Returns:
    _type_: _description_
"""

import configparser

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

def loadPixelsize(path: str):
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
