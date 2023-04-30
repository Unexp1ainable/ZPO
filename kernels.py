"""
=========================================================================
Brief: Function for creation of convolution kernels
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: kernels.py
Date: April 2023
=========================================================================
"""

from typing import Tuple, Union
import numpy as np
import cv2 as cv


def get_ellipse_size(width: int) -> Tuple[int, int]:
    """Calculate height using sin(55) and width

    Args:
        width (int): Width of the ellipse

    Returns:
        Tuple[int,int]: (width, height)
    """
    return width, int(width*np.sin(np.deg2rad(55)))


def half_empty(width: int, height: Union[int, None] = None):
    """Create kernel with an ellipse, that is 0 in the upper half of the kernel.

    Args:
        width (int): width of the ellipse
        height (int, optional): Height of the ellipse. Uses sin(55)*width if None. Defaults to None.

    Returns:
        np.ndarray: Kernel
    """
    if not height:
        width, height = get_ellipse_size(width)
    kernel = np.zeros((height, width), dtype=np.float32)
    kernel[height//2:, :] = 2  # make upper half of filter equal to zero
    filled_region = cv.ellipse(kernel, (width//2, height//2), (width//2, height//2), 0, 0, 360, 1, thickness=-1)
    filled_region_mask = np.array(filled_region == 1)
    kernel[filled_region_mask] = -1
    kernel[:height//2, :] = 0
    kernel[kernel == 2] = 1
    return kernel


def half_empty_norm(width, height=None):
    """Create a normalized kernel with an ellipse, that is 0 in the upper half.

    Args:
        width (int): width of the ellipse
        height (int, optional): Height of the ellipse. Uses sin(55)*width if None. Defaults to None.

    Returns:
        np.ndarray: Kernel
    """
    kernel = half_empty(width, height)
    filled_region_mask = kernel == -1
    kernel[filled_region_mask] /= np.count_nonzero(filled_region_mask)
    positive_mask = kernel == 1
    kernel[positive_mask] /= np.count_nonzero(positive_mask)
    return kernel


def lower20(width, height=None):
    """Create a normalized kernel with an ellipse, that is 0 in the upper 80% if the kernel.

    Args:
        width (int): width of the ellipse
        height (int, optional): Height of the ellipse. Uses sin(55)*width if None. Defaults to None.

    Returns:
        np.ndarray: Kernel
    """
    if not height:
        width, height = get_ellipse_size(width)

    area = np.pi * width//2 * height//2
    val = 2/(width*height - area)
    kernel = np.full((height, width), val/3, dtype=np.float32)
    cv.ellipse(kernel, (width//2, height//2), (width//2, height//2), 0, 0, 180, -2.5/area, thickness=-1)
    kernel[:int(height*0.75), :] = 0
    return kernel
