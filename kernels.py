import numpy as np
import cv2 as cv

def get_ellipse_size(width):
    return width, int(width*np.sin(np.deg2rad(55)))


def half_empty(width, height = None, use_outline=False):
    if not height:
        width, height = get_ellipse_size(width)
    kernel = np.zeros((height,width), dtype=np.float32)
    kernel[height//2:,:] = 2 #make upper half of filter equal to zero
    filled_region = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    filled_region_mask = np.array(filled_region==1)
    kernel[filled_region_mask] = -1
    outline_region = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=1)
    outline_region_mask = np.array(outline_region == 1)
    kernel[outline_region_mask] = 1 if use_outline else -1
    kernel[:height//2,:] = 0
    kernel[kernel==2] = 1
    return kernel


def half_empty_norm(width, height = None, use_outline=False):
    kernel = half_empty(width, height, use_outline)
    filled_region_mask = kernel==-1
    kernel[filled_region_mask] /= np.count_nonzero(filled_region_mask)
    positive_mask = kernel == 1
    kernel[positive_mask] /= np.count_nonzero(positive_mask)
    return kernel


def lower20(width, height = None, use_outline=False):
    if not height:
        width, height = get_ellipse_size(width)

    area = np.pi * width//2 * height//2
    val = 2/(width*height - area)
    kernel = np.full((height, width), val/3, dtype=np.float32)
    cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, -2/area, thickness=-1)
    cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,180,360, -0.5/area, thickness=-1)
    kernel[:int(height*0.8),:] = 0
    return kernel
