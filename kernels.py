import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_ellipse_size(width):
    return width, int(width*np.sin(np.deg2rad(55)))

# total bullshit
def negative_inside(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    inside = np.array(cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=thickness)
    return kernel

# popici kernel
def half_empty(width, use_outline=False):
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

# este trochu lepsi ako half_empty
# za mna popici
def half_empty_norm(width, use_outline=False):
    kernel = half_empty(width, use_outline)
    filled_region_mask = kernel==-1
    kernel[filled_region_mask] /= np.count_nonzero(filled_region_mask)
    positive_mask = kernel == 1
    kernel[positive_mask] /= np.count_nonzero(positive_mask)
    return kernel

# pre vacsinu pripadov vytvara prilis male kruznice
# je to kvoli tomu ze nie je normalizovany a prilis ho bijeme parchanta
def edged(width, use_outline=False):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height,width), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    outline_region = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=1)
    outline_region_mask = np.array(outline_region == 1)
    kernel[outline_region_mask] = 1 if use_outline else -1
    kernel[kernel==2] = 1
    return kernel

# pre vacsinu pripadov vytvara prilis male kruznice
# je to kvoli tomu ze nie je normalizovany a prilis ho bijeme parchanta
def edged_norm(width, use_outline=False):
    kernel = edged(width, use_outline)
    filled_region_mask = kernel==-1
    kernel[filled_region_mask] /= np.count_nonzero(filled_region_mask)
    positive_mask = kernel == 1
    kernel[positive_mask] /= np.count_nonzero(positive_mask)
    return kernel


# dava male kruznice
def half_negative(width, use_outline=False):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height,width), dtype=np.float32)
    kernel[height//2:,:] = 2 #make upper half of filter equal to -1
    filled_region = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    filled_region_mask = np.array(filled_region==1)
    kernel[filled_region_mask] = -1
    outline_region = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=1)
    outline_region_mask = np.array(outline_region == 1)
    kernel[outline_region_mask] = 1 if use_outline else -1
    kernel[:height//2,:] = -1
    kernel[kernel==2] = 1
    return kernel

# dava male kruznice + umiestnuje ich nizsie ako treba
def half_negative_norm(width, use_outline=False):
    kernel = half_negative(width, use_outline)
    filled_region_mask = kernel==-1
    kernel[filled_region_mask] /= np.count_nonzero(filled_region_mask)
    positive_mask = kernel == 1
    kernel[positive_mask] /= np.count_nonzero(positive_mask)
    return kernel