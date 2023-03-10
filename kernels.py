import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_ellipse_size(width):
    return width, int(width*np.sin(np.deg2rad(55)))


# robi si co chce, niekedy da uplne maly odhad, inokedy to prestreli
def negative_inside(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    inside = np.array(cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=thickness)
    return kernel

# podobne ako edgedNormalized
def half_empty(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    kernel[:height//2,:] = 0
    return kernel

# pre vacsinu pripadov vytvara prilis male kruznice
def edged(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    return kernel

# celkom good, no pre niektore vytvara vacsie ako treba
def edged_norm(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    #kernel[:height//2,:] = 0

    inside_mask = kernel == -1
    inside_pxs = np.count_nonzero(inside_mask)
    kernel[inside_mask] /=inside_pxs
    border_mask = kernel == 1
    border_pxs = np.count_nonzero(border_mask)
    kernel[border_mask] /=border_pxs
    return kernel

# trochu lepsi ako halfempty normalized
def half_empty_norm(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    kernel[:height//2,:] = 0

    inside_mask = kernel == -1
    inside_pxs = np.count_nonzero(inside_mask)
    kernel[inside_mask] /= inside_pxs
    border_mask = kernel == 1
    border_pxs = np.count_nonzero(border_mask)
    kernel[border_mask] /=border_pxs
    return kernel

# popici kernel
def half_negative_norm(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    kernel[:height//2,:] = -1

    inside_mask = kernel == -1
    inside_pxs = np.count_nonzero(inside_mask)
    kernel[inside_mask] /= inside_pxs
    border_mask = kernel == 1
    border_pxs = np.count_nonzero(border_mask)
    kernel[border_mask] /=border_pxs
    return kernel

# vracia mensie ako treba
def half_negative(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    kernel[height//2:,:] = 2
    filled_ellipse = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)
    inside = np.array(filled_ellipse==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[kernel==2] = 1
    kernel[:height//2,:] = -1
    return kernel