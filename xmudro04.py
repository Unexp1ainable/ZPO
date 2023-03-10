import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from helpers import *

# 3 is fucked up
INDEX = 5

paths = [
    ("data/0.png",102, False),
    ("data/2.png",85, False),
    ("data/3.png",18, False),
    ("data/6.png",141, False),
    ("data/7.png",90, False),
    ("data/5.png",80, False),
    ("data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_15kV_50nA.png", 40, True),
    ("data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_EV_test_300nA.png", 113, True),
    ("data/finalizace - FIB spots/122-0007X manual 5s/_2022_122-0007 S8252X, US_images_FIB_Spots_EV_test_3uA.png", 78, True)
   
]

path = paths[INDEX][0]
ellipse_size = paths[INDEX][1], int(paths[INDEX][1]*np.sin(np.deg2rad(55)))
true_width = ellipse_size[0]
has_label = paths[INDEX][2]

def fitEllipseFixedSize(img):
    kernel = np.zeros((ellipse_size[1]+1,ellipse_size[0]+1), dtype=np.uint8)
    kernel = cv.ellipse(kernel, (ellipse_size[0]//2,ellipse_size[1]//2), (ellipse_size[0]//2,ellipse_size[1]//2), 0,0,360, 1, thickness=1)
    mask = cv.filter2D(img, cv.CV_32F, kernel)
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    plotImageAs3D(mask)
    
    sobel = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)
    plotImageAs3D(sobel)
    amin = np.unravel_index(np.argmax(sobel, axis=None), sobel.shape)
    center = (amin[1],amin[0])
    width = ellipse_size[0]
    height = ellipse_size[1]
    phi = 0
    fig, ax = plt.subplots(1,2,figsize=(12, 6))
    ax[0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    ax[0].axis('equal')
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax[0].add_patch(ellipse)

    ax[1].imshow(sobel)
    ax[1].axis('equal')
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax[1].add_patch(ellipse)
    plt.show()

def get_ellipse_size(width):
    return width, int(width*np.sin(np.deg2rad(55)))

def getNegativeInsideKernel(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    inside = np.array(cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=thickness)
    return kernel

def getPlaygroundKernel(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = np.zeros((height+1,width+1), dtype=np.float32)
    inside = np.array(cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=-1)==1)
    kernel[inside] = -1
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,180, 1, thickness=thickness)
    kernel[:height//2,:] = 0

    inside_mask = kernel == -1
    border_mask = kernel == 1
    border_pxs = np.count_nonzero(border_mask)
    inside_pxs = np.count_nonzero(inside_mask)
    kernel[inside_mask] /=inside_pxs
    kernel[border_mask] /=border_pxs
    return kernel

def getNegativeKernel(width, thickness=1):
    width, height = get_ellipse_size(width)
    kernel = -np.ones((height+1,width+1), dtype=np.float32)
    kernel = cv.ellipse(kernel, (width//2,height//2), (width//2,height//2), 0,0,360, 1, thickness=thickness)
    return kernel

def fitEllipse(img, plot=False):
    width, height = img.shape
    min_width = int(width*0.05)
    max_width = int(width*0.8)
    best_score = 0
    best_center = None
    best_width = None
    best_sobel = None
    for width in range(min_width, max_width):
        kernel = getNegativeInsideKernel(width,2)
        mask = cv.filter2D(img, cv.CV_32F, kernel)
        sobel = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)
        amin = np.unravel_index(np.argmax(sobel, axis=None), sobel.shape)
        center = (amin[1],amin[0])
        print(f"width: {width}\tcenter: {center}\tscore {sobel[amin]}")
        if sobel[amin] > best_score:
            best_score = sobel[amin]
            best_center = center
            best_width = width
            best_sobel = sobel
    print("="*50)
    print(f"width: {best_width}\ttrue_width={paths[INDEX][1]}\tcenter: {best_center}\tscore {best_score}")
    
    width, height = get_ellipse_size(best_width)
    center = best_center
    if plot:
        plt.imshow(img)
        plt.show()
        plt.imshow(kernel)
        plt.colorbar()
        plt.show()
        plotImageAs3D(best_sobel)
        
        fig, ax = plt.subplots(1,2,figsize=(12, 6))
        ax[0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        ax[0].axis('equal')
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0].add_patch(ellipse)
        ax[1].imshow(best_sobel)
        ax[1].axis('equal')
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[1].add_patch(ellipse)
        plt.show()



if __name__ == "__main__":
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if(has_label):
        img = img[:-79]
    fitEllipse(img, True)