import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from helpers import *
from kernels import *


def fitEllipse(img, kernel_func):
    img_width, _ = img.shape
    min_width = int(img_width*0.1)
    max_width = int(img_width*0.9)
    best_score = -np.inf
    for width in range(min_width, max_width):
        kernel = kernel_func(width)
        mask = cv.filter2D(img, cv.CV_32F, kernel)
        amin = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        center = (amin[1], amin[0])
        if mask[amin] > best_score:
            best_score = mask[amin]
            best_center = center
            best_width = width

    width, height = get_ellipse_size(best_width)
    center = best_center
    return (center), (width, height)


def fitEllipseAndPlot(img, kernel_func, plot=False):
    img_width, img_height = img.shape
    min_width = int(img_width*0.1)
    max_width = int(img_width*0.9)
    best_score = -np.inf
    best_center = None
    best_width = None
    best_mask = None
    best_kernel = None

    for width in range(min_width, max_width):
        kernel = kernel_func(width)
        mask = cv.filter2D(img, cv.CV_32F, kernel)
        amin = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        center = (amin[1], amin[0])
        if mask[amin] > best_score:
            best_score = mask[amin]
            best_center = center
            best_width = width
            best_mask = mask
            best_kernel = kernel

    width, height = get_ellipse_size(best_width)
    center = best_center
    print((center), (width, height))
    if plot:
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(best_kernel)
        # plt.colorbar()
        # plt.show()
        # plotImageAs3D(best_mask)

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax[0, 0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        ax[0, 0].axis('equal')
        ax[0, 0].set_title("Image")
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0, 0].add_patch(ellipse)

        ax[0, 1].imshow(best_mask)
        ax[0, 1].axis('equal')
        ax[0, 1].set_title("Convolved image")
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0, 1].add_patch(ellipse)

        center_x, center_y = center
        left, right = max(0, center_x - width//2), min(center_x + width//2, img_width-1)
        top, bottom = max(0, center_y - height//2), min(center_y + height//2, img_height-1)
        crop = img[top:bottom, left:right]
        ax[1, 0].imshow(crop)
        ax[1, 0].axis('equal')
        ax[1, 0].set_title("Cropped section")

        cb_han = ax[1, 1].imshow(best_kernel)
        ax[1, 1].axis('equal')
        ax[1, 1].set_title("Best filter")
        fig.colorbar(cb_han, ax=ax[1, 1])
        plt.show()


if __name__ == "__main__":
    files = []
    with open("gut.txt") as file:
        files = file.readlines()

    for path in files:
        path = path.strip()
        img = load_image(path)

        kernel = half_empty_norm
        fitEllipseAndPlot(img, kernel, True)
