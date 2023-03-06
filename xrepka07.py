import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse
from helpers import *
# path = "data/finalizace - FIB spots/121-0201G manual looks like 1s though/_2022_121-0201 S9251G, CN_images_FIB_Spots_30 keV; 50 nA.png"
path = "data/1.png"


WINDOW_SIZE = 10
ALLOWED_DEVIATION = 10


def processColumns(img: np.ndarray):
    wc = img.copy().T
    width = wc.shape[0]
    for coli, col in enumerate(img):
        aa = []
        avg = np.average(col)
        val = np.sum(col[0:0+WINDOW_SIZE]) / WINDOW_SIZE
        if (abs(avg - val) < ALLOWED_DEVIATION):
            print(f"No significant feature expected. {coli}")
            continue

        currRef = val
        for i in range(width - WINDOW_SIZE):
            val = np.sum(col[i:i+WINDOW_SIZE]) / WINDOW_SIZE
            if abs(currRef-val) > ALLOWED_DEVIATION:
                print()
            aa.append(val)

        # plt.cla()
        # plt.ylim([0, 255])
        # plt.plot(aa)
        # plt.show()


def extractBackground(img: np.ndarray):
    # find background mean and standard deviation
    center = (img.shape[1]//2, img.shape[0]//2)

    # filter part of the image, where only background should be
    ellipseMask = np.zeros_like(img, dtype=np.uint8)
    ellipseMask = cv.ellipse(ellipseMask, center, center, 0, 0, 360, 255, -1)
    ellipseMaskInv = np.bitwise_not(ellipseMask).astype(np.uint8)
    background = np.bitwise_and(img, ellipseMaskInv)

    # calculate its characteristics
    counts, bins = np.histogram(background, 256)
    counts[0] = 0
    probs = counts / np.sum(counts)
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.sum(probs * mids)
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))

    # mask out where the background should be
    mask = np.logical_and(img < (mean+sd*3), img > mean-sd*3)
    mask = np.logical_not(mask)
    mask = mask.astype(np.uint8)

    # filter out background to reduce noise
    mask = np.logical_and(mask, ellipseMask).astype(np.uint8)
    # further remove noise (values are 1 or 0, if pixel does not have enough neighbours, the value will be rounded to 0)
    mask = cv.medianBlur(mask, 9)
    # close and double floodfill to hopefuly extract only the hole
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5)))
    ffmask = np.zeros_like(mask, dtype=np.uint8)
    ffmask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
    cv.floodFill(mask, ffmask, (0, 0), 1, flags=cv.FLOODFILL_MASK_ONLY)
    mask = np.zeros_like(ffmask, dtype=np.uint8)
    cv.floodFill(ffmask[1:-1, 1:-1], mask, center, 1, flags=cv.FLOODFILL_MASK_ONLY)
    mask = mask[1:-1, 1:-1]
    sobel = np.abs(cv.Sobel(mask.astype(np.float32), cv.CV_32F, 1, 0))
    sobel += np.abs(cv.Sobel(mask.astype(np.float32), cv.CV_32F, 0, 1))
    sobel = sobel != 0
    # plt.imshow(sobel)
    # plt.show()

    a, b = np.nonzero(sobel)
    fitEllipse(b, a, img)


def fitEllipse(X1, X2, img):
    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()

    print(f'center: {center[0]:.3f}, {center[1]:.3f}')
    print(f'width: {width:.3f}')
    print(f'height: {height:.3f}')
    print(f'phi: {phi:.3f}')

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    ax.axis('equal')
    # ax.plot(X1, X2, 'ro', zorder=1)
    ellipse = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # for i in range(9):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img = img[100: 400, 100: 400]
    img = cv.GaussianBlur(img, (5, 5), 0)

    # processColumns(img)
    # extractBackground(img)
    plotImageContours(img, "columns")
    # img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img = cv.GaussianBlur(img, (5, 5), 0)
    # # plt.plot(img[img.shape[0]//2])
    # # plt.show()

    # plt.ion()
    # # fig, ax = plt.subplots()
    # # ax.set_ylim([0, 255])

    # for row in img:
    #     plt.clf()
    #     plt.ylim([0, 255])
    #     plt.plot(row)
    #     plt.pause(0.05)
    #     plt.draw()
