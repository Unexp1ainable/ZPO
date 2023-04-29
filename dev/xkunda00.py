import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
import os
from xmudro04 import fitEllipse
from kernels import *
from helpers import loadImage


def merge(xs, tresh=20):
    l = len(xs)
    if l < 2:
        return xs
    out = []
    i = 0
    while i < l - 1:
        if (xs[i+1] - xs[i]) <= tresh:
            out.append((xs[i+1] + xs[i]) // 2)
            i += 1
            if i == l - 1:
                return out
        else:
            out.append(xs[i])
        i += 1
    out.append(xs[l-1])
    return out


def find_height(img):
    # Iterate over the columns
    max_count = 0
    max_col = 0
    max_start_pos = 0
    max_end_pos = img.shape[0]
    for j in range(img.shape[1]):
        col = img[:, j]
        start = False
        c = count = start_pos = 0
        end_pos = img.shape[0]
        for i in range(img.shape[0]):
            if col[i] == 0:
                if not start:
                    start = True
                    start_pos = i
                c += 1
            elif start:
                start = False
                end_pos = i
                if c > count:
                    count = c
                c = 0

        if count > max_count:
            max_count = count
            max_col = j
            max_start_pos = start_pos
            max_end_pos = end_pos

    return max_count, max_col, max_start_pos, max_end_pos


def determineHeight(img):
    #img = cv.GaussianBlur(img, (5, 5), 0)

    # Get histogram
    r = img.ravel()
    counts, bins = np.histogram(r, bins=256, range=(0, 255))

    # Smooth the values
    counts = np.convolve(counts, [1, 1, 1, 1, 1, 1, 1], 'same')

    # Show histogram
    # fig, ax = plt.subplots()
    # ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor="black", align="edge")
    # Find the local minimums and filter the edges
    # mins = argrelextrema(counts, np.less, order=8)[0]
    # mins = list(filter(lambda x: 20 <= x <= 220, mins))

    # min = merge(mins)[0]
    firstPeak = 0
    expectedPeakRising = (img.shape[0]*0.1) * (img.shape[1]*0.1) # 0.1 is empirical value that is scaled according to image size
    for i, item in enumerate(counts):
        if item > expectedPeakRising:
            firstPeak = i
            break

    minMin = 9999999999
    minI = 0
    rising = 0
    # find first valley to the left of first peak
    for i in range(firstPeak, 0, -1):
        if minMin > counts[i]:
            minMin = counts[i]
            minI = i
            rising = 0
        else:
            rising+=1
            if rising > 5:
                break

    # hard minimum necessary for some ugly images
    minI = max(minI, 45)

    # plt.axvline(minI, color='g')
    _, out = cv.threshold(img, minI, 255, cv.THRESH_BINARY)
    # plt.show()
    # plt.figure()

    return find_height(out)


def drawLine(img, max_count, max_col, max_start_pos, max_end_pos):
    # Draw a red line over the longest sequence of black pixels
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.line(img, (max_col, max_start_pos), (max_col, max_end_pos), (0, 0, 255), thickness=2)
    cv.putText(img, str(max_count), (max_col, max_end_pos+30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    # Show the results
    plt.title(f'img - {path}')
    plt.imshow(img)
    plt.show()


# Otsu's thresholding
def otsu(path, double=False):
    # Load image
    img = loadImage(path)

    # Blur Image and apply Otsu's method
    blur = cv.GaussianBlur(img, (7, 7), 0)
    ret1, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Optional second application
    if double:
        masked = cv.bitwise_and(img, img, mask=th)
        ret2, th = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Show results
    images = np.concatenate((img, th), axis=1)
    plt.title(f'img - {path}')
    plt.imshow(images)
    plt.show()


if __name__ == "__main__":
    # processImage('./data/4.png')
    #processImage('./data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_100nA.png')
    # exit()

    with open("gut.txt") as file:
        files = file.readlines()
    for path in files:
        path = path.strip()
        # path = "data/finalizace - FIB spots/122-0007X manual 5s/_2022_122-0007 S8252X, US_images_FIB_Spots_EV_test_300nA.png"
        path = "data/4.png"
        print(path)
        # Load image
        img = loadImage(path)
        res = determineHeight(img)
        drawLine(img, *res)
        #kernel = half_empty
        #print(fitEllipseAndPlot(img, kernel, plot=True))
