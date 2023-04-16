import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
import os
from xmudro04 import fitEllipse, fitEllipseAndPlot
from kernels import *
from helpers import load_image


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


def processImage(path):
    # Load image
    img = load_image(path)

    #img = cv.GaussianBlur(img, (5, 5), 0)

    # Get histogram
    r = img.ravel()
    counts, bins = np.histogram(r, bins=256, range=(0, 255))

    # Smooth the values
    counts = np.convolve(counts, [1, 1, 1, 1, 1, 1, 1], 'same')

    # Show histogram
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor="black", align="edge")

    # Find the local minimums and filter the edges
    mins = argrelextrema(counts, np.less, order=8)[0]
    mins = list(filter(lambda x: 20 <= x <= 220, mins))

    min = merge(mins)[0]

    plt.axvline(min, color='g')
    _, out = cv.threshold(img, min, 255, cv.THRESH_BINARY)
    plt.show()

    max_count, max_col, max_start_pos, max_end_pos = find_height(out)

    # Draw a red line over the longest sequence of black pixels
    img_with_line = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.line(img_with_line, (max_col, max_start_pos), (max_col, max_end_pos), (0, 0, 255), thickness=2)
    cv.putText(img_with_line, str(max_count), (max_col, max_end_pos+30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    # Show the results
    images = np.concatenate((img, img_with_line), axis=1)
    cv.imshow(f'img - {path}', images)
    cv.waitKey(0)


# Otsu's thresholding
def otsu(path, double=False):
    # Load image
    img = load_image(path)

    # Blur Image and apply Otsu's method
    blur = cv.GaussianBlur(img, (7, 7), 0)
    ret1, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Optional second application
    if double:
        masked = cv.bitwise_and(img, img, mask=th)
        ret2, th = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Show results
    images = np.concatenate((img, th), axis=1)
    cv.imshow(f'img - {path}', images)
    cv.waitKey(0)


if __name__ == "__main__":
    #processImage('./data/4.png')
    #processImage('./data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_100nA.png')
    #exit()

    with open("gut.txt") as file:
        files = file.readlines()
    for path in files:
        path = path.strip()
        img = processImage(path)
        #kernel = half_empty
        #print(fitEllipseAndPlot(img, kernel, plot=True))
