import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ellipse import LsqEllipse
from helpers import *


def findSeed(img: np.ndarray):
    ksize = min(img.shape)//5
    kernel = np.ones((ksize, ksize), np.float32)/(ksize**2)
    im = img.astype(np.float32)
    dst = cv.filter2D(im, -1, kernel)
    _, _, minLoc, _ = cv.minMaxLoc(dst)
    return minLoc


def detectHalo(img: np.ndarray):
    # find background mean and standard deviation
    center = (img.shape[1]//2, img.shape[0]//2)

    # filter part of the image, where only background should be
    ellipseMask = np.zeros_like(img, dtype=np.uint8)
    ellipseMask = cv.ellipse(ellipseMask, center, center, 0, 0, 360, 255, -1)
    ellipseMaskInv = np.bitwise_not(ellipseMask).astype(np.uint8)
    background = np.bitwise_and(img, ellipseMaskInv)
    # plt.title("Eliptická maska")
    # plt.imshow(ellipseMaskInv)
    # plt.show()
    # plt.title("Vyfiltrované pozadie")
    # plt.imshow(cv.cvtColor(background, cv.COLOR_GRAY2RGB))
    # plt.show()

    # calculate its characteristics
    counts, bins = np.histogram(background, 256, [0, 256])
    counts[0] = 0
    thresh = max(counts) / 10
    counts[counts < thresh] = 0
    # plt.bar(range(0, 256), counts, 1)
    # plt.show()
    probs = counts / np.sum(counts)
    vals = list(range(256))
    mean = np.sum(probs * vals)
    sd = np.sqrt(np.sum(probs * (vals - mean)**2))

    # mask out where the background should be
    mask = np.logical_or(img > (mean+sd*3), img < mean-sd*3)
    mask = mask.astype(np.uint8)
    # plt.imshow(mask)
    # plt.show()

    # filter out background to reduce noise
    mask = np.logical_and(mask, ellipseMask).astype(np.uint8)

    amask = cv.medianBlur(mask, 3)

    # further remove noise (values are 1 or 0, if pixel does not have enough neighbours, the value will be rounded to 0)
    # plt.title("Vyfiltrované pozadie na celej snímke")
    # plt.imshow(mask)
    # plt.show()
    exmask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7)))
    # plt.title("Morfologické zatváranie")
    # plt.imshow(mask)
    # plt.show()
    mmask = cv.medianBlur(exmask, 5)
    # plt.title("Filtrovanie šumu")
    # plt.imshow(mask)
    # plt.show()
    # close and double floodfill to hopefuly extract only the hole
    ffmask = np.zeros_like(mmask, dtype=np.uint8)
    ffmask = np.zeros((mmask.shape[0]+2, mmask.shape[1]+2), dtype=np.uint8)

    cv.floodFill(mmask, ffmask, (0, 0), 1, flags=cv.FLOODFILL_MASK_ONLY)
    # mask = np.invert(ffmask)
    fmask2 = np.zeros_like(ffmask)
    seed = findSeed(ffmask)
    # plt.title("floodfill")
    # plt.imshow(ffmask)
    # plt.show()

    cv.floodFill(ffmask[1:-1, 1:-1], fmask2, seed, 1, flags=cv.FLOODFILL_MASK_ONLY)
    fmask2 = fmask2[1:-1, 1:-1]

    # plt.title("Dvojitý floodfill")
    # plt.imshow(fmask2)
    # plt.show()

    dilated = cv.morphologyEx(fmask2, cv.MORPH_DILATE, np.ones((3, 3)))
    edge = dilated - fmask2
    # plt.title("Extrakcia hrany")
    # plt.imshow(edge)
    # plt.show()

    a, b = np.nonzero(edge)
    params = fitEllipse(b, a)
    elimg = draw_ellipse(cv.cvtColor(img, cv.COLOR_GRAY2BGR), *params)
    # plt.title("Skoro výsledok")
    # plt.imshow(elimg)
    # plt.axis('off')
    # plt.show()

    center, width, height, phi = params
    c = np.rint(center).astype(int)
    a = np.rint((width-10, height-10)).astype(int)
    toDraw = edge.copy()
    cv.ellipse(toDraw, c, a, np.rad2deg(phi), 0, 360, 0, -1)
    # plt.imshow(toDraw)

    a, b = np.nonzero(toDraw)
    params = fitEllipse(b, a)
    # nimg = draw_ellipse(cv.cvtColor(img, cv.COLOR_GRAY2BGR), *params)
    # plt.title("Výsledok")
    # plt.imshow(nimg)
    # plt.axis('off')
    # plt.show()
    return params


def fitEllipse(X1, X2):
    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    return reg.as_parameters()


if __name__ == "__main__":
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    with open("gut.txt") as file:
        files = file.readlines()
    # files = ["data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_EV_test_3uA.png"]
    # files = ["data/finalizace - FIB spots/122-0007X manual 5s/_2022_122-0007 S8252X, US_images_FIB_Spots_EV_test_3uA.png"]
    for path in files:
        print(f"Processing: {path}")
        path = path.strip()
        img = load_image(path)
        img = cv.GaussianBlur(img, (5, 5), 0)

        # processColumns(img)
        params = detectHalo(img)
        nimg = draw_ellipse(cv.cvtColor(img, cv.COLOR_GRAY2BGR), *params)
        plt.title("Výsledok")
        plt.imshow(nimg)
        plt.axis('off')
        plt.show()
    # plotImageContours(img, "columns")

    # img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img = cv.GaussianBlur(img, (5, 5), 0)
    # # plt.plot(img[img.shape[0]//2])
    # # plt.show()
    # plotImageAs3D(img)
    # plt.ion()
    # # fig, ax = plt.subplots()
    # # ax.set_ylim([0, 255])

    # for row in img:
    #     plt.clf()
    #     plt.ylim([0, 255])
    #     plt.plot(row)
    #     plt.pause(0.05)
    #     plt.draw()
