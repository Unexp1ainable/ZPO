import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ellipse import LsqEllipse
from helpers import *


def furtherEllipse(img: np.ndarray):
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
    # plt.title("Vyfiltrované pozadie na celej snímke")
    # plt.imshow(mask)
    # plt.show()
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7)))
    # plt.title("Morfologické zatváranie")
    # plt.imshow(mask)
    # plt.show()
    mask = cv.medianBlur(mask, 9)
    # plt.title("Filtrovanie šumu")
    # plt.imshow(mask)
    # plt.show()
    # close and double floodfill to hopefuly extract only the hole
    ffmask = np.zeros_like(mask, dtype=np.uint8)
    ffmask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)

    cv.floodFill(mask, ffmask, (0, 0), 1, flags=cv.FLOODFILL_MASK_ONLY)
    # mask = np.invert(ffmask)
    fmask2 = np.zeros_like(ffmask)
    cv.floodFill(ffmask[1:-1, 1:-1], fmask2, center, 1, flags=cv.FLOODFILL_MASK_ONLY)
    fmask2 = fmask2[1:-1, 1:-1]

    # plt.title("Dvojitý floodfill")
    # plt.imshow(fmask2)
    # plt.show()

    sobel = np.abs(cv.Sobel(fmask2.astype(np.float32), cv.CV_32F, 1, 0))
    sobel += np.abs(cv.Sobel(fmask2.astype(np.float32), cv.CV_32F, 0, 1))
    sobel = sobel != 0
    # plt.title("Extrakcia hrany")
    # plt.imshow(sobel)
    # plt.show()

    a, b = np.nonzero(sobel)
    params = fitEllipse(b, a, img)
    elimg = draw_ellipse(cv.cvtColor(img, cv.COLOR_GRAY2BGR), *params)
    # plt.title("Výsledok")
    plt.imshow(elimg)
    plt.axis('off')
    plt.show()


def fitEllipse(X1, X2, img):
    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    return reg.as_parameters()


if __name__ == "__main__":
    # files = ["data/2.png"]
    with open("gut.txt") as file:
        files = file.readlines()
    for path in files:
        path = path.strip()
        # path = f"data/{i}.png"
        # path = "data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_2uA_spot_only.png"
        img = load_image(path)
        # plt.hist(img.flatten(), 256, [0, 256])
        # plt.show()
        # _, ret1 = cv.threshold(img, 184, 255, cv.THRESH_BINARY)
        # plt.imshow(ret1)
        # plt.show()
        # _, ret = cv.threshold(img, 114, 255, cv.THRESH_BINARY)
        # plt.imshow(ret)
        # plt.show()
        # img = img[100: 400, 100: 400]
        img = cv.GaussianBlur(img, (5, 5), 0)

        # processColumns(img)
        furtherEllipse(img)
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
