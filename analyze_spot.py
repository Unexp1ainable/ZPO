from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from correlation_procedure import fitEllipse
from halo_detection import detectHalo
from helpers import loadImage, loadPixelsize
from histogram_analysis import determineHeight
from argparse import ArgumentParser
import cv2 as cv


def parse_args():
    args = ArgumentParser()
    args.add_argument("image", help="Path to the image")
    return args.parse_args()


def plotMeasurement(img, spot_params, halo_params):
    spot_center, spot_size, spot_phi = spot_params
    halo_center, halo_size, halo_phi = halo_params

    plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    ellipse = Ellipse(
        xy=spot_center, width=spot_size[0], height=spot_size[1], angle=spot_phi,
        edgecolor='b', fc='None', lw=2, label='Spot'
    )
    plt.gca().add_patch(ellipse)

    ellipse2 = Ellipse(
        xy=halo_center, width=halo_size[0]*2, height=halo_size[1]*2, angle=np.rad2deg(halo_phi),
        edgecolor='r', fc='None', lw=2, label='Halo'
    )
    plt.gca().add_patch(ellipse2)
    plt.legend()
    plt.show()

def prefixAndMultiplier(n):
    prefixes = ["", "m", "u", "n", "p"]

    multiplier = 1

    for prefix in prefixes:
        if n > 1:
            return multiplier, prefix
        n *= 1000
        multiplier *= 1000

    return multiplier, prefixes[-1]
    


def printResults(pxsx, pxsy, spot_size, halo_width, halo_height, halo_phi):
    unit = "px" if pxsx == 1. and pxsy == 1. else "m"

    multiplier, prefix = prefixAndMultiplier(spot_size[0]*pxsx)

    print("Spot measurements:")
    print(f"Width: {spot_size[0] * multiplier * pxsx:.3f} {prefix}{unit}")
    print(f"Height: {spot_size[1] * multiplier * pxsy:.3f} {prefix}{unit}")

    print("\nHalo measurements:")
    print(f"Width: {halo_width*2 * multiplier * pxsx:.3f} {prefix}{unit}")
    print(f"Height: {halo_height*2 * multiplier * pxsy:.3f} {prefix}{unit}")
    print(f"Angle: {halo_phi:.3f} rad")


def main(path):
    try:
        img = loadImage(path)
        pxsx, pxsy = loadPixelsize(path)

        img = cv.GaussianBlur(img, (5, 5), 0)

        firstHeight = determineHeight(img)
        spot_center, spot_size = fitEllipse(img, firstHeight)
        halo_center, halo_width, halo_height, halo_phi = detectHalo(img)

        printResults(pxsx, pxsy, spot_size, halo_width, halo_height, halo_phi)
        plotMeasurement(img, (spot_center, spot_size, 0.), (halo_center, (halo_width, halo_height), halo_phi))

    except Exception as e:
        print("Error: " + str(e))


if __name__ == "__main__":
    args = parse_args()
    main(args.image)
