"""
=========================================================================
Brief: FIB spot measurement for ZPO project
Authors:
    Marek Mudroň (xmudro04)
    Matej Kunda  (xkunda00)
    Samuel Repka (xrepka07)
File: analyze_spot.py
Date: April 2023
=========================================================================
"""

from typing import Tuple
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from correlation_procedure import fitEllipse
from halo_detection import detectHalo
from helpers import loadImage, loadPixelsize
from histogram_analysis import determineHeight
from argparse import ArgumentParser, Namespace
import cv2 as cv


def parse_args() -> Namespace:
    """Parse arguments given to the script

    Returns:
        Namespace: Parsed arguments
    """
    args = ArgumentParser()
    args.add_argument("image", help="Path to the image")
    return args.parse_args()


def plotMeasurement(
        img: np.ndarray, spot_params: Tuple[Tuple[float, float],
                                            Tuple[float, float],
                                            float],
        halo_params: Tuple[Tuple[float, float],
                           Tuple[float, float],
                           float]) -> None:
    """Show measurements overlayed over the image

    Args:
        img (np.ndarray): Input image
        spot_params (Tuple[(int,int), (int,int)): Parameters of the spot ellipse. ((xpos,ypos), (width, height), angle). Angle should be in radians.
        halo_params (Tuple[(int,int), (int,int)): Parameters of the halo ellipse. ((xpos,ypos), (width, height), angle). Angle should be in radians.
    """
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


def prefixAndMultiplier(n: float) -> Tuple[int, str]:
    """Determine best SI prefix and multiplier based on n.

    Args:
        n (float): Number for which to determine prefix

    Returns:
        Tuple[int,str]: (multiplier, prefix)
    """
    prefixes = ["", "m", "u", "n", "p"]

    multiplier = 1

    for prefix in prefixes:
        if n > 1:
            return multiplier, prefix
        n *= 1000
        multiplier *= 1000

    return multiplier, prefixes[-1]


def printResults(
        pxsx: float, pxsy: float, spot_size: Tuple[float, float],
        halo_width: float, halo_height: float, halo_phi: float) -> None:
    """Print results to stdin

    Args:
        pxsx (float): Pixelsize X
        pxsy (float): Pixelsize Y
        spot_size (Tuple[float,float]): Spot size (width, height)
        halo_width (float): Halo width
        halo_height (float): Halo height
        halo_phi (float): Halo angle
    """
    unit = "px" if pxsx == 1. and pxsy == 1. else "m"

    multiplier, prefix = prefixAndMultiplier(spot_size[0]*pxsx)

    print("Spot measurements:")
    print(f"Width: {spot_size[0] * multiplier * pxsx:.3f} {prefix}{unit}")
    print(f"Height: {spot_size[1] * multiplier * pxsy:.3f} {prefix}{unit}")

    print("\nHalo measurements:")
    print(f"Width: {halo_width*2 * multiplier * pxsx:.3f} {prefix}{unit}")
    print(f"Height: {halo_height*2 * multiplier * pxsy:.3f} {prefix}{unit}")
    print(f"Angle: {halo_phi:.3f} rad")


def main(path: str) -> None:
    """Main function

    Args:
        path (str): path to image
    """
    # load required data
    img = loadImage(path)
    pxsx, pxsy = loadPixelsize(path)

    # smooth the image
    img = cv.GaussianBlur(img, (5, 5), 0)

    # find parameters
    firstHeight = determineHeight(img)
    spot_center, spot_size = fitEllipse(img, firstHeight)
    halo_center, halo_width, halo_height, halo_phi = detectHalo(img)

    # report measurements
    printResults(pxsx, pxsy, spot_size, halo_width, halo_height, halo_phi)
    plotMeasurement(img, (spot_center, spot_size, 0.), (halo_center, (halo_width, halo_height), halo_phi))


if __name__ == "__main__":
    args = parse_args()
    main(args.image)
