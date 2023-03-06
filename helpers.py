from time import sleep
from typing import Literal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plotImageAs3D(img: np.ndarray) -> None:
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = np.meshgrid(list(range(img.shape[1])), list(range(img.shape[0])))

    ax.plot_surface(X, Y, img, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.contour(X, Y, img, zdir='z', cmap='coolwarm')
    ax.contour(X, Y, img, zdir='x', cmap='coolwarm')
    ax.contour(X, Y, img, zdir='y', cmap='coolwarm')

    plt.show()


def plotImageContours(img: np.ndarray, mode: Literal["rows", "columns"] = "rows") -> None:

    title = ""
    workingCopy = img.copy()
    if mode == "columns":
        workingCopy = workingCopy.T
        title = "Column " + title
    else:
        title = "Row " + title

    width = workingCopy.shape[0]
    title += "{}/" + str(width)

    stahp = [False]
    shownLine = [0]
    anim = [True]

    def updateCanvas():
        plt.clf()
        plt.title(title.format(shownLine[0]+1))
        plt.ylim([0, 255])
        plt.plot(workingCopy[shownLine[0]])
        plt.pause(0.01)
        plt.draw()

    def on_press(event):
        if event.key == "escape":
            stahp[0] = True
        elif event.key == "right":
            if shownLine[0] < width-1:
                shownLine[0] += 1
                updateCanvas()

        elif event.key == "left":
            if shownLine[0] > 0:
                shownLine[0] -= 1
                updateCanvas()

        elif event.key == " ":
            anim[0] = not anim[0]
        else:
            print(event.key)

    plt.gca().figure.canvas.mpl_connect('key_press_event', on_press)
    plt.ion()
    updateCanvas()
    while not stahp[0]:
        if stahp[0]:
            break

        if anim[0]:
            shownLine[0] = (shownLine[0] + 1) % width
            updateCanvas()
        else:
            plt.pause(0.5)
