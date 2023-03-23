import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

path = "data/0.png"

def processImage(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img = cv.convertScaleAbs(img, alpha=1.2, beta=0)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (7, 7), 0)
    ret1, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    ret, o1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    masked = cv.bitwise_and(img, img, mask=th)

    #ret2, th2 = cv.threshold(masked, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    cv.floodFill(masked, None, (0, 0), 255)


    edges = cv.Canny(th, 100, 120)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.axvline(ret1, color='red', linewidth=1)
    #plt.axvline(ret2, color='yellow', linewidth=1)
    plt.show()

    images = np.concatenate((img, th, o1), axis=1)
    cv.imshow(f'img - {path}', images)
    cv.waitKey(0)


if __name__ == "__main__":
    #processImage('./data/6.png')
    #exit()

    path = './data'

    for fileName in os.listdir(path):
        file = os.path.join(path, fileName)
        processImage(file)






# th = 80
# max_val = 255
# ret, o1 = cv.threshold(img, th, max_val, cv.THRESH_BINARY)
# #cv.putText(o1, "Thresh_Binary", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
# ret, o2 = cv.threshold(img, th, max_val, cv.THRESH_BINARY_INV)
# #cv.putText(o2, "Thresh_Binary_inv", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
# ret, o3 = cv.threshold(img, th, max_val, cv.THRESH_TOZERO)
# #cv.putText(o3, "Thresh_Tozero", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
# ret, o4 = cv.threshold(img, th, max_val, cv.THRESH_TOZERO_INV)
# #cv.putText(o4, "Thresh_Tozero_inv", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
# ret, o5 = cv.threshold(img, th, max_val, cv.THRESH_TRUNC)
# #cv.putText(o5, "Thresh_trunc", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
# ret, o6 = cv.threshold(img, th, max_val, cv.THRESH_OTSU)
# #cv.putText(o6, "Thresh_OSTU", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)
#
# thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 3)
# thresh3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 5)
# thresh4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 4)
#
# final = np.concatenate((o1, o2, o3), axis=1)
# final1 = np.concatenate((o4, o5, o6), axis=1)
#
# final2 = np.concatenate((thresh1, thresh2, thresh3, thresh4), axis=1)
#
# #images = np.concatenate((img, th), axis=1)
# cv.imshow(f'img - {path}', final2)
# #cv.imshow(f'img2 - {path}', final1)
# cv.waitKey(0)


