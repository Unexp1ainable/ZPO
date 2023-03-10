import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from helpers import *
from kernels import *


# def fitEllipseFixedSize(img, plot=False):
#     img_width, img_height = img.shape
#     kernel = getEdgedKernelNormalized(true_width,2)
#     mask = cv.filter2D(img, cv.CV_32F, kernel)
#     sobel = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)
#     amin = np.unravel_index(np.argmax(sobel, axis=None), sobel.shape)
#     center = (amin[1],amin[0])
#     if plot:
#         plt.imshow(img)
#         plt.show()
#         plt.imshow(kernel)
#         plt.show()
#         plotImageAs3D(sobel)
#         width, height = get_ellipse_size(true_width)
#         fig, ax = plt.subplots(2,3,figsize=(18, 12))
#         ax[0,0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
#         ax[0,0].axis('equal')
#         ellipse = Ellipse(
#             xy=center, width=width, height=height, angle=0,
#             edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
#         )
#         ax[0,0].add_patch(ellipse)

#         ax[0,1].imshow(sobel)
#         ax[0,1].axis('equal')
#         ellipse = Ellipse(
#             xy=center, width=width, height=height, angle=0,
#             edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
#         )
#         ax[0,1].add_patch(ellipse)
#         ax[0,2].imshow(mask)
#         ax[0,2].axis('equal')
#         ellipse = Ellipse(
#             xy=center, width=width, height=height, angle=0,
#             edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
#         )
#         ax[0,2].add_patch(ellipse)

#         center_x, center_y = center
#         left, right = max(0,center_x - width//2), min(center_x + width//2, img_width-1)
#         top, bottom = max(0,center_y - height//2), min(center_y+ height//2,img_height-1)
#         crop = img[top:bottom, left:right]
#         ax[1,0].imshow(crop)
#         ax[1,0].axis('equal')
#         plt.show()



def fitEllipse(img, kernel_func, plot=False):
    img_width, img_height = img.shape
    min_width = int(img_width*0.1)
    max_width = int(img_width*0.8)
    best_score = -np.inf
    best_center = None
    best_width = None
    best_sobel = None
    best_mask = None
    best_kernel = None
    for width in range(min_width, max_width):
        kernel = kernel_func(width,2)
        mask = cv.filter2D(img, cv.CV_32F, kernel)
        sobel = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)
        amin = np.unravel_index(np.argmax(sobel, axis=None), sobel.shape)
        center = (amin[1],amin[0])
        
        if mask[amin] > best_score:
            #print(f"width: {width}\tcenter: {center}\tscore {sobel[amin]}")
            best_score = mask[amin]
            best_center = center
            best_width = width
            best_sobel = sobel
            best_mask = mask
            best_kernel = kernel

    print("="*50)
    print(f"width: {best_width}\ttrue_width={paths[INDEX][1]}\tcenter: {best_center}\tscore {best_score}")
    
    width, height = get_ellipse_size(best_width)
    center = best_center
    if plot:
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(best_kernel)
        # plt.colorbar()
        # plt.show()
        # plotImageAs3D(best_sobel)
        
        fig, ax = plt.subplots(2,3,figsize=(18,12))
        ax[0,0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        ax[0,0].axis('equal')
        ax[0,0].set_title("Image")
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0,0].add_patch(ellipse)
        ax[0,1].imshow(best_sobel)
        ax[0,1].axis('equal')
        ax[0,1].set_title("Sobel filter")
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0,1].add_patch(ellipse)

        ax[0,2].imshow(best_mask)
        ax[0,2].axis('equal')
        ax[0,2].set_title("Convolved image")
        ellipse = Ellipse(
            xy=center, width=width, height=height, angle=0,
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ax[0,2].add_patch(ellipse)

        center_x, center_y = center
        left, right = max(0,center_x - width//2), min(center_x + width//2, img_width-1)
        top, bottom = max(0,center_y - height//2), min(center_y+ height//2,img_height-1)
        crop = img[top:bottom, left:right]
        ax[1,0].imshow(crop)
        ax[1,0].axis('equal')
        ax[1,0].set_title("Cropped section")

        cb_han = ax[1,1].imshow(best_kernel)
        ax[1,1].axis('equal')
        ax[1,1].set_title("Best filter")
        fig.colorbar(cb_han,ax=ax[1,1])
        plt.show()



if __name__ == "__main__":
    

    paths = [
        ("data/0.png",102, False),
        ("data/1.png",14,False),
        ("data/2.png",85, False),
        ("data/3.png",18, False),
        ("data/6.png",130, False),
        ("data/7.png",80, False),
        ("data/5.png",80, False),
        ("data/8.png",128, False),
        ("data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_15kV_50nA.png", 40, True),
        ("data/finalizace - FIB spots/121-0201G manual looks like 1s though/_2022_121-0201 S9251G, CN_images_FIB_Spots_30 keV; 50 pA.png", 85, True),
        ("data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_EV_test_300nA.png", 113, True),
        ("data/finalizace - FIB spots/122-0007X manual 5s/_2022_122-0007 S8252X, US_images_FIB_Spots_EV_test_3uA.png", 78, True),
        ("data/finalizace - FIB spots/122-0049X manual 5s/_2022_122-0049 S8254X, GB_images_FIB_Spots_EV_test_300nA.png", 140, True),
    ]

    for INDEX in range(0,len(paths)):
        path = paths[INDEX][0]
        true_width = paths[INDEX][1]
        has_label = paths[INDEX][2]

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if(has_label):
            img = img[:-79]

        # PLAYGROUND
        # possible values for kernels are
        # - half_empty
        # - half_empty_norm
        # - edged
        # - edged_norm
        # - half_negative
        # - half_negative_norm
        kernel = half_negative_norm
        fitEllipse(img, kernel, True)
