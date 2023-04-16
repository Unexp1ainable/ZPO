from xmudro04 import fitEllipse, half_empty_norm
from xrepka07 import detectHalo
from xkunda00 import determineHeight
from helpers import *
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    with open("gut.txt") as file:
        files = file.readlines()
    # files = ["data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_EV_test_3uA.png"]
    for path in files:
        print(path)
        img = load_image(path.strip())
        img = cv.GaussianBlur(img, (5, 5), 0)
        marek_center, marek_size = fitEllipse(img, half_empty_norm)
        halo_center, halo_width, halo_height, halo_phi = detectHalo(img)
        max_count, max_col, max_start_pos, max_end_pos = determineHeight(img)

        draw_ellipse(img, (marek_center[0], max_start_pos + max_count//2),
                     round(marek_size[0]/2), round(max_count/2), 0, 2)
        draw_ellipse(img, halo_center, halo_width, halo_height, halo_phi, 2)

        plt.imshow(img)
        plt.show()
