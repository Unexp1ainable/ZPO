from xmudro04 import fitEllipse, half_empty_norm
from xrepka07 import detectHalo
from xkunda00 import determineHeight
from helpers import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import *


if __name__ == "__main__":
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    with open("series2.txt") as file:
    # with open("gut.txt") as file:
        files = file.readlines()
    # files = ["data/series2/set5/30keV50nA-11.png"]
    # files = ["data/series2/set1/30keV2.5nA.png"]
    for path in files:
        path = path.strip()
        print(path)
        img = loadImage(path)
        img = cv.GaussianBlur(img, (5, 5), 0)
        try:

            height, max_col, max_start_pos, max_end_pos = determineHeight(img)
            marek_center, marek_size = fitEllipse(img, half_empty_norm, height)
            halo_center, halo_width, halo_height, halo_phi = detectHalo(img)

            plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
            ellipse = Ellipse(
                xy=marek_center, width=marek_size[0], height=marek_size[1], angle=0,
                edgecolor='b', fc='None', lw=2, label='Spot'
            )
            plt.gca().add_patch(ellipse)

            ellipse2 = Ellipse(
                xy=halo_center, width=halo_width*2, height=halo_height*2, angle=np.rad2deg(halo_phi),
                edgecolor='r', fc='None', lw=2, label='Halo'
            )
            plt.gca().add_patch(ellipse2)
            plt.legend()

            # draw_ellipse(img, marek_center, marek_size[0]//2, height//2, 0,2)
            # draw_ellipse(img, halo_center, halo_width, halo_height, halo_phi,2)

            plt.show()
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            exit(1)
        except Exception as e:
            print("^ did not make it. ):")
            print(e)
