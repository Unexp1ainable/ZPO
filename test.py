import matplotlib
import analyze_spot

if __name__ == "__main__":
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    # with open("dev/gut.txt") as file:
    with open("dev/series2.txt") as file:
        # with open("note.txt") as file:
        files = file.readlines()
    # files = ["data/finalizace - FIB spots/121-0319X manual 5s/_2022_121-0319 S8251X, CN_images_FIB_Spots_EV_test_3uA.png"]
    # files = ["data/finalizace - FIB spots/122-0007X manual 5s/_2022_122-0007 S8252X, US_images_FIB_Spots_EV_test_3uA.png"]
    # files = ["data/series2/set5/30keV50nA-11.png"]
    # files = ["data/series2/set1/30keV2.5nA.png"]
    files = files[10:]
    files = ["data/series2/set5/30keV250pA-21.png"]
    for path in files:
        try:
            print(f"Processing: {path}")
            path = path.strip()
            analyze_spot.main(path)
        except KeyboardInterrupt as e:
            print(e)
            exit(1)
        except Exception as e:
            print(e)
