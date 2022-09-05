try:
    import cv2
except ImportError as e:
    print(e)

import matplotlib.pyplot as plt

import pathlib
import pytest
import sys
import subprocess


def frame_capture():
    input_file = pathlib.Path().absolute().parent / \
        "bdd100k/videos/test/00a0f008-3c67908e.mov"

    if not input_file.is_file():
        print("Error reading input file.")
        exit(0)

    try:
        subprocess.run(["ffmpeg",
                        "-i",
                        input_file,
                        "-qscale",
                        "0",
                        "tmp/00a0f008-3c67908e.mp4",
                        "-loglevel",
                        "quiet"])

    except subprocess.CalledProcessError as e:
        print(e.output)

    capture = cv2.VideoCapture("tmp/00a0f008-3c67908e.mp4")

    _, axs = plt.subplots(5, 5, figsize=(30, 20))
    axs = axs.flatten()

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    image_index = 0
    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        if frame % 50 == 0:
            axs[image_index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[image_index].set_title("Frame: {}".format(frame))
            axs[image_index].axis("off")
            image_index += 1

    plt.tight_layout()
    plt.savefig("tmp/gPYWXUtVaz")
    capture.release()


@pytest.mark.skipif("cv2" not in sys.modules, reason="requires cv2 library")
def test_frame_capture():
    im = cv2.imread("tmp/gPYWXUtVaz.png")
    assert im.size != 0


if __name__ == "__main__":
    frame_capture()
