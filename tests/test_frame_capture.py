import cv2
import matplotlib.pyplot as plt

import pathlib
import subprocess


if __name__ == "__main__":

    input_file = pathlib.Path().absolute().parent / \
        "bdd100k/videos/train/00a0f008-3c67908e.mov"

    try:
        subprocess.run(["ffmpeg",
                        "-i",
                        input_file,
                        "-qscale",
                        "0",
                        "00a0f008-3c67908e.mp4",
                        "-loglevel",
                        "quiet"])

    except subprocess.CalledProcessError as e:
        print(e.output)

    capture = cv2.VideoCapture('00a0f008-3c67908e.mp4')

    ret, img = capture.read()

    fig, axs = plt.subplots(5, 5, figsize=(30, 20))
    axs = axs.flatten()

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    image_index = 0
    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        if frame % 100 == 0:
            axs[image_index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[image_index].set_title("Frame: {}".format(frame))
            axs[image_index].axis("off")
            image_index += 1

    plt.tight_layout()
    plt.savefig("test.png")
    capture.release()
