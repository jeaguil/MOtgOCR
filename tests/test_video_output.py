import cv2

from test_frame_annotations import get_video_labels, write_annotations

import logging
from util import setup_cli_logging


def video_output():
    VIDEO_CODEC = "mp4v"
    fps = 59.94
    width = 1280
    height = 720

    out = cv2.VideoWriter("tmp/ann_out_test.mp4", cv2.VideoWriter_fourcc(
        *VIDEO_CODEC), fps, (width, height))

    capture = cv2.VideoCapture("tmp/026c7465-309f6d33.mp4")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    video_labels = get_video_labels()

    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        img = write_annotations(img, frame, video_labels)
        out.write(img)

    out.release()
    capture.release()

    logging.info("Annotated video test saved to tmp/ann_out_test")


if __name__ == "__main__":
    setup_cli_logging()
    video_output()
