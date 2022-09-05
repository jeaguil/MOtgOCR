import pandas as pd
import cv2

import pathlib


def frame_annotations():
    annotation_location = pathlib.Path().absolute().parent / \
        "bdd100k/mot_labels.csv"
    label_annotations_csv = pd.read_csv(annotation_location, low_memory=False)

    video_labels = (
        label_annotations_csv.query(
            'videoName == "00a0f008-3c67908e"').reset_index(drop=True).copy()
    )
    video_labels["video_frame"] = (
        video_labels["frameIndex"] * 11.9).round().astype("int")

    capture = cv2.VideoCapture("tmp/00a0f008-3c67908e.mp4")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_nbr = 800  # which frame within the video should the test be conducted
    annotated_img = None
    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        if frame == frame_nbr:
            annotated_img = img.copy()
            cv2.imwrite("tmp/frame{}.png".format(frame_nbr), img)

    max_frame = video_labels.query("video_frame <= {}".format(frame_nbr))[
        "video_frame"].max()
    frame_labels = video_labels.query("video_frame == {}".format(max_frame))
    for _, d in frame_labels.iterrows():
        pt1 = int(d["box2d.x1"]), int(d["box2d.y1"])
        pt2 = int(d["box2d.x2"]), int(d["box2d.y2"])
        annotated_img = cv2.rectangle(annotated_img, pt1, pt2, (0, 0, 255), 3)

    cv2.imwrite("tmp/frame{}_annotated.png".format(frame_nbr), annotated_img)
    capture.release()


if __name__ == "__main__":
    frame_annotations()
