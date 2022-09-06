import pandas as pd
import cv2

import pathlib


def write_annotations(img, frame, video_labels):

    annotations_color_mapping = {
        "car": (0, 0, 255),
        "truck": (0, 0, 100),
        "pedestrian": (255, 0, 0),
        "other vehicle": (0, 0, 150),
        "rider": (200, 0, 0),
        "bicycle": (0, 255, 0),
        "other person": (200, 0, 0),
        "trailer": (0, 150, 150),
        "motorcycle": (0, 150, 0),
        "bus": (0, 0, 100),
    }

    max_frame = video_labels.query("video_frame <= {}".format(frame))[
        "video_frame"].max()
    frame_labels = video_labels.query("video_frame == {}".format(max_frame))
    for _, d in frame_labels.iterrows():
        pt1 = int(d["box2d.x1"]), int(d["box2d.y1"])
        pt2 = int(d["box2d.x2"]), int(d["box2d.y2"])
        color = annotations_color_mapping[d["category"]]
        img = cv2.rectangle(img, pt1, pt2, color, 3)

    return img


def get_video_labels():
    annotation_location = pathlib.Path().absolute().parent / \
        "bdd100k/mot_labels.csv"
    label_annotations_csv = pd.read_csv(annotation_location, low_memory=False)

    video_labels = (
        label_annotations_csv.query(
            'videoName == "026c7465-309f6d33"').reset_index(drop=True).copy()
    )
    video_labels["video_frame"] = (
        video_labels["frameIndex"] * 11.9).round().astype("int")

    return video_labels


def frame_annotations():
    video_labels = get_video_labels()

    capture = cv2.VideoCapture("tmp/026c7465-309f6d33.mp4")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_nbr = 2300  # which frame within the video should the test be conducted
    annotated_img = None
    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        if frame == frame_nbr:
            annotated_img = img.copy()
            cv2.imwrite("tmp/frame.png", img)
            annotated_img = write_annotations(img, frame, video_labels)
            cv2.imwrite("tmp/annframe.png", annotated_img)

    capture.release()


if __name__ == "__main__":
    frame_annotations()
