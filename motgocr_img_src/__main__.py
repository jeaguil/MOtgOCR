""" performance of yolov5 custom model on images/ """

import torch
import os
import cv2

import xml.etree.ElementTree as ET


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    true_bndbox_crds = ()
    for child in root.iter("bndbox"):
        xmin = int(child.find("xmin").text)
        ymin = int(child.find("ymin").text)
        xmax = int(child.find("xmax").text)
        ymax = int(child.find("ymax").text)

        true_bndbox_crds = (xmin, ymin, xmax, ymax)

    return true_bndbox_crds


def crd_difference(true_bndbox_crds, found_bndbox_crds):
    def percentage_difference(v1, v2):
        return 100 * ((abs(v1 - v2)) / ((v1+v2) / 2.0))

    crds_dif_lst = []
    for i in range(len(found_bndbox_crds)):
        if true_bndbox_crds[i] == 0 or found_bndbox_crds[i] == 0:
            return 0
        crds_dif_lst += [percentage_difference(true_bndbox_crds[i],
                                               found_bndbox_crds[i])]

    return sum(crds_dif_lst) / len(crds_dif_lst)


if __name__ == "__main__":

    model_weights = os.path.join(os.getcwd(), "weights/10epoch/best.pt")
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', model_weights)

    imgs_path = os.path.join(os.getcwd(), "images")
    annotations_path = os.path.join(os.getcwd(), "annotations")
    tmp = os.path.join(os.getcwd(), "motgocr_img_src/tmp")
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    crds_dif_lst = []
    for img in os.listdir(imgs_path):
        f = os.path.join(imgs_path, img)
        ret = cv2.imread(f)

        f_x_shape = ret.shape[1]
        f_y_shape = ret.shape[0]

        model_on_f = yolo(ret)

        try:
            label, license_plate_coordinates = (
                model_on_f.xyxyn[0][:, -1], model_on_f.xyxyn[0][:, :-1])
            r = license_plate_coordinates[0]
        except IndexError:
            # index error if model did not find license plates.
            # image 1/1: 270x471 (no detections)
            continue

        crds = (x1, y1, x2, y2) = (int(r[0] * f_x_shape), int(r[1] * f_y_shape),
                                   int(r[2] * f_x_shape), int(r[3] * f_y_shape))

        base_name = os.path.splitext(img)[0]

        # parse xml to check if calculated bndbox coordinates
        # are within range of true bndbox coordinates in annotations.
        true_crds = parse_xml(os.path.join(
            annotations_path, base_name + ".xml"))

        crds_dif_lst += [crd_difference(true_crds, crds)]

        pt1 = int(crds[0]), int(
            crds[1])
        pt2 = int(crds[2]), int(
            crds[3])

        img = ret.copy()
        img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(
            os.getcwd(), "motgocr_img_src/tmp/" + base_name + ".png"), img)

    print("Custom yolov5 model finished with an average percent differnce of {:.2f}% between experimental coordinates and true coordinates.".format(
        sum(crds_dif_lst) / len(crds_dif_lst)))
