import torch
import os
import cv2

if __name__ == "__main__":

    model_weights = os.path.join(os.getcwd(), "weights/10epoch/best.pt")
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', model_weights)

    imgs_path = os.path.join(os.getcwd(), "images")

    for img in os.listdir(imgs_path):
        f = os.path.join(imgs_path, img)
        ret = cv2.imread(f)

        f_x_shape = ret.shape[1]
        f_y_shape = ret.shape[0]

        model_on_f = yolo(ret)

        label, license_plate_coordinates = (
            model_on_f.xyxyn[0][:, -1], model_on_f.xyxyn[0][:, :-1])

        try:
            r = license_plate_coordinates[0]

            crds = (x1, y1, x2, y2) = (int(r[0] * f_x_shape), int(r[1] * f_y_shape),
                                       int(r[2] * f_x_shape), int(r[3] * f_y_shape))

            base_name = os.path.splitext(img)[0]

            pt1 = int(crds[0]), int(
                crds[1])
            pt2 = int(crds[2]), int(
                crds[3])

            img = ret.copy()
            img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(
                os.getcwd(), 'motgocr_img_src/tmp/' + base_name + '.png'), img)
        except:
            pass
