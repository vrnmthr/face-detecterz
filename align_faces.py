import argparse

import cv2
import dlib
import numpy as np
from imutils import face_utils

# below is code to align one image


def align_and_extract_faces(img, test=False):
    """
    :param img: standard image ndarray
    :return: array of extracted, aligned faces
    """
    FACE_SIZE = 96

    TEMPLATE = np.load("openface_template.npy")
    TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
    MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN) * FACE_SIZE
    # left eye inside, right eye inside, nose
    LANDMARKS = [39, 42, 33]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gets coordinates of all the faces in the image
    faces = detector(gray, 0)

    result = []

    for f in faces:

        if test:
            (x, y, w, h) = face_utils.rect_to_bb(f)
            face_original = img[y:y + h, x:x + w]
            cv2.imshow("original", face_original)
            cv2.waitKey(0)

        points = predictor(gray, f)
        points = face_utils.shape_to_np(points)
        r_outside, r_inside, l_outside, l_inside, nose = points

        M = cv2.getAffineTransform(np.float32([l_inside, r_inside, nose]),
                                   np.float32(MINMAX_TEMPLATE[LANDMARKS]))

        out = cv2.warpAffine(img, M, (FACE_SIZE, FACE_SIZE))

        if test:
            cv2.imshow("output", out)
            cv2.waitKey(0)

        result.append(out)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to image to align")
    args = vars(parser.parse_args())

    img = cv2.imread(args["path"])
    align_and_extract_faces(img, test=True)
