import argparse

import cv2
import dlib
import numpy as np
from imutils import face_utils

# below is code to align one image

FACE_SIZE = 256
LEFT_EYE_LOC = (.35, .35)
RIGHT_EYE_LOC = (.65, .65)
GOAL_DIST = 76.8

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


def align_and_extract_faces(img, test=False):
    """
    :param img: standard image ndarray
    :return: array of extracted, aligned faces
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gets coordinates of all the faces in the image
    faces = detector(gray, 0)

    result = []

    for f in faces:
        (x, y, w, h) = face_utils.rect_to_bb(f)
        face_original = img[y:y + h, x:x + w]

        if test:
            cv2.imshow("original", face_original)
            cv2.waitKey(0)

        points = predictor(gray, f)
        points = face_utils.shape_to_np(points)
        l1 = points[0]
        l2 = points[1]
        r1 = points[2]
        r2 = points[3]

        left_eye_center = [(l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2]
        right_eye_center = [(r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2]

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle_between_eyes = np.degrees(np.arctan2(dY, dX)) - 180

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        dist = np.sqrt(dX ** 2 + dY ** 2)
        scale = GOAL_DIST / dist

        M = cv2.getRotationMatrix2D(eyes_center, angle_between_eyes, scale)
        tX = FACE_SIZE * .5
        tY = FACE_SIZE * LEFT_EYE_LOC[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        out = cv2.warpAffine(img, M, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_CUBIC)

        if test:
            cv2.imshow("output", out)
            cv2.waitKey(0)

        result.append(out)

    return np.asarray(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to image to align")
    args = vars(parser.parse_args())

    img = cv2.imread(args["path"])
    align_and_extract_faces(img, test=True)
