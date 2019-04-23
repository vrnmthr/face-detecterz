import argparse

import cv2
import dlib
import numpy as np
from imutils import face_utils

# below is code to align one image

FACE_SIZE = 256
# outside corners
LEFT_EYE_LOC = (0.252418718401, 0.331052263829)
RIGHT_EYE_LOC = (0.782865376271, 0.321305281656)
NOSE_LOC = (0.520712933176, 0.634268222208)
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
        l_outside = points[0]
        l_inside = points[1]
        r_inside = points[2]
        r_outside = points[3]
        nose = points[4]

        left_eye_center = [(l_outside[0] + l_inside[0]) / 2, (l_outside[1] + l_inside[1]) / 2]
        right_eye_center = [(r_inside[0] + r_outside[0]) / 2, (r_inside[1] + r_outside[1]) / 2]

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle_between_eyes = np.degrees(np.arctan2(dY, dX)) - 180

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        dist = np.sqrt(dX ** 2 + dY ** 2)
        scale = GOAL_DIST / dist
        arr = [l_outside, r_outside, nose]

        M = cv2.getAffineTransform(np.float32([l_outside, r_outside, nose]),
                                   np.float32([LEFT_EYE_LOC, RIGHT_EYE_LOC, NOSE_LOC]))

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
