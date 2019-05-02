import argparse

import cv2
import dlib
import numpy as np
from imutils import face_utils

FACE_SIZE = 96
TEMPLATE = np.load("data/openface_template.npy")
TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN) * FACE_SIZE
# left eye inside, right eye inside, nose
LANDMARKS = [39, 42, 33]
DETECTOR = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
PREDICTOR = dlib.shape_predictor("data/shape_predictor_5_face_landmarks.dat")
CONFIDENCE = 0.75


def align_faces(img, rects, test=False):
    """
    Aligns
    :param img: grey image
    :param faces: diagonal coordinates of bounding boxes for faces
    :return: array of extracted, aligned faces
    """
    result = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for rect in rects:

        if test:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face_original = img[y:y + h, x:x + w]
            cv2.imshow("original", face_original)
            cv2.waitKey(0)

        points = PREDICTOR(gray, rect)
        points = face_utils.shape_to_np(points)
        r_outside, r_inside, l_outside, l_inside, nose = points
        M = cv2.getAffineTransform(np.float32([l_inside, r_inside, nose]),
                                   np.float32(MINMAX_TEMPLATE[LANDMARKS]))
        out = cv2.warpAffine(img, M, (FACE_SIZE, FACE_SIZE))

        if test:
            cv2.imshow("output", out)
            cv2.waitKey(0)

        result.append(out)

    result = np.asarray(result)
    return result


def extract_faces(img):
    """
    Extracts all faces from an image
    Inspiration from: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/#post_downloads
    :param img: standard openCV extracted image
    :return: list of dlib rectangles
    """
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    DETECTOR.setInput(blob)
    detections = DETECTOR.forward()
    filtered = []

    for i in range(detections.shape[2]):
        # extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > CONFIDENCE:
            # compute box_coords as (startX, startY, endX, endY)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            # convert into dlib rectangle
            rect = dlib.rectangle(startX, startY, endX, endY)
            filtered.append(rect)

    return filtered


def align_and_extract_faces(img, test=False):
    """
    :param img: standard image ndarray
    :return: array of extracted, aligned faces
    """
    rects = extract_faces(img)
    aligned = align_faces(img, rects)
    return aligned


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to image to align")
    args = vars(parser.parse_args())

    img = cv2.imread(args["path"])
    align_and_extract_faces(img, test=True)
