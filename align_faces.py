import numpy as np
import cv2
import dlib
import argparse
from imutils import face_utils
from imutils import resize

# below is code to align one image

FACE_SIZE = 256
LEFT_EYE_LOC = (.35, .35)
RIGHT_EYE_LOC = (.65, .65)
GOAL_DIST = 76.8

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
img = cv2.imread("test_1_align.jpeg")
#cv2.imshow("hi", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 0)

for f in faces:
    print("doing a face")
    print("total faces: " + str(len(faces)))
    (x, y, w, h) = face_utils.rect_to_bb(f)
    face_original = img[y:y+h, x:x+w]
    cv2.imshow("original face", face_original)
    cv2.waitKey(0)
    points = predictor(gray, f)
    points = face_utils.shape_to_np(points)
    l1 = points[0]
    l2 = points[1]
    r1 = points[2]
    r2 = points[3]
    print(points)

    left_eye_center = [(l1[0]+l2[0])/2, (l1[1]+l2[1])/2]
    right_eye_center = [(r1[0]+r2[0])/2, (r1[1]+r2[1])/2]

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle_between_eyes = np.degrees(np.arctan2(dY, dX))-180

    eyes_center = ((left_eye_center[0]+right_eye_center[0])//2,
        (left_eye_center[1]+right_eye_center[1])//2)

    print(eyes_center)

    dist = np.sqrt(dX**2 + dY**2)
    scale = GOAL_DIST/dist

    M = cv2.getRotationMatrix2D(eyes_center, angle_between_eyes, scale)
    tX = FACE_SIZE*.5
    tY = FACE_SIZE*LEFT_EYE_LOC[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    print(M)

    out = cv2.warpAffine(img, M, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_CUBIC)
    print(out)
    cv2.imshow("output???", out)
    cv2.waitKey(0)







#
