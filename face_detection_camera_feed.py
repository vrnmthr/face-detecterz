import cv2
import sys

faceCascade = cv2.CascadeClassifier("/Users/Eleanor/Desktop/CryptoBeat Videos/haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:

    #ret is error code but we don't care about it
    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Trained on the haarcascade_frontalface_default dataset
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Camera Feed', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
