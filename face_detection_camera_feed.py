import cv2
import mlp
import align_faces
from skimage.transform import resize
import torch.nn
from align_faces.py import align_and_extract_faces
from openface import load_openface
from sklearn import svm

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
openFace = load_openface(device)
classifier = svm.

CONF_THRESHOLD = .5

def promptFaceTraining(seconds):
    print("Capture starting in 5 seconds. Bring your face 2 feet from camera and slowly rotate!")
    time.sleep(5)
    print("Capturing! about to capture" + seconds + " seconds of video")
    time = time.now()
    while time.now() - time < seconds:
        ret, frame = video_capture.read()
        # Trained on the haarcascade_frontalface_default dataset
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceCrop = frame[x:x + w, y:y + h]
        faceCrop = align_and_extract_faces(faceCrop)
        latentFaceVector = openFace(faceCrop)
        print("Not yet implemented")
        # TODO: ahhh save theFACE to disk somewhere!!
    # TODO: ahhhhhhHHHHHH TRAIN THE NETWORK ON NEW FACES!!


def main():
    # to store previous confidences to determine whether a face exists
    CONF_TO_STORE = 30
    prev_conf = np.zeros(CONF_TO_STORE)
    conf_idx = 0
    while True:
        # ret is error code but we don't care about it
        ret, frame = video_capture.read()

        # Trained on the haarcascade_frontalface_default dataset
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceCrop = frame[x:x + w, y:y + h]
            faceCrop = align_and_extract_faces(faceCrop)
            latentFaceVector = openFace(faceCrop)
            classPredicted = np.argmax(softmax)
            confidence = softmax[classPredicted]
            prev_conf[conf_idx] = confidence
            conf_idx += 1
            if np.sum(prev_conf) / CONF_TO_STORE < CONF_THRESHOLD:  # TODO: Create heuristic for confidence and track frame history.
                print("We don't recognize you!")
                #promptFaceTraining()
            # TODO: Tag the frame with facerino -- get the name somehow
            else:
                cv2.putText(frame, str(classPredicted), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow('Camera Feed', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
