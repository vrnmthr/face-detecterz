import cv2
#import mlp
import torch.nn
<<<<<<< HEAD
from align_faces import align_and_extract_faces
from openface import load_openface, preprocess_single, preprocess_batch
import numpy as np
import time
=======
from align_faces.py import align_and_extract_faces
from openface import load_openface, preprocess_single
from sklearn import svm
from classifier import train, test
>>>>>>> 3874d60a042aa29ebf9eaa6ccfdc772ebea93d67

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
openFace = load_openface(device)
clf = svm.SVC(kernel="linear", C=1.6)

CONF_THRESHOLD = .5

idxToName = {} #TODO: Write function to populate.

def promptFaceTraining(seconds):
    print("Capture starting in 5 seconds. Bring your face 2 feet from camera and slowly rotate!")
    time.sleep(5)
    print("Capturing! about to capture " + str(seconds) + " seconds of video")
    t = time.time()
    samples = []
    count = 0
    while time.time() - t < seconds:
        ret, frame = video_capture.read()
        # Trained on the haarcascade_frontalface_default dataset
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if (len(faces) == 1): #yeet
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                faceCrop = frame[y:y + h, x:x + w]
                cv2.imshow("yeet1", faceCrop)
                cv2.waitKey(0)
                faceCrop = align_and_extract_faces(faceCrop)[0]
                samples.append(faceCrop)
                cv2.imshow("yeet2", faceCrop)
                cv2.waitKey(0)
                cv2.imwrite("test_faces/eleanor" + str(count) + ".png", np.float32(faceCrop))
                count += 1
        else: #yote
            print("We have found " + str(len(faces)) + ", and there should only be one. Try again.")
    # batch
    faces_proc = preprocess_batch(samples)
    latentFaceVectors = openFace(faces_proc)
    np.save("test_faces/eleanor.npy", latentFaceVectors)



def main():
    # to store previous confidences to determine whether a face exists
    CONF_TO_STORE = 30
    prev_conf = np.zeros(CONF_TO_STORE)
    conf_idx = 0
    while True:
        # ret is error code but we don't care about it
        ret, frame = video_capture.read()
        # cv2.imshow("wow", frame)
        # cv2.waitKey(0)

        #Trained on the haarcascade_frontalface_default dataset
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceCrop = frame[y:y + h, x:x + w]
            faceCrop = align_and_extract_faces(faceCrop)[0]
            cv2.imshow("proposed face extraction", faceCrop)
            faceCrop = preprocess_single(faceCrop)
            latentFaceVector = openFace(faceCrop)
            latentFaceVector = latentFaceVector.detach().numpy()
            pred = clf.predict([latentFaceVector])
            print(pred)
            # softmax = nn.Softmax(result)
            # classPredicted = np.argmax(softmax)
            #confidence = softmax[classPredicted]
            #prev_conf[conf_idx] = confidence
            conf_idx += 1
            if np.sum(prev_conf) / CONF_TO_STORE < CONF_THRESHOLD:  # TODO: Create heuristic for confidence and track frame history.
                print("We don't recognize you!")
                promptFaceTraining(5)
            # TODO: Tag the frame with facerino -- get the name somehow
            else:
                cv2.putText(frame, str(classPredicted), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(0)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": promptFaceTraining(5)
