import time

import cv2
import numpy as np
import torch.nn

from align_faces import extract_faces, align_faces
from openface import load_openface, preprocess_single, preprocess_batch
from tqdm import tqdm

video_capture = cv2.VideoCapture(0)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
openFace = load_openface(device)
# clf = svm.SVC(kernel="linear", C=1.6)

CONF_THRESHOLD = .5

idxToName = {}  # TODO: Write function to populate.


def capture_faces(seconds=5, sampling_duration=0.1):
    print("Capturing! about to capture {} seconds of video".format(seconds))
    start_time = time.time()

    # face_locs stores the bounding box coordinates
    face_locs = []
    # frames stores the actual images
    frames = []

    while time.time() - start_time < seconds:
        ret, frame = video_capture.read()
        if ret:
            faces = extract_faces(frame)
            if len(faces) == 1:
                frames.append(frame)
                face_locs.append(faces[0])
                print("Taken sample.")

            if len(faces) == 0:
                print("No faces found.")

            if len(faces) > 1:
                print("We have found {} faces, and there should only be one".format(len(faces)))
        else:
            print("ERROR: No sample taken")
        # lock the loop to system time
        time.sleep(sampling_duration - ((time.time() - start_time) % sampling_duration))

    # extract the faces afterwards
    print("Extracting faces from samples")
    samples = []
    for i in tqdm(range(len(face_locs)), total=len(face_locs)):
        rect = face_locs[i]
        frame = frames[i]
        sample = align_faces(frame, [rect])[0]
        samples.append(sample)
        cv2.imshow("samples", sample)
        cv2.waitKey(0)

    return samples


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
            faceCrop = frame[y:y + h, x:x + w]
            faceCrop = align_and_extract_faces(faceCrop)
            faceCrop = preprocess_single(faceCrop)
            latentFaceVector = openFace(faceCrop)
            latentFaceVector = latentFaceVector.detach().numpy()
            pred = clf.predict([latentFaceVector])
            print(pred)
            # softmax = nn.Softmax(result)
            # classPredicted = np.argmax(softmax)
            # confidence = softmax[classPredicted]
            # prev_conf[conf_idx] = confidence
            conf_idx += 1
            if np.sum(
                    prev_conf) / CONF_TO_STORE < CONF_THRESHOLD:  # TODO: Create heuristic for confidence and track frame history.
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


if __name__ == "__main__":
    capture_faces(seconds=10, sampling_duration=0.25)
