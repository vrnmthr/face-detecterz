import time

import cv2
import numpy as np
import torch.nn
from imutils import face_utils
from tqdm import tqdm

from align_faces import extract_faces, align_faces
from openface import load_openface, preprocess_batch

video_capture = cv2.VideoCapture(0)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
openFace = load_openface(device)
# clf = svm.SVC(kernel="linear", C=1.6)

CONF_THRESHOLD = .75

idxToName = {}  # TODO: Write function to populate.


def capture_faces(seconds=5, sampling_duration=0.1, debug=False):
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
        if debug:
            cv2.imshow("samples", sample)
            cv2.waitKey(0)

    return samples


def main():
    # to store previous confidences to determine whether a face exists
    CONF_TO_STORE = 30
    prev_conf = []
    conf_idx = 0

    while True:
        # ret is error code but we don't care about it
        ret, frame = video_capture.read()
        if ret:
            # extract and align faces
            rects = extract_faces(frame)
            faces = align_faces(frame, rects)

            # generate embeddings
            tensor = preprocess_batch(faces)
            embeddings = openFace(tensor)

            # predict classes for all faces
            pred = clf.predict(embeddings)
            # TODO: convert integers to names
            # pred_names =
            print(pred)

            # determine if we need to trigger retraining
            # TODO: get confidence here
            prev_conf.append(confidence)
            if len(prev_conf) > CONF_TO_STORE:
                prev_conf.pop(0)
            if np.sum(prev_conf) / CONF_TO_STORE < CONF_THRESHOLD and len(
                    prev_conf) == CONF_TO_STORE:  # TODO: Create heuristic for confidence and track frame history.
                print("We don't recognize you!")
                capture_faces()

            # draw all bounding boxes with text
            for i in range(len(rects)):
                x, y, w, h = face_utils.rect_to_bb(rects[i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(pred[i]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(0)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    samples = capture_faces(seconds=10, sampling_duration=0.25, debug=True)
