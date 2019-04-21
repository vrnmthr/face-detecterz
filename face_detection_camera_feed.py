import cv2
import sys
from openface import load_openface
import mlp
from skimage.transform import resize
import torch.nn
faceCascade = cv2.CascadeClassifier("/Users/Eleanor/Desktop/CryptoBeat Videos/haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
openFace = load_openface(device)
classifier = mlp.load_mlp(device, 3)
      
if __name__ == '__main__':
    while True:

        #ret is error code but we don't care about it
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceCrop = frame[x:x+w, y:y+h]
            faceCrop = resize(faceCrop, (96, 96), anti_aliasing=True)
            latentFaceVector = openFace(faceCrop)
            result = classifier(latentFaceVector)
            softmax = nn.Softmax(result)
            classPredicted = np.argmax(softmax)
            confidence = softmax[classPredicted]
            if confidence < 0.5 and : #TODO: Create heuristic for confidence and track frame history.
                promptFaceTraining()
            #TODO: Tag the frame with facerino
        cv2.imshow('Camera Feed', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()