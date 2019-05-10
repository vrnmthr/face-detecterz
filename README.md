# face-detecterz
the best cs1430 project ever


To run the project, we recommend creating a virtual environment and installing all libraries in the requirements.txt file. You can do this easily with pip by running `pip install -r requirements.txt`. Next, run `python3 repl.py` to be recognized! Note that the system will not prompt you to enter your name for retraining unless there is only one face in the video stream and it has been unknown for 30 frames, so retraining will not occur if two or more people are in the scene at the same time. When the system starts to capture new images for retraining, the video feed will appear to temporarily freeze, but it is recording you live. So long as the "sample taken" statements are being printed out, samples are being taken to add you to the dataset. If you tilt your head back and forth slightly, this will help with variety of images for retraining.

The emeddings from the CASIA face dataset are located in embeddings/known, and the faces added to the database during the process are stored in data/embeddings.
