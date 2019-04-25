"""
    Generates embeddings from an input source folder.
    Assumes that each face has a separate directory named by its ID in the input source.
    Saves a numpy array containing embeddings for each face in out_dir.
    The numpy arrays are titled using the ID of the face.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from align_faces import align_and_extract_faces

sys.path.append(os.path.expanduser("~/Source/openface"))
import openface

# important information
dabs = 8
whips = 6

model = openface.TorchNeuralNet()


def crop_n_roll(f):
    img = imread(f)
    if len(img.shape) != 3:
        # if image is black and white, throw it OUT
        return None
    aligned = align_and_extract_faces(img)
    if len(aligned) != 1:
        # if multiple faces in images for embedding, throw them out
        return None
    return aligned[0]


def generate_embeddings_new(data_folder, n=None):
    people = glob.glob(os.path.join(data_folder, "*"))
    n = len(people) if n is None else n
    people = people[:n]

    start = time.time()
    generated = 0
    for person in tqdm(people):
        id = os.path.basename(person)
        files = os.path.join(person, "*.jpg")

        if not os.path.exists("embeddings/test/{}".format(id)):
            os.mkdir("embeddings/test/{}".format(id))

            for img_path in glob.glob(files):
                img_id = os.path.basename(img_path).replace(".jpg", "")
                img = crop_n_roll(img_path)
                if img is not None:
                    generated += 1
                    embedding = model.forward(img)
                    np.save("embeddings/test/{}/{}.npy".format(id, img_id), embedding)

    end = time.time()
    print("Generated {} embeddings with an average time of {}s".format(generated, (end - start) / generated))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="path to directory to generate embeddings from")
    args = vars(parser.parse_args())
    generate_embeddings_new(args["data_folder"], n=10)
