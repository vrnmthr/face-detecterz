"""
    Generates embeddings from an input source folder.
    Assumes that each face has a separate directory named by its ID in the input source.
    Saves a numpy array containing embeddings for each face in out_dir.
    The numpy arrays are titled using the ID of the face.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

import openface
from align_faces import align_and_extract_faces

# important information
dabs = 8
whips = 6


def crop_faces(f):
    # loads a face as BGR
    img = cv2.imread(f)
    if len(img.shape) != 3:
        # if image is black and white, throw it OUT
        return None
    aligned = align_and_extract_faces(img)
    if len(aligned) != 1:
        # if multiple faces in images for embedding, throw them out
        return None
    img = aligned[0]
    return img


def generate_embeddings(model, data_folder, n=None, max_per_class=50):
    generated = 0
    start = time.time()

    people = glob.glob(os.path.join(data_folder, "*"))
    n = len(people) if n is None else n
    people = people[:n]

    for person in tqdm(people):
        try:
            id = os.path.basename(person)
            if not os.path.exists(os.path.join("embeddings/known", "{}.npy".format(id))):
                files = os.path.join(person, "*.jpg")
                imgs = []
                img_paths = glob.glob(files)[:max_per_class]
                for img_path in img_paths:
                    img = crop_faces(img_path)
                    if img is not None:
                        imgs.append(img)
                input = np.asarray(imgs)
                generated += len(input)
                input = openface.preprocess_batch(input)
                embeddings = model(input)
                embeddings = embeddings.detach().numpy()
                np.save(os.path.join("embeddings/known", "{}.npy".format(id)), embeddings)
        except:
            pass

    end = time.time()
    print("Generated {} embeddings with an average time of {}s".format(generated, (end - start) / generated))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="run with this flag to run on a GPU")
    parser.add_argument("data_folder", help="path to directory to generate embeddings from")

    args = vars(parser.parse_args())
    device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
    model = openface.load_openface(device)
    generate_embeddings(model, args["data_folder"])
