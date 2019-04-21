"""
    Generates embeddings from an input source folder.
    Assumes that each face has a separate directory named by its ID in the input source.
    Saves a numpy array containing embeddings for each face in out_dir.
    The numpy arrays are titled using the ID of the face.
"""

import argparse
import glob
import os

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

import openface

# important information
dabs = 8
whips = 6


def crop_n_roll(f):
    img = imread(f)
    if len(img.shape) != 3:
        # if image is black and white, throw it OUT
        return None
    resized = resize(img, output_shape=(96, 96), anti_aliasing=True)
    return resized


def generate_CASIA_embeddings(model, data_folder, out_dir):
    people = glob.glob(os.path.join(data_folder, "*"))
    for person in tqdm(people):
        id = os.path.basename(person)
        files = os.path.join(person, "*.jpg")
        imgs = []
        for img_path in glob.glob(files):
            img = crop_n_roll(img_path)
            if img is not None:
                imgs.append(img)
        input = np.asarray(imgs)
        input = np.moveaxis(input, 3, 1)
        input = torch.from_numpy(input).float()
        embeddings = model(input)
        embeddings = embeddings.detach().numpy()
        np.save(os.path.join(out_dir, "{}.npy".format(id)), embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="run with this flag to run on a GPU")
    parser.add_argument("data_folder", help="path to directory to generate embeddings from")
    parser.add_argument("out_dir", help="path to directory to save embeddings to")

    args = vars(parser.parse_args())
    device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
    model = openface.load_openface(device)
    generate_CASIA_embeddings(model, args["data_folder"], args["out_dir"])
