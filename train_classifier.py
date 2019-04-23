import argparse
import glob
import os

import numpy as np
import torch.utils.data as td
from tqdm import tqdm


class FaceDataset(td.Dataset):

    def __init__(self, dir, n=None):
        """
        Creates new dataset
        :param dir: path to directory of saved embeddings
        :param n: number of faces to load from the dataset
        """
        self.paths = glob.glob(os.path.join(dir, "*.npy"))
        n = len(self.paths) if n is None else n
        self.embeddings = []
        self.labels = []
        for ix, path in tqdm(enumerate(self.paths[:n]), total=n):
            e = np.load(path)
            self.embeddings.append(e)
            self.labels.append(np.full(len(e), ix))
        self.embeddings = np.concatenate(self.embeddings)
        self.labels = np.concatenate(self.labels)
        assert len(self.embeddings) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample = self.embeddings[idx]
        label = self.labels[idx]
        return sample, label


def get_dev_test_train(dataset):
    """
    :param dataset: a Torch dataset
    :return: dev, test, train dataloaders
    """
    n = len(dataset)
    dev_ratio, test_ratio, train_ratio = 0.8, 0.1, 0.1
    dev_length, test_length, train_length = int(n * dev_ratio), int(n * test_ratio), int(n * train_ratio)
    # make sure the ratios sum properly
    train_length += n - dev_length - test_length - train_length
    dev, test, train = td.random_split(dataset, [dev_length, test_length, train_length])
    return dev, test, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", help="path to directory with embeddings")
    args = vars(parser.parse_args())
