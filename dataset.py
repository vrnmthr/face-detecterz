import argparse
import glob
import os

import numpy as np


class FaceDataset:

    def __init__(self, dir, unknown_dir, n=None, num_unknown=250):
        """
        Creates new dataset
        :param dir: path to directory of saved embeddings
        :param n: number of faces to load from the dataset
        """
        self.paths = glob.glob(os.path.join(dir, "*.npy"))
        n = len(self.paths) if n is None else n
        self.data = [[], [], []]
        self.labels = [[], [], []]
        self.by_class = {}
        self.paths = self.paths[:n]
        np.random.shuffle(self.paths)
        self.ix_to_name = {}
        unknown_idx = len(self.paths)
        for ix, path in enumerate(self.paths):
            e = np.load(path)
            l = len(e)
            name = os.path.basename(path).replace(".npy", "")
            self.by_class[name] = e
            self.ix_to_name[ix] = name
            # get random indices
            ixs = np.arange(l)
            np.random.shuffle(ixs)
            self.data[0].append(e[ixs[:int(0.8 * l)]])
            self.data[1].append(e[ixs[int(0.8 * l):int(0.9 * l)]])
            self.data[2].append(e[ixs[int(0.9 * l):]])

            self.labels[0].append(np.full(len(e[:int(0.8 * l)]), ix))
            self.labels[1].append(np.full(len(e[int(0.8 * l):int(0.9 * l)]), ix))
            self.labels[2].append(np.full(len(e[int(0.9 * l):]), ix))
        self.unknown_paths =  glob.glob(os.path.join(unknown_dir, "*.npy"))
        self.unknown_paths = self.unknown_paths[:num_unknown]
        np.random.shuffle(self.unknown_paths)
        for ix, path in enumerate(self.unknown_paths):
            e = np.load(path)
            l = len(e)
            name = os.path.basename(path).replace(".npy", "")
            self.by_class["unknown_class"] = e
            self.ix_to_name[unknown_idx] = "unknown_class"

            # get random indices
            ixs = np.arange(l)
            np.random.shuffle(ixs)
            self.data[0].append(e[ixs[:int(0.8 * l)]])
            self.data[1].append(e[ixs[int(0.8 * l):int(0.9 * l)]])
            self.data[2].append(e[ixs[int(0.9 * l):]])

            self.labels[0].append(np.full(len(e[:int(0.8 * l)]), unknown_idx))
            self.labels[1].append(np.full(len(e[int(0.8 * l):int(0.9 * l)]), unknown_idx))
            self.labels[2].append(np.full(len(e[int(0.9 * l):]), unknown_idx))

        # vvvv shitass code but it wasn't working otherwise idk??
        self.data = np.concatenate(self.data[0]), np.concatenate(self.data[1]), np.concatenate(self.data[2])
        self.labels = np.concatenate(self.labels[0]), np.concatenate(self.labels[1]), np.concatenate(self.labels[2])

    def __len__(self):
        return len(self.labels)

    def train(self):
        return self.data[0], self.labels[0]

    def test(self):
        return self.data[1], self.labels[1]

    def dev(self):
        return self.data[2], self.labels[2]

    def all(self):
        return np.concatenate(self.data), np.concatenate(self.labels), self.ix_to_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", help="path to directory with embeddings")
    args = vars(parser.parse_args())
