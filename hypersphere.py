import numpy as np


def to_spherical(rectangular):
    """
    Converts rectangular coordinates in n-dimensions to spherical coordinates in n-dimensions
    Reference: https://en.wikipedia.org/wiki/N-sphere
    :param rectangular: numpy array of (x,n) where x is number of samples and n is # of dimensions
    :return numpy array of (x, n) in spherical coordinates
    """
    sq = np.square(rectangular)
    cumsum = np.sqrt(np.fliplr(np.cumsum(np.fliplr(sq), axis=1)))
    angles = np.arccos(rectangular[:, :-1] / cumsum[:, :-1])
    for i in range(len(rectangular)):
        if rectangular[i, -1] < 0:
            angles[i, -1] = 2 * np.pi - angles[i, -1]
    r = np.reshape(cumsum[:, 0], (len(rectangular), 1))
    spherical = np.concatenate([r, angles], axis=1)
    return spherical


if __name__ == '__main__':
    e = np.load("embeddings/test3/0000045.npy")
    to_spherical(e)
