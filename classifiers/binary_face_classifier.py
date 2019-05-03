import argparse
import copy
import glob
import os

import numpy as np
import plotly as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

"""
A first pass for a feed-forward network
"""

# Model params
LOSS_FUNC = nn.CrossEntropyLoss()
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10


class BinaryFaceClassifier:

    def __init__(self, network, threshold):
        self.centroids = []
        self.network = network
        self.threshold = threshold

    def fit(self, data, labels):
        centroids = []
        # these should be sorted
        for i in np.unique(labels):
            samples = data[labels == i]
            centroid = np.mean(samples, axis=0)
            centroids.append(centroid)
        self.centroids = np.asarray(centroids)
        return self

    def predict_proba(self, data):
        """
        Returns a probability prediction for each face
        :param data:
        :return:
        """
        probs = []
        for sample in data:
            firsts = np.asarray([sample for _ in self.centroids])
            seconds = copy.deepcopy(self.centroids)  # TODO: check if this is necessary
            pred = self.network.evaluate(firsts, seconds)
            prob = pred[:, 1]
            probs.append(prob)
        return np.asarray(probs)


class BinaryFaceNetwork(nn.Module):

    def __init__(self, device):
        """
        3-layer classifier that takes in concatenated vectors as input and outputs
        1 if they belong to the same class and 0 otherwise
        """
        super(BinaryFaceNetwork, self).__init__()
        self.device = device

        # Define the layers
        self.first = nn.Linear(256, 64)
        self.second = nn.Linear(64, 16)
        self.third = nn.Linear(16, 2)
        self.to(self.device)

    def forward(self, first, second):
        """
        Implements a forward pass through the network
        :param inputs: a (batch, 128) tensor
        :return: size (batch, 2) tensor representing logits
        """
        out = torch.cat((first, second), dim=1)
        out = F.relu(self.first(out))
        out = F.relu(self.second(out))
        out = self.third(out)
        return out

    @torch.no_grad()
    def evaluate(self, firsts, seconds):
        """
        Runs a single batch through the network and returns the results
        :param firsts: (n, 128) numpy array
        :param seconds: (n, 128) numpy array
        :return: (n, 2) logits
        """
        firsts = torch.from_numpy(firsts)
        seconds = torch.from_numpy(seconds)
        results = self.forward(firsts, seconds)
        results = F.softmax(results)
        results = results.detach().numpy()
        return results

    @torch.enable_grad()
    def train_with(self, train, dev, num_epochs, learning_rate, save: str):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        loaders = {"train": train, "dev": dev}
        loss_history = {mode: [[] for _ in range(num_epochs)] for mode in loaders}

        for epoch in range(num_epochs):
            for mode in loaders:
                loader = loaders[mode]
                # run training step across entire batch
                for batch in tqdm(loader, desc='{} epoch {}/{}'.format(mode, epoch + 1, num_epochs)):

                    firsts = torch.Tensor(batch['first']).to(self.device)
                    seconds = torch.Tensor(batch['second']).to(self.device)
                    labels = torch.LongTensor(batch['label']).to(self.device)

                    if mode == 'train':
                        self.train()  # tell pytorch to set the model to train mode
                        self.zero_grad()
                        out = self.forward(firsts, seconds)
                        loss = LOSS_FUNC(out, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        self.eval()
                        with torch.no_grad():
                            out = self.forward(firsts, seconds)
                            loss = LOSS_FUNC(out, labels)

                    loss_history[mode][epoch].append(loss.item())

                loss_this_epoch = sum(loss_history[mode][epoch]) / len(loader)
                print("Average {} loss this epoch: {}".format(mode, loss_this_epoch))

            ckpt = os.path.join(save, "model-epoch{}.ckpt".format(epoch))
            torch.save(self.state_dict(), ckpt)
            print("Checkpoint saved to {}".format(ckpt))

        return loss_history

    def test_with(self, data: Dataset) -> np.ndarray:
        """
        Tests the model
        :param data: dataset to test on
        :param device: device to test on
        :return: confusion matrix from testing
        """
        confusion = np.zeros((2, 2))
        for i in tqdm(range(len(data))):
            sample = data[i]
            tag, first, second = sample['label'], sample['first'], sample['second']
            first = np.expand_dims(first, 0)
            second = np.expand_dims(second, 0)
            result = self.evaluate(first, second)
            result = np.argmax(result, axis=1)[0]
            confusion[result, tag] += 1
        confusion /= len(data)
        return confusion


def load_facepair_dataset(path, size=10000, ratio=0.3):
    # load all faces
    faces = []
    paths = glob.glob(os.path.join(path, "*.npy"))
    np.random.shuffle(paths)
    for ix, path in enumerate(paths):
        e = np.load(path)
        faces.append(e)

    firsts, seconds, labels = [], [], []

    # generate all embeddings that are from the same people
    n = int(size * ratio)
    people = np.random.randint(0, high=len(faces), size=n)
    for p in people:
        samples = faces[p]
        ixs = np.random.randint(0, high=len(samples), size=2)
        first, second = samples[ixs]
        firsts.append(first)
        seconds.append(second)
        labels.append(1)

    # generate all embeddings that are from different people
    n = int(size * (1 - ratio))
    for _ in range(n):
        people = np.random.choice(len(faces), size=2, replace=False)
        samples = faces[people]
        first = samples[0][np.random.randint(0, high=len(samples[0]))]
        second = samples[1][np.random.randint(0, high=len(samples[1]))]
        firsts.append(first)
        seconds.append(second)
        labels.append(0)

    firsts, seconds, labels = np.asarray(firsts), np.asarray(seconds), np.asarray(labels)
    return firsts, seconds, labels


class PairDataset(Dataset):
    """
    Loads a generic labelled dataset
    """

    def __init__(self, firsts, seconds, labels):
        """
        :param csv: path to processed csv file
        :param transform: optional transform to be applied on a sample
        """
        self.firsts = firsts
        self.seconds = seconds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample = {
            'label': self.labels[idx],
            'first': self.firsts[idx],
            'second': self.seconds[idx]
        }
        return sample


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="path to load embeddings")
    parser.add_argument("--restore", help="path to restore model for testing", default="")
    parser.add_argument("--device", help="pass --device cuda to run on gpu (default 'cpu')", default="cpu")
    parser.add_argument("--save", help="path to save directory (default 'out')", default="out")

    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    device = torch.device('cpu')
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    print("Using device {}".format(device))

    model = BinaryFaceNetwork(device)
    print(model)

    print("Building datasets")
    firsts, seconds, labels = load_facepair_dataset(os.path.join(args.dir, "train"), size=10000000)
    train_set = PairDataset(firsts, seconds, labels)

    firsts, seconds, labels = load_facepair_dataset(os.path.join(args.dir, "dev"), size=1000000)
    dev_set = PairDataset(firsts, seconds, labels)

    firsts, seconds, labels = load_facepair_dataset(os.path.join(args.dir, "test"), size=1000000)
    test_set = PairDataset(firsts, seconds, labels)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)

    if args.restore == "":
        print("Training...")
        loss_history = model.train_with(train_loader, dev_loader, NUM_EPOCHS, LEARNING_RATE, args.save)

        print("Graphing losses...")
        train_agg = [sum(x) / len(train_loader) for x in loss_history['train']]
        train_agg = plt.graph_objs.Scatter(x=np.arange(NUM_EPOCHS), y=train_agg, name="train_agg")
        dev_agg = [sum(x) / len(dev_loader) for x in loss_history['dev']]
        dev_agg = plt.graph_objs.Scatter(x=np.arange(NUM_EPOCHS), y=dev_agg, name="dev_agg")
        plt.offline.plot([train_agg, dev_agg], filename="train_loss.html")
    else:
        model.load_state_dict(torch.load(args.restore))
        print("Model restored from disk.")

    print("Testing...")
    confusion = model.test_with(test_set)
    print("Accuracy: {}%".format(100 * np.sum(np.diag(confusion))))
    print(confusion)
