import argparse
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
NUM_EPOCHS = 3


class BinaryFaceClassifier(nn.Module):

    def __init__(self, device):
        """
        3-layer classifier that takes in concatenated vectors as input and outputs
        1 if they belong to the same class and 0 otherwise
        """
        super(BinaryFaceClassifier, self).__init__()
        self.device = device

        # Define the layers
        self.first = nn.Linear(256, 128)
        self.second = nn.Linear(128, 64)
        self.third = nn.Linear(64, 2)
        self.to(self.device)

    def forward(self, inputs):
        """
        Implements a forward pass through the network
        :param inputs: a batch-size x features tensor
        :return: a batch-size x categories tensor representing probability of being in each category
        """
        out = F.relu(self.first(inputs))
        out = F.relu(self.second(out))
        out = self.third(out)
        return out

    @torch.no_grad()
    def evaluate(self, seq: torch.Tensor):
        """
        Returns a class code for a given sequence
        :param seq: (features) tensor
        :return: int representing activity class
        """
        out = self.forward(seq)
        return torch.argmax(out).item()

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
                    inputs = torch.Tensor(batch['sequence']).to(self.device)
                    labels = torch.LongTensor(batch['label']).to(self.device)

                    if mode == 'train':
                        self.train()  # tell pytorch to set the model to train mode
                        self.zero_grad()
                        out = self.forward(inputs)
                        loss = LOSS_FUNC(out, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        self.eval()
                        with torch.no_grad():
                            out = self.forward(inputs)
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
            tag = sample['label']
            input = torch.from_numpy(sample['sequence'])
            result = self.evaluate(input)
            confusion[result, tag] += 1
        confusion /= len(data)
        return confusion


def load_facepair_dataset(path, size=10000, ratio=0.5):
    # load all faces
    faces = []
    paths = glob.glob(os.path.join(path, "*.npy"))
    np.random.shuffle(paths)
    for ix, path in enumerate(paths):
        e = np.load(path)
        faces.append(e)
    faces = np.asarray(faces)

    data = []
    labels = []

    # generate all embeddings that are from the same people
    n = int(size * ratio)
    people = np.random.randint(0, high=len(faces), size=n)
    for p in people:
        samples = faces[p]
        ixs = np.random.randint(0, high=len(samples), size=2)
        data.append(np.concatenate(samples[ixs]))
        labels.append(1)

    # generate all embeddings that are from different people
    n = int(size * (1 - ratio))
    for _ in range(n):
        people = np.random.choice(len(faces), size=2, replace=False)
        samples = faces[people]
        first = samples[0][np.random.randint(0, high=len(samples[0]))]
        second = samples[1][np.random.randint(0, high=len(samples[1]))]
        data.append(np.concatenate([first, second]))
        labels.append(0)

    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


class LabelledDataset(Dataset):
    """
    Loads a generic labelled dataset
    """

    def __init__(self, data, labels):
        """
        :param csv: path to processed csv file
        :param transform: optional transform to be applied on a sample
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = {'label': self.labels[idx], 'sequence': self.data[idx]}
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

    model = BinaryFaceClassifier(device)
    print(model)

    # randomly slow down the data by a little bit
    data, labels = load_facepair_dataset(args.dir, size=1000000)
    n = len(data)
    train_set = LabelledDataset(data[:int(0.8 * n)], labels[:int(0.8 * n)])
    dev_set = LabelledDataset(data[int(0.8 * n):int(0.9 * n)], labels[int(0.8 * n):int(0.9 * n)])
    test_set = LabelledDataset(data[int(0.9 * n):], labels[int(0.9 * n):])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

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
    confusion = model.test_with(train_set)
    print("Accuracy: {}%".format(100 * np.sum(np.diag(confusion))))
    print(confusion)
