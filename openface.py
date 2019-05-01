import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import Conv2d, BatchNorm, CrossMapLRN, Inception, Linear


class OpenFace(nn.Module):
    def __init__(self, device):
        super(OpenFace, self).__init__()

        self.device = device

        self.layer1 = Conv2d(3, 64, (7, 7), (2, 2), (3, 3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer5 = CrossMapLRN(device, 5, 0.0001, 0.75)
        self.layer6 = Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3, 3), (1, 1), (1, 1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(device, 5, 0.0001, 0.75)
        self.layer13 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer14 = Inception(192, (3, 5), (1, 1), (128, 32), (96, 16, 32, 64),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer15 = Inception(256, (3, 5), (1, 1), (128, 64), (96, 32, 64, 64),
                                 nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer16 = Inception(320, (3, 5), (2, 2), (256, 64), (128, 32, None, None),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer17 = Inception(640, (3, 5), (1, 1), (192, 64), (96, 32, 128, 256),
                                 nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer18 = Inception(640, (3, 5), (2, 2), (256, 128), (160, 64, None, None),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96, 96, 256), nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96, 96, 256),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer22 = nn.AvgPool2d((3, 3), stride=(1, 1), padding=(0, 0))
        self.layer25 = Linear(736, 128)

        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

        self.to(device)

    def forward(self, input):
        """
        :param input: torch tensor with dimensions (batch_size, 3, 96, 96)
        :return: embeddings with dimensions (batch_size, 128)
        """
        x = input
        x = x.to(self.device)

        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x ** 2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))

        return x


def load_openface(device):
    model = OpenFace(device)
    path = "data/openface_20180119.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    # no need to backprop over openface
    model.eval()
    return model


def preprocess_single(img):
    """
    Preprocessing method for a single face for transformation.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).float()
    return tensor


def preprocess_batch(imgs):
    """
    Preprocess a batch of images for input into openface
    :param imgs:
    :return:
    """
    imgs = np.flip(imgs, axis=3)
    imgs = np.moveaxis(imgs, 3, 1)
    imgs = imgs.astype(np.float32) / 255.0
    tensor = torch.from_numpy(imgs).float()
    return tensor


if __name__ == '__main__':
    device = torch.device("cpu")
    openface = load_openface(device)
    print(openface)

    start = time.time()
    I = np.reshape(np.array(range(96 * 96), dtype=np.float32) * 0.01, (1, 96, 96))
    I = np.concatenate([I, I, I], axis=0)
    I_ = torch.from_numpy(I).unsqueeze(0)
    I_ = Variable(I_.to(device), requires_grad=False)
    end = time.time()
    print("Ran one sample in {}s".format(end - start))

    result = openface(I_)
    print(result)
