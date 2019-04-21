import torch.nn as nn
import numpy as np
import torch
class Simple3MLP(nn.Module):
    def __init__(self, inputSize=128, mlpHiddenSize1=300, mlpHiddenSize2=128, numClasses=100): #100 is a placeholder for  now
        super(Simple3MLP, self).__init__()
        self.mlp1 = nn.Linear(inputSize, mlpHiddenSize1)
        self.mlp2 = nn.Linear(mlpHiddenSize1, mlpHiddenSize2)
        self.mlp3 = nn.Linear(mlpHiddenSize2, numClasses)

    def forward(self, faceImg):
        out = self.mlp1(faceImg)
        out = torch.relu(out)
        out = self.mlp2(out)
        out = torch.relu(out)
        out = self.mlp3(out)
        return out
class Simple2MLP(nn.Module):
    def __init__(self, inputSize=128, mlpHiddenSize=300, numClasses=100):
        super(Simple2MLP, self).__init__()
        self.mlp1 = nn.Linear(inputSize, mlpHiddenSize) 
        self.mlp2 = nn.Linear(mlpHiddenSize, numClasses)

    def forward(self, faceImg):
        out = self.mlp1(faceImg)
        out = torch.relu(out)
        out = self.mlp2(out)
        return out

def load_mlp(device, numLayers):
    model  = None
    if numLayers == 2:
        model = Simple2MLP(device)
    else:
        model = Simple3MLP(device)
    path = "weights/mlp_weights.pt"
    if device == torch.device("cpu"):
        model.load_state_dict(
            torch.load(path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(path))
    # no need to backprop over openface
    return model