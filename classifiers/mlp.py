import torch.nn as nn
import numpy as np
import torch
class Simple3MLP(nn.Module):
    def __init__(self, inputSize=128, mlpHiddenSize1=300, mlpHiddenSize2=200, numClasses=100, batch_size=10): #100 is a placeholder for  now
        super(Simple3MLP, self).__init__()
        self.mlp1 = nn.Linear(inputSize, mlpHiddenSize1)
        self.mlp2 = nn.Linear(mlpHiddenSize1, mlpHiddenSize2)
        self.mlp3 = nn.Linear(mlpHiddenSize2, numClasses)
        self.batch_size = batch_size
        self.params = list(self.mlp1.parameters()) + list(self.mlp2.parameters()) + list(self.mlp3.parameters())
    def forward(self, faceImg):
        out1 = self.mlp1(faceImg)
        out2 = torch.relu(out1)
        out3 = self.mlp2(out2)
        out4 = torch.relu(out3)
        return self.mlp3(out4)
    def fit(self, data, labels):
        num_batches = len(labels) // self.batch_size
        optimizer = torch.optim.Adam(self.params, lr=0.001)
        lossFunc = nn.CrossEntropyLoss()
        print("=======")
        for i in range(2):
            run_loss = 0
            acc = 0
            for i in range(num_batches):
                data_batch = data[i*self.batch_size: (i+1)*self.batch_size]
                labels_batch = labels[i*self.batch_size: (i+1)*self.batch_size]
                logits = self.forward(torch.tensor(data_batch))
                loss = lossFunc(logits, torch.tensor(labels_batch))
                run_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc += np.float32(np.sum(np.argmax(logits.detach().numpy()) == labels_batch)) / len(labels_batch)
            print(run_loss / num_batches)
            print(acc / num_batches)
    def predict(self, data):
        print(np.argmax(self.forward(torch.tensor(data)).detach().numpy()), axis=1)
        return np.argmax(self.forward(torch.tensor(data)).detach().numpy())

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
def train(model, numIters):
    loss = nn.CrossEntropyLoss
    for i in range(len(numIters)):
        prediction = model(data)
