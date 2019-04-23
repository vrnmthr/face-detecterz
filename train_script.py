import torch
import train_classifier
from classifiers.mlp import Simple3MLP
import numpy as np
from torch.autograd import Variable

numIters = 10000
lr = 0.001

model = Simple3MLP()
CEloss = torch.nn.CrossEntropyLoss()
faceDataset = train_classifier.FaceDataset("/home/bmo/work/cs143/aligned_embeddings", n=100)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(numIters):
	s, l = faceDataset[i]
	s = Variable(torch.from_numpy(s))
	s = s.unsqueeze(0)
	l = Variable(torch.LongTensor([l]))
	predicted = model(s)
	loss = CEloss(predicted, l)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	print(loss.item())




