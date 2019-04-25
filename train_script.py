import torch
import train_classifier
from classifiers.mlp import Simple3MLP
import numpy as np
from torch.autograd import Variable
import torch.utils.data as td

epochs = 10000
lr = 0.001
batch_size = 20

model = Simple3MLP()
CEloss = torch.nn.CrossEntropyLoss()
faceDataset = train_classifier.FaceDataset("/home/bmo/work/cs143/aligned_embeddings", n=100)#.to_dataloader(batch_size=bsz)
dev, test, train = train_classifier.get_dev_test_train(faceDataset)
train = td.DataLoader(train, batch_size=batch_size, shuffle=True)
dev = td.DataLoader(dev, batch_size=batch_size, shuffle=True)
test = td.DataLoader(test, batch_size=batch_size, shuffle=True)
print("====================")
print(len(train))
print(len(dev))
print(len(test))
print("===================")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(epochs):
	runLoss = 0
	for s, l in train:
		predicted = model(s)
		loss = CEloss(predicted, l)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		runLoss += loss.item()
	print(runLoss)
	runLoss = 0
	if i % 100 == 0:
		runLoss = 0
		for s, l in dev:
			predicted = model(s)
			loss = CEloss(predicted, l)
			runLoss += loss.item()
		print("VALIDATION LOSS: " + str(runLoss))
for s, l in test:
	predicted = model(s)
	loss = CEloss(predicted, l)
	runLoss += loss.item()
	print("TEST LOSS: " + str(runLoss))




