import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

def next_batch(inputs, targets, batchSize):
    for i in range(0, inputs.shape[0], batchSize):
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}....".format(DEVICE))

print("[INFO] preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, 
                    cluster_std=2.5, random_state=95)

(trainX, testX, trainY, testY) = train_test_split(X, y, 
                                                  test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

mlp = mlp.get_training_model().to(DEVICE)
print(mlp)

opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

for epoch in range(0, EPOCHS):
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp.train()

    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        predictions = mlp(batchX)
        loss = lossFunc(predictions, batchY.long())

        opt.zero_grad()
        loss.backward()
        opt.step()

        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)
    
    trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples),
                              (trainAcc / samples)))
    
testLoss = 0
testAcc = 0
samples = 0
mlp.eval()

with torch.no_grad():

	for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):

		(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

		predictions = mlp(batchX)
		loss = lossFunc(predictions, batchY.long())

		testLoss += loss.item() * batchY.size(0)
		testAcc += (predictions.max(1)[1] == batchY).sum().item()
		samples += batchY.size(0)

	testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
	print(testTemplate.format(epoch + 1, (testLoss / samples),
		(testAcc / samples)))
	print("")