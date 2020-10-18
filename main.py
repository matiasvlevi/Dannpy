from Dann import *
from numba import *
class Data:
    def __init__(self,inputs,targets):
        self.inputs = inputs
        self.targets = targets


dataset = [];
dataset.append(Data([1,1],[0]))
dataset.append(Data([1,0],[1]))
dataset.append(Data([0,1],[1]))
dataset.append(Data([0,0],[0]))

nn = Dann(2,1)
nn.addHiddenLayer(4,leakyRelu)
nn.makeWeights();
nn.lr = 0.1
nn.log();

print(nn.feedForward([1,1]))
print(nn.feedForward([1,0]))
print(nn.feedForward([0,1]))
print(nn.feedForward([0,0]))
for e in range(1000):
    for i in range(len(dataset)):
        nn.backpropagate(dataset[i].inputs,dataset[i].targets)
        print(nn.loss)

print(nn.feedForward([1,1]))
print(nn.feedForward([1,0]))
print(nn.feedForward([0,1]))
print(nn.feedForward([0,0]))



#cuda.api.detect()
