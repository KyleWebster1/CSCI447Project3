# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin
import random
import pre_processing
import dataset
import numpy as np

class FeedForwardNeuralNetwork:
    #Inspired from code from hackermoon.com blogpost by Niranjan Kumar
    def __init__(self, inputs, outputs, hiddenLayer = [3]):
        self.input = inputs
        self.outputNumber = outputs
        self.numberLayers = len(hiddenLayer)
        self.sizes = [inputs] + hiddenLayer + [outputs]
        self.bias = {}
        self.weight = {}
        for i in range(self.numberLayers + 1):
            self.weight[i + 1] = np.random.rand(self.sizes[i], self.sizes[i + 1])
            self.bias[i + 1] = np.zeros((1, self.sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    def regressionPred(self, x):
        pred = []
        #print(x)
        for point in x:
            p = self.sigmoid(self.feedForward(point))
            pred.append(np.average(p))
        #print(pred)
        #print(self.weight)
        return pred

    def classificationPred(self, x):
        pred = []
        for point in x:
            p = self.feedForward(point)
            pred.append(np.argmax(p))
        return pred
    def MSE(self, pred, actual):
        return np.mean(np.square(actual - pred))

    def accuracy(self,true, pred):
        truePositive = 0
        for i in range(len(true)):
            if pred[i]==true[i]:
                truePositive+=1
        return truePositive/len(true)

    def feedForward(self,x):
        self.A = {}
        self.H = {}
        self.H[0] = x
        for i in range(self.numberLayers):
            self.A[i + 1] = np.matmul(self.H[i], self.weight[i + 1]) + self.bias[i + 1]
            self.H[i + 1] = self.sigmoid(np.dot(self.H[i], self.weight[i + 1]) + self.bias[i + 1])
        self.A[self.numberLayers + 1] = np.matmul(self.H[self.numberLayers], self.weight[self.numberLayers + 1]) + self.bias[self.numberLayers + 1]
        self.H[self.numberLayers + 1] = self.sigmoid(self.A[self.numberLayers + 1])
        return self.H[self.numberLayers + 1]

    def backprop(self, x, y):
        self.feedForward(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.numberLayers + 1
        self.dA[L] = (self.H[L]- y[:,None])
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k - 1] = np.matmul(self.dA[k], self.weight[k].T)
            self.dA[k - 1] = np.multiply(self.dH[k - 1], self.sigmoid_deriv(self.H[k - 1]))

    def update(self, eda, x, y):
        ffn.backprop(x, y)
        m = x.shape[1]
        for i in range(self.numberLayers + 1):
            self.weight[i + 1] -= eda * (self.dW[i + 1] / m)
            temp = eda * (self.dB[i + 1] / m)
            self.bias[i + 1] -= temp[i]
            # print(self.weight)
            # print()
            # print(self.bias)
            # print()
"""
tData = pre_processing.pre_processing("data/machine.data")
trainData = dataset.dataset(tData.getData())
original=np.array(trainData.getTrainingSet(0))
test = np.array(trainData.getTestSet(0))
x = original[:,:-1]
y = original[:,-1]
xsize = x.shape #6 is input nodes Last column is correct_answer
ysize = np.unique(y).shape
#Simple test case. AND gate
#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [1]])
ffn = FeedForwardNeuralNetwork(xsize[1], ysize[0], [xsize[1]]*2)
#print(ffn.feedForward(x))
#print(ffn.weight)

for i in range (2000):
    ffn.update(.1, x, y)
#print(ffn.weight)


#print(y)
#clas = ffn.classificationPred(x)
#print(clas)
reg = ffn.regressionPred(x)
print(ffn.MSE(y, reg))
#print(ffn.accuracy(y, clas))

tData = pre_processing.pre_processing("data/segmentation.data")
trainData = dataset.dataset(tData.getData())
original=np.array(trainData.getTrainingSet(0))
test = np.array(trainData.getTestSet(0))
x = original[:,:-1]
y = original[:,-1]
xsize = x.shape #6 is input nodes Last column is correct_answer
ysize = np.unique(y).shape
#Simple test case. AND gate
#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [1]])
ffn = FeedForwardNeuralNetwork(xsize[1], ysize[0], [xsize[1]]*2)
#print(ffn.feedForward(x))
#print(ffn.weight)

for i in range (2000):
    ffn.update(.1, x, y)
#print(ffn.weight)


#print(y)
clas = ffn.classificationPred(x)
#reg = ffn.regressionPred(x)
#print(reg)
#print(ffn.MSE(y, reg))
print(ffn.accuracy(y, clas))



ffn.add(Layer(size[1], size[1]))
ffn.add(Layer(size[1], size[1]))
#ffn.add(Layer(4, 4))
#ffn.add(Layer(4, 4))
ffn.add(Layer(size[1], 6))

ffn.train(x,y,0.01,2000)
print('Accuracy: %.2f%%' % (ffn.accuracy(ffn.final_pass(x), y.flatten()) * 100))
#print(ffn.accuracy(final_x,y))
"""
