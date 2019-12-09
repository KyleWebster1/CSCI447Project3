# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin
import random

import pre_processing
import dataset
import random
import numpy as np
from numpy import exp, array, random, dot


class NeuralNetwork:
    def __init__(self, numInput, numOutput, numHiddenLayers):
        self.numInput = numInput
        self.numOutput = numOutput
        self.numHiddenLayers = numHiddenLayers
        self.bias = [random.random() for _ in range(self.numHiddenLayers+1)]
        #list of weights as size(next) by size(source)
        self.weights = []
        if numHiddenLayers == 0:
            self.weights.append(np.random.randn(self.numInput, self.numOutput))
        else:
            #makes network of input + number of hidden layers
            for i in range(numHiddenLayers+1):
                #Shape is input layer to hidden layer is same
                self.weights.append(np.random.randn(self.numInput, self.numInput))
        #shape is input layer size to output layer size
        self.weights.append(np.random.randn(self.numInput, self.numOutput))

    def getWeights(self):
        return self.weights

    def updateWeights(self, newWeights):
        self.weights = newWeights

    #Set to true for sigmoid derivative
    def sigmoid(self, input, deriv= False):
        if (deriv == True):
            return input*(1-input)
        return 1 / (1 + np.exp(-input))

    def feedForward(self, input):
        merge = np.dot(input, self.weights[0]) + self.bias[0]
        sig = self.sigmoid(merge)
        for hLayer in range(1, self.numHiddenLayers+1):
            merge = np.dot(sig, self.weights[hLayer]) + self.bias[hLayer]
            sig = self.sigmoid(merge)
        merge = np.dot(sig, self.weights[-1]) + self.bias[-1]
        return self.sigmoid(merge)

    def backprop(self, input, trueValues, output):
        outputError = np.square(np.subtract(trueValues, output)).mean()
        delta = []
        delta.append = outputError * self.sigmoid(output, deriv=True)
        lastWError = delta.dot(self.weights[0].T)
        for hlayer in reversed(range(self.numHiddenLayers)):
            pass

        for hLayer in range(1, self.numHiddenLayers):
            delta = outputError * self.sigmoid(outputError, deriv=True)

    def MSELoss(self, true, pred):
        return np.mean(np.square(true - pred))

    def train(self, input, true):
        output = self.feedForward(input)
        self.backprop(input, true, output)





if __name__ == '__main__':
    nn = NeuralNetwork(2, 1, 2)
    for blank in range(10):
        print(nn.feedForward([blank, blank]))




# #Attempt 1
# def sigmoid(input):
#     return 1 / (1 + exp(-input))
#
#
# def derSigmoid(input):
#     s = sigmoid(input)
#     return s * (1 - s)
#
# class Neuron:
#     def __init__(self, bias):
#         self.bias = bias
#         self.weights = []
#
#     def calc_output(self, input):
#         self.input = input
#         self.output = sigmoid(sum([_input * weight for _input, weight in zip(input, self.weights)]))
#         return self.output
#
# class NeuronLayer:
#     def __init__(self, num):
#         self.num = num
#         self.bias = random.random()
#         self.neurons = [Neuron(self.bias) for _ in range(self.num)]
#
#     def forward(self, input):
#         return [neuron.calc_output(input) for neuron in self.neurons]
#
# class NeuralNetwork:
#     def __init__(self, numInput, numHidden, numOutput):
#         self.numInput = numInput
#         self.hiddenLayer = NeuronLayer(numHidden)
#         self.outputLayer = NeuronLayer(numOutput)
#         self.numHidden = numHidden
#         self.initWeight()
#
#     def initWeight(self):
#         for neuron in self.hiddenLayer.neurons:
#             neuron.weights = [random.random() for _ in range(self.numInput)]
#
#         for neuron in self.outputLayer.neurons:
#             neuron.weights = [random.random() for _ in range(self.numHidden)]
#
#     def feedForward(self, input):
#         hiddenLayerOutput = self.hiddenLayer.forward(input)
#         return self.outputLayer.forward(hiddenLayerOutput)
#
# if __name__ == '__main__':
#     nn = NeuralNetwork(2, 2, 2)
#     for blank in range(10):
#         print(nn.feedForward([.5, .1]))

"""
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
    	for i in reversed(range(len(self.layers))):
    		layer = self.layers[i]
    		errors = list()
    		if i != len(network)-1:
    			for j in range(len(layer)):
    				error = 0.0
    				for neuron in network[i + 1]:
    					error += (neuron['weights'][j] * neuron['delta'])
    				errors.append(error)
    		else:
    			for j in range(len(layer)):
    				neuron = layer[j]
    				errors.append(expected[j] - neuron['output'])
    		for j in range(len(layer)):
    			neuron = layer[j]
    			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(network, row, l_rate):
    	for i in range(len(network)):
    		inputs = row[:-1]
    		if i != 0:
    			inputs = [neuron['output'] for neuron in network[i - 1]]
    		for neuron in network[i]:
    			for j in range(len(inputs)):
    				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
    			neuron['weights'][-1] += l_rate * neuron['delta']


    def MSE(self, pred, actual):
        return np.mean(np.square(actual - pred))

    def accuracy(self,true, pred):
        truePositive = 0
        for i in range(len(true)):
            if pred[i]==true[i]:
                truePositive+=1
        return truePositive/len(true)
        
    def backprop(self, x, y):
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
        self.backprop(x, y)
        m = x.shape[1]
        for i in range(self.numberLayers + 1):
            self.weight[i + 1] -= eda * (self.dW[i + 1] / m)
            temp = eda * (self.dB[i + 1] / m)
            self.bias[i + 1] -= temp[i]
            # print(self.weight)
            # print()
            # print(self.bias)
            # print()
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
