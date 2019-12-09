# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin
import random
import pre_processing
import dataset
import numpy


class Layer:

    def __init__(self,weightMatrix, isLastReg):
        self.weightMatrix = weightMatrix
        self.isLastReg = isLastReg
        self.nodeInput = []

    #takes input to layer, returns sigmoid of next layer
    def feedForward(self, nodeInput):

        self.nodeInput = nodeInput
        v = self.getLayerOutput()
        if self.isLastReg:
            v = [v]

        v = self.sigmoid(v)

        return v

    #returns output array for given node
    def getNodeOutput(self, k):
        return numpy.multiply(self.nodeInput[k], self.weightMatrix[k])

    #returns the output array for layer
    def getLayerOutput(self):
        return numpy.matmul(self.nodeInput, self.weightMatrix)

    #returns the change in weights of the layer using it's delta
    def findWeightChange(self, delta):
        weight_change = []
        for k in range(len(self.nodeInput)):
            weight_change.append([])
            for j in range(len(delta[k])):
                weight_change[k].append(self.nodeInput[k] * delta[k][j])

        return weight_change

    #finds the delta for an output layer
    def findOutputDelta(self,target):
        delta = []
        for k in range(len(self.nodeInput)):
            o = self.getNodeOutput(k)
            print(k)
            print(o)
            print(target)
            delta.append([])
            for j in range(len(o)):
                delta[k].append((target[j] - o[j]) * o[j] * (1-o[j]))
        

        return delta

    #finds the delta for a hidden layer
    def findHiddenDelta(self, kdelta):
        o = self.getLayerOutput()
        delta = []
        for k in range(len(self.nodeInput)):
            delta.append([])
            for j in range(len(kdelta)):
                subsum = 0
                for i in range(len(kdelta[j])):
                    subsum += kdelta[j][i]
                print(self.weightMatrix[k][j])
                delta[k].append(subsum * self.weightMatrix[k][j] * (o[j]) * (1-o[j]))

        return delta

    #finds the sigmoid of each element of the array
    def sigmoid(self, input):
        for i in range(len(input)):
            input[i] = 1 / (1 + numpy.exp(-input[i]))
        return input

    #finds the sigmoid derivative of each element of the array
    def sigDeriv(self, output):
        for i in range(len(output)):
            output[i] = output[i] * (1 - output[i])
        return output



class FeedForwardNeuralNetwork:
    
    def __init__(self, inputs, outputs, hiddenLayers):
        self.input = inputs
        self.outputNumber = outputs
        self.numberLayers = hiddenLayers + 1
        self.layers = []

        self.initWeights()

    #initializes weights
    def initWeights(self):
        self.layers = [0] * self.numberLayers
        for i in range(self.numberLayers - 1):
            self.layers[i] = Layer(numpy.random.rand(self.input, self.input), False)
        self.layers[-1] = Layer(numpy.random.rand(self.input, self.outputNumber), self.outputNumber == 1)

    def setWeights(self, weightMatrix):
        i = 0
        for matrix in weightMatrix:
            self.layers[i].weightMatrix = matrix
            i += 1
            
    #runs train/test for a set of train/test sets
    def run(self, trainData, learningRate):
        for i in range(10):
            self.train(trainData.getTrainingSet(i), learningRate)
            performance = self.test(trainData.getTestSet(i))
            if self.outputNumber == 1:
                print("K: " + str(i) + "MSE: " + str(performance))
            else:
                print("K: " + str(i) + "ACC: " + str(performance))
            
    #trains the net
    def train(self, training_set, learningRate):
        change = 10000
        while change > .0001:
            x = random.choice(training_set)
            #print(x)
            y = x[:-1]
            pred = self.makePrediction(y)
            if (self.outputNumber == 1):
                d = [x[-1]]
            else:
                d = pred.copy()
                for i in range(len(d)):
                    if i == x[-1]:
                        d[i] = 1
                    else:
                        d[i] = 0
            print(d)
            change = self.backProp(d, learningRate)

    #tests the net
    def test(self, test_set):
        mse = 0
        acc = 0
        for x in test_set:
            y = x[:-1]
            pred = self.makePrediction(y)
            if self.outputNumber == 1:
                mse += pow(x[-1] - pred, 2)
            else:
                acc += self.getAcc(pred, x[-1])

        if self.outputNumber == 1:
            mse /= len(test_set)
            return mse
        else:
            acc /= len(test_set)
            return acc

    #makes prediction for given input
    def makePrediction(self, x):
        
        for layer in self.layers:
            x = layer.feedForward(x)

        if self.outputNumber == 1:
            return x[0]
        return x

    #updates weights using all of the layers' deltas
    def update_weights(self, layer, delta, learningRate):
        weight_change = layer.findWeightChange(delta)
        weight_change = numpy.multiply(-learningRate, weight_change)
        layer.weightMatrix = numpy.add(layer.weightMatrix, weight_change)

    #backpropagates the error of the network
    def backProp(self, expected, learningRate):

        totalDelta = []
        delta = self.layers[-1].findOutputDelta(expected)
        totalDelta.append(delta)
        i = 0
        for layer in reversed(self.layers[:-1]):
            i += 1
            delta = layer.findHiddenDelta(delta)
            totalDelta.append(delta)
            
        i = 0
        for layer in reversed(self.layers):
            self.update_weights(layer, totalDelta[i], learningRate)
            i += 1

        change = 0
        for i in totalDelta[0]:
            for j in i:
                change += abs(j)

        return change

    #checks whether the output predicts the desired class
    def getAcc(self, o, d):
        high = 0
        index = 0
        for i in range(len(o)):
            if o[i] > high:
                high = o[i]
                index = i
        if i == d:
            return 1
        else:
            return 0

tData = pre_processing.pre_processing("data/machine.data")
trainData = dataset.dataset(tData.getData())
net = FeedForwardNeuralNetwork(9, 1, 1)
net.run(trainData, .001)

