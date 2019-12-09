# Feed Forward Network Implementation
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

    #finds the delta for an output layer
    def findOutputChange(self, delta):

        change = numpy.multiply(delta, self.nodeInput)
        return change

    #finds the delta for a hidden layer
    def findHiddenChange(self, delta, downstream):

        change = numpy.multiply(delta, self.nodeInput)
        change = numpy.matmul(change, downstream.weightMatrix)

        return change

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
            self.layers[i] = Layer(numpy.multiply(.01,numpy.random.standard_normal(size = (self.input,self.input))), False)
        self.layers[-1] = Layer(numpy.multiply(.01,numpy.random.standard_normal(size = (self.input,self.outputNumber))), self.outputNumber == 1)

    #instantly set all weights in network. Matrix is [layer][node][weight]. 
    def setWeights(self, weightMatrix):
        i = 0
        for matrix in weightMatrix:
            self.layers[i].weightMatrix = matrix
            i += 1

    #runs train/test for a set of train/test sets
    def run(self, trainData, learningRate):
        for i in range(10):
            self.train(trainData.getTrainingSet(i), trainData.getTestSet(i), learningRate)
            performance = self.test(trainData.getTestSet(i))
            if self.outputNumber == 1:
                print("K: " + str(i) + " MSE: " + str(performance))
            else:
                print("K: " + str(i) + " ACC: " + str(performance))

    #trains the net
    def train(self, training_set, test_set, learningRate):
        change = [1,1,1,1]
        u = 0
        while numpy.sum(change) > .001:
            x = random.choice(training_set)
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
            del change[0]
            change.append(self.backProp(d, learningRate))
            print(numpy.sum(change))
            u += 1
            if u == 10:
                u = 0
                self.test(test_set)
            

    #tests the net over given test set and array of possible class values (from dataset)
    def test(self, test_set, classes):
        mse = 0
        acc = 0

        for x in test_set:
            y = x[:-1]
            pred = self.makePrediction(y)
            if self.outputNumber == 1:
                mse += pow(x[-1] - pred[0], 2)
            else:
                acc += self.getAcc(pred, x[-1], classes)

        if self.outputNumber == 1:
            mse /= len(test_set)
            return mse
        else:
            acc /= len(test_set)
            return acc

    #makes prediction for given input
    def makePrediction(self, x):

        for layer in self.layers:
            x = layer.feedForward(x)\

        if self.outputNumber == 1:
            return x[0]
        return x

    #backpropagates the error of the network
    def backProp(self, expected, learningRate):

        changeMatrix = []
        delta = numpy.subtract(self.layers[-1].getLayerOutput(), expected)
        delta = numpy.multiply(learningRate, delta)
        changeMatrix.append(self.layers[-1].findOutputChange(delta))
        downstream = self.layers[-1]
        for layer in reversed(self.layers[:-1]):
            changeMatrix.append(layer.findHiddenChange(delta, downstream))
            downstream = layer

        i = 0
        for layer in reversed(self.layers):
            layer.weightMatrix = numpy.subtract(layer.weightMatrix,changeMatrix[i])
            i += 1

        change = 0
        for i in changeMatrix[0]:
            change += abs(i)

        return change

    #checks whether the output predicts the desired class
    def getAcc(self, o, d, classes):
        high = 0
        index = 0
        for i in range(len(o)):
            if o[i] > high:
                high = o[i]
                index = i
        #print("!p: " + str(index))
        #print("!!d: " + str(d))
        if index == classes.index(d):
            return 1
        else:
            return 0


#tData = pre_processing.pre_processing("data/machine.data")
#trainData = dataset.dataset(tData.getData())
#net = FeedForwardNeuralNetwork(9, 1, 1)
#net.run(trainData, .01)
