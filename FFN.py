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


    def feedForward(self, nodeInput):

        self.nodeInput = nodeInput
        v = self.getLayerOutput()
        if self.isLastReg:
            v = [v]

        v = self.sigmoid(v)

        return v

    def getNodeOutput(self, k):
        return numpy.multiply(self.nodeInput[k], numpy.transpose(self.weightMatrix)[k])

    def getLayerOutput(self):
        return numpy.matmul(self.nodeInput, self.weightMatrix)

    def findWeightDelta(self, delta):
        weight_delta = []
        for i in range(len(self.nodeInput)):
            weight_delta.append([])
            for j in range(len(delta)):
                weight_delta[i].append(self.nodeInput[i] * delta[j])

        return weight_delta

    def findOutputDelta(self,target):
        o = self.getLayerOutput()
        delta = []
        for j in range(len(o)):
            delta.append(-(target[j] - o[j]) * o[j] * (1-o[j]))

        return delta

    def findHiddenDelta(self, downstream, kdelta):
        o = self.getLayerOutput()
        delta = []
        for j in range(len(o)):
            sum = 0
            for k in range(len(kdelta)):
                
                sum += kdelta[k] * downstream.weightMatrix[j][k]
            print(sum)
            delta.append(o[j]*(1-o[j]) * sum)

        return delta
    def sigmoid(self, input):
        for i in range(len(input)):
            input[i] = 1 / (1 + numpy.exp(-input[i]))
        return input

    def sigDeriv(self, output):
        for i in range(len(output)):
            output[i] = output[i] * (1 - output[i])
        return output



class FeedForwardNeuralNetwork:
    #Inspired from code from hackermoon.com blogpost by Niranjan Kumar
    def __init__(self, inputs, outputs, hiddenLayers):
        self.input = inputs
        self.outputNumber = outputs
        self.numberLayers = hiddenLayers + 1
        self.layers = []

        self.initWeights()

    def initWeights(self):
        self.layers = [0] * self.numberLayers
        for i in range(self.numberLayers - 1):
            self.layers[i] = Layer(numpy.random.rand(self.input, self.input), False)
        self.layers[-1] = Layer(numpy.random.rand(self.input, self.outputNumber), self.outputNumber == 1)
        
    def run(self, trainData, learningRate):
        for i in range(10):
            self.train(trainData.getTrainingSet(i), learningRate)
            performance = self.test(trainData.getTestSet(i))
            if self.outputNumber == 1:
                print("K: " + str(i) + "MSE: " + str(performance))
            else:
                print("K: " + str(i) + "ACC: " + str(performance))
            
            
    def train(self, training_set, learningRate):
        change = 10000
        while change > .0001:
            x = random.choice(training_set)
            y = x[:-1]
            pred = self.makePrediction(y)
            if (self.outputNumber == 1):
                d = [pred]
            else:
                d = pred.copy()
                for i in range(len(d)):
                    if i == x[-1]:
                        d[i] = 1
                    else:
                        d[i] = 0
                
            change = self.backProp(d, learningRate)
            #print(change)


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

    def makePrediction(self, x):
        
        for layer in self.layers:
            x = layer.feedForward(x)

        if self.outputNumber == 1:
            return x[0]
        return x

    def update_weights(self, layer, delta, learningRate):
        weight_change = layer.findWeightDelta(delta)
        weight_change = numpy.multiply(-learningRate, weight_change)
        layer.weightMatrix = numpy.add(layer.weightMatrix, weight_change)

    def backProp(self, expected, learningRate):

        totalDelta = []
        delta = self.layers[-1].findOutputDelta(expected)
        totalDelta.append(delta)
        downstream = self.layers[-1]
        i = 0
        for layer in reversed(self.layers[:-1]):
            print(i)
            i += 1
            delta = layer.findHiddenDelta(downstream, delta)
            totalDelta.append(delta)
            downstream = layer
            
        i = 0
        for layer in reversed(self.layers):
            self.update_weights(layer, totalDelta[i], learningRate)
            i += 1

        change = 0
        for j in totalDelta[0]:
            change += abs(j)

        return change

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

tData = pre_processing.pre_processing("data/car.data")
trainData = dataset.dataset(tData.getData())
net = FeedForwardNeuralNetwork(6, 7, 1)
net.run(trainData, .001)

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
