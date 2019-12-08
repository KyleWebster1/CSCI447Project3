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

    def __init__(self,weightMatrix, isLast):
        self.weightMatrix = weightMatrix
        self.isLast = isLast
        self.layerOutput = []

    def feedForward(self, nodeInput):

        v = numpy.matmul(nodeInput, self.weightMatrix)
        if self.isLast:
            v = [v]

        v = self.sigmoid(v)
        return v

    def backPropDelta(self, output, expected):

        matrix = numpy.transpose(self.weightMatrix)
        delta = []
        for i in range(len(matrix)):
            delta.append(numpy.multiply(matrix[i], self.sigDeriv(output)))

        return numpy.transpose(delta)

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

        for i in range(self.numberLayers - 1):
            self.layers.append(Layer(numpy.random.rand(inputs, inputs), False))
        self.layers.append(Layer(numpy.random.rand(outputs,inputs), True))

    def makePrediction(self, x):
        #print(x)
        i = 0
        for layer in self.layers:
            x = layer.feedForward(x)
            #print(str(i) + " " + str(x))
            i += 1

        if self.outputNumber == 1:
            return x[0]

        #print(pred)
        #print(self.weight)
        return x

    def setWeights(self, weightMatricies):
        for i in range(len(self.layers)):
            self.layers[i].weightMatrix = weightMatricies[i]
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
