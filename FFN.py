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
    def __init__(self):
        self.total_layers = []

    def MSE(self, net, actual):
        return np.mean(np.square(actual - ffn.feed_forward(net)))

    def add(self, layer):
        self.total_layers.append(layer)

    def feed_forward(self, node):
        for layer in self.total_layers:
            node = layer.activate_function(node)
        return node

    def find_error(self, correct_answer, output):
        return correct_answer - output

    def calc_delta(self, error, deriv):
        return error*deriv

    def change_weights(self, layer, delta, weights, momentum, inp):
        #print((layer.delta*momentum)/weights)
        return delta*inp.T*momentum

    def backprop(self, data, correct_answer, momentum, output):
        for i in reversed(range(len(self.total_layers))):
            layer = self.total_layers[i]
            if(layer == self.total_layers[-1]):
                layer.error = self.find_error(correct_answer,output)
                layer.delta = layer.sigmoid_deriv(output)*layer.error
            else:
                next_layer = self.total_layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error*layer.sigmoid_deriv(layer.already_activated)
        for i in range(len(self.total_layers)):
            layer = self.total_layers[i]
            if(i==0):
                inp = np.atleast_2d(data)
            else:
                inp = np.atleast_2d(self.total_layers[i-1].already_activated)
            layer.weights += layer.delta * inp.T * momentum
        #TODO
    def final_pass(self, net):
        ffn = self.feed_forward(net)
        if ffn.ndim == 1:
            return np.argmax(ffn)
        return np.argmax(ffn, axis=1)

    def train(self, net, actual, momentum, max_iterations):
        mse = 1
        count = 0
        #while(mse > 0.1):
        for i in range(max_iterations):
            for j in range(len(net)):
                output = self.feed_forward(net[j])
                self.backprop(net[j], actual[j], momentum, output)
            #mse = self.MSE(net, actual)
            count+=1

    def accuracy(self,true, pred):
        return (pred == true).mean()

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.bias = np.random.rand(num_neurons)
        self.already_activated = None
        self.error = None
        self.delta = None

    def activate_function(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.already_activated = self.sigmoid(r)
        return self.already_activated

    def sigmoid(self, r):
        return 1 / (1 + np.exp(-r))

    def sigmoid_deriv(self, r):
        return r * (1 - r)

#TODO
"""tData = pre_processing.pre_processing("data/car.data")
trainData = dataset.dataset(tData.getData())
x=np.array(trainData.getTrainingSet(0))
x = x[:,:-1]
y = x[:,-1]
size = x.shape #6 is input nodes Last column is correct_answer
print(size[1])
#Simple test case. AND gate
#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [1]])
ffn = FeedForwardNeuralNetwork()
ffn.add(Layer(size[1], size[1]))
ffn.add(Layer(size[1], size[1]))
#ffn.add(Layer(4, 4))
#ffn.add(Layer(4, 4))
ffn.add(Layer(size[1], 4))

ffn.train(x,y,0.01,2000)
print('Accuracy: %.2f%%' % (ffn.accuracy(ffn.final_pass(x), y.flatten()) * 100))
#print(ffn.accuracy(final_x,y))
