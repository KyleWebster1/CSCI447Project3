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

    def change_weights(self, layer, delta, inp, momentum):
        return layer.delta * inp.T * momentum
 
    def backprop(self, data, correct_answer, momentum, output):
        pass
        #TODO 
    def train(self, net, actual, momentum, max_iterations):
        mse = 1
        count = 0
        while(mse > 0.1):
            for j in range(len(net)):
                output = self.feed_forward(net[j])
                self.backprop(net[j], actual[j], momentum, output)
            mse = self.MSE(net, actual)

            count+=1

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
tData = pre_processing.pre_processing("data/car.data")
trainData = dataset.dataset(tData.getData())
x=np.array(trainData.getTrainingSet())

#Simple test case. AND gate
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
ffn = FeedForwardNeuralNetwork()
ffn.add(Layer(2, 4))
ffn.add(Layer(4, 4))
ffn.add(Layer(4, 4))
ffn.add(Layer(4, 4))
ffn.add(Layer(4, 3))

ffn.train(x,y,0.2,500)
