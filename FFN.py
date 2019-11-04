# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import KNN
import FFN

class ff_neural_net:
    def __init__(self,training_set,test_set,outputs,num_hidden_layers,num_hidden_nodes):
        self.training_set = training_set
        self.test_set = test_set
        self.outputs = outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        #TODO
        #create network

        #initialize weights to zero

    def train(self,training_rate,momentum):
        #TODO

    def test(self):
        result = None
        #TODO
        return result

    def feed_forward(self, j):
        #TODO

    def back_prop(self, j, delta, training_rate, momentum)
        result = None
        #TODO
        return result

    def activation(self, i, j):
        result = None
        #TODO
        return result
