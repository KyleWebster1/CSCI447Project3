# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin
import random
import math

class Neuron:
    #neuron is created with a weight
    #not sure how to represent activation function in this class
    def __init__(self, w, b):
        self.w = w
        self.bias = b
    def dot(self, x= []):
        """Apply a dot product with a weight value and add a bias onto a vector x
        :param x: The vector to have a weight applied to it. Must be the same size as w.
        :param w: The weight vector to apply to an input vector. Must be the same size as x.
        :param b: The bias value to be added to dot product
        :return: the finalized dot product
        """
        for i in range(len(x)):
            x[i] = x[i] * self.w[i] + self.b
        return sum(x)

    def sigmoid(self, x = [], isHyper=False):
        """Apply a sigmoid function onto a vector x
        :param x: The input vector into the node to have the sigmoid function applied to it.
        :param isHyper: A binary value with default value false. If False, then use Logistic sigmoid function. If True, then use hyperbolic sigmoid function.
        :return: Returns the vector after having the sigmoid function applied
        """
        # Logistic Sigmoid Function
        if isHyper is False:
            return(1/(1+math.exp(-1*Neuron.dot(x))))
        # Hyperbolic Tangent Sigmoid Function
        else:
            return(math.tan(Neuron.dot(x)))
    def __str__(self):
        return str(self.w)
    def activation(self, inputs=[], isSigmoidal= False):
        if isSigmoidal:
            return self.sigmoid(inputs)
        else:
            return self.dot(inputs)
    def getW(self):
        return self.w

    def setW(self, w):
        self.w = w

class ff_neural_net:
    """
        A class used to represent a Feed Forward Neural Network

        Attributes
        ----------
        training_set: The training values
        test_set: The values to test the model with
        outputs: The number of outputs
        num_hidden_layers: The number of hidden layers
        num_hidden_nodes: The number of hidden nodes

        Methods
        -------
        dot(x,w,b)
            Returns the dot product between vector x and a weight vector w with a bias addition b.
        sigmoid(self, x, isHyper=False)
            Returns the vector x after being affected by a sigmoid function
        train(training_rate, momentum)

        test()

        feed_forward(j)

        back_prop(j, delta, training_rate, momentum)

        activation(i, j)
        """

    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes

        # initialize weights for hidden layers
        self.hWeights = []
        for i in range(self.num_hidden_layers):
            layer_w = []
            for j in range(self.num_hidden_nodes):
                neuron = Neuron(random.random())
                layer_w.append(neuron)
            self.hWeights.append(layer_w)

        self.oWeights = []
        for j in range(self.num_outputs):
            neuron = Neuron(random.random())
            self.oWeights.append(neuron)






    def train(self, training_rate, momentum):
        pass
        # TODO

    def test(self):
        result = None
        # TODO
        return result

    def feed_forward(self, j):
        pass
        # TODO

    def back_prop(self, j, delta, training_rate, momentum):
        result = None
        # TODO
        return result

    def activation(self, i, j):
        result = None
        # TODO
        return result

