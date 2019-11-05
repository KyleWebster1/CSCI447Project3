# Feed Forward Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

class neuron:
    #neuron is created with a weight
    #not sure how to represent activation function in this class
    def __init__(self, w):
        self.w = w
        
    def getW(self):
        return self.w
    
    def setW(self, w)
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

    def __init__(self, training_set, test_set, outputs, num_hidden_layers, num_hidden_nodes):
        self.training_set = training_set
        self.test_set = test_set
        self.outputs = outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        # TODO
        # create network

        # initialize weights to zero

    def dot(self, x, w, b):
        """Apply a dot product with a weight value and add a bias onto a vector x
        :param x: The vector to have a weight applied to it. Must be the same size as w.
        :param w: The weight vector to apply to an input vector. Must be the same size as x.
        :param b: The bias value to be added to dot product
        :return: weighted vector x
        """
        for i in range(len(x)):
            x[i] = x[i] * w[i] + b
        return x

    def sigmoid(self, x, isHyper=False):
        """Apply a sigmoid function onto a vector x
        :param x: The input vector to have the sigmoid function applied to it.
        :param isHyper: A binary value with default value false. If False, then use Logistic sigmoid function. If True, then use hyperbolic sigmoid function.
        :return: Returns the vector after having the sigmoid function applied
        """
        # Logistic Sigmoid Function
        if isHyper is False:
            pass
        # Hyperbolic Tangent Sigmoid Function
        else:
            pass

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

