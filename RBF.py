# Radial Basis Function Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import KNN
import FFN
import random

class rb_neural_net:
    """
        A class used to represent a Radial Basis Function Neural Network

        Attributes
        ----------
        training_set: The set of values to train the model
        test_set:
        outputs:
        gaussians:

        Methods
        -------

    """
    def __init__(self,training_set,test_set,outputs,gaussians):
        """
        :training_set: The training set
        :test_set: The test set
        :outputs: The number of outputs
        :gaussians: The number of gaussian functions
        """
        self.training_set = training_set
        self.test_set = test_set
        self.outputs = outputs
        self.weights = [[]]

        for j in range(0, outputs):
            weights.append(m)
            for i in range(0,gaussians):
                weights[j].append(random())


        #find gaussians unsupervised
        knn_instance = k_nearest_neighbor()
        self.gaussians = knn_instance.kMeans(training_set, gaussians)

        return

    def run_sample(sample):

        kernal_values = []
        target = sample[-1]
        del sample[-1]

        #calculate kernal values
        for i in range(len(gaussians)):
            kernal_values.append(gaussian(sample, i))

        #calculate output values
        output_values = []
        for j in range(0,outputs):
            output_values.append(0)
            for i in range(len(kernal_values)):
                output_values[j] += kernal_values[i] * weights[i][j]

        return kernal_values, output_values

    def train(self, learning_rate):
        """
        :learning_rate: The learning rate for training this net
        """
        #repeat until convergence
        change = 1
        while change > 0.01:

            sample = training_set[random.randrange(len(training_set))] #choose sample at random
            target = sample[-1]

            kernal_values, output_values = run_sample(sample)

            #apply gradient descent and track change in weights
            ####I'm not super sure how he wants us to handle having multiple outputs for approximation so here I just use 1####
            change = math.sqrt(vector_magnitude_squared(gradient_descent(kernel_values, output_values[0], target, learning_rate)))

        return

    def test(self):
        """
        :return: average mean squared error of test set
        """
        mean_sqr_err = 0

        for x in range(len(test_set)):
            k, o = run_sample(test_set[x])
            err = test_set[x][-1] - o[0]
            mean_sqr_err += err * err

        mean_sqr_err /= len(test_set)

        return mean_sqr_err


    def gaussian(self, input, i, sigma):
        """
        :param i:
        :param j:
        :return:
        """
        diffVect = vector_subtract(input, gaussians[i]);
        diffMagSqr = vector_magnitude_squared(diffVect)
        sigma = 1

        result = math.exp(diffVect/(-2*sigma))
        #TODO
        return result

    def gradient_descent(kernal_values, predicted_value, target_value, learning_rate):

        error = -2*(target_value - predicted_value)
        weight_change = []
        for i in range(len(kernal_values)):
            weight_change.append(error * kernal_values[i] * learning_rate)

        for i in range
            weights[0] = vector_add(weights[0], weight_change)

        return weight_change

    def vector_add(x, y):
        """
        Vector x + Vector y
        """
        if (len[x] != len[y])
            return

        z = []
        for i in range(len(x))
            z.append(x[i]+y[i])

        return z

    def vector_subtract(x, y):
        """
        Vector x - Vector y
        """
        z = []
        if len(x) != len(y):
            return
        for i in range(len(x))
            z.append(x[i]-y[i])

        return z

    def vector_magnitude_squared(x):
        """
        ||Vector x||^2
        """
        mag = 0
        for i in range(len(x)):
            mag += x[i] * x[i]

        return mag
