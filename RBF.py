# Radial Basis Function Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

from KNN import k_nearest_neighbor
import FFN
import random
import math
import pre_processing
import dataset


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

    def __init__(self, training_set, test_set, outputs, gaussians):
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
            self.weights.append([])
            for i in range(0, gaussians):
                self.weights[j].append(random.random())

        # find gaussians unsupervised
        knn_instance = k_nearest_neighbor()
        unsupervised_set = []
        for i in range(len(training_set)):
            sample = self.training_set[i].copy()
            del sample[-1]
            unsupervised_set.append(sample)
        self.gaussians = knn_instance.kMeans(unsupervised_set, gaussians)

        return

    def run_sample(self, sample):

        kernal_values = []
        target = sample[-1]
        del sample[-1]

        # calculate kernal values
        for i in range(len(self.gaussians)):
            kernal_values.append(self.gaussian(sample, i, 1))

        # calculate output values
        output_values = []
        for j in range(self.outputs):
            output_values.append(0)
            for i in range(len(kernal_values)):
                output_values[j] = output_values[j] + (kernal_values[i] * self.weights[j][i])

        return kernal_values, output_values

    def train(self, learning_rate):
        """
        :learning_rate: The learning rate for training this net
        """
        # repeat until convergence
        change = 1
        while change > 0.01:
            sample = self.training_set[random.randrange(len(self.training_set))]  # choose sample at random
            target = sample[-1]

            kernal_values, output_values = self.run_sample(sample)

            # apply gradient descent and track change in weights
            ####I'm not super sure how he wants us to handle having multiple outputs for approximation so here I just use 1####

            if self.outputs == 1:
                change = self.gradient_descent_regression(kernal_values, output_values[0], target, learning_rate)
            else:
                change = self.gradient_descent_classification(kernal_values, output_values, target, learning_rate)

        return

    def test(self):
        """
        :return: average mean squared error of test set
        """
        if (self.outputs == 1):
            mean_sqr_err = 0

            for x in range(len(self.test_set)):
                k, o = self.run_sample(self.test_set[x])
                err = self.test_set[x][-1] - o[0]
                mean_sqr_err += err * err

            mean_sqr_err /= len(self.test_set)

            return mean_sqr_err
        else:
            acc = 0

            for x in range(len(self.test_set)):
                k, o = self.run_sample(self.test_set[x])
                highest_class_value = 0
                for i in range(len(o)):
                    if o[i] > highest_class_value:
                        highest_class = i
                        highest_class_value = o[i]

                if self.test_set[x][-1] == highest_class:
                    acc += 1

            acc /= len(self.test_set)

            return acc

    def gaussian(self, input, i, sigma):
        """
        :param i:
        :param j:
        :return:
        """
        diffVect = self.vector_subtract(input, self.gaussians[i])
        diffMagSqr = self.vector_magnitude_squared(diffVect)
        sigma = 1

        result = math.exp(diffMagSqr / (-2 * sigma))
        return result

    def gradient_descent_regression(self, kernal_values, predicted_value, target_value, learning_rate):

        error = -2 * (target_value - predicted_value)
        weight_change = []
        for i in range(len(kernal_values)):
            weight_change.append(error * kernal_values[i] * learning_rate)

        self.weights[0] = self.vector_add(self.weights[0], weight_change)

        return math.sqrt(self.vector_magnitude_squared(weight_change))

    def gradient_descent_classification(self, kernal_values, output_values, target_class, learning_rate):

        weight_change = []
        average_change = 0

        for j in range(self.outputs):

            weight_change.append([])

            for i in range(len(kernal_values)):
                kernal_out = kernal_values[i] * self.weights[j][i]
                change = (-1*(1-kernal_out)*kernal_out*(1-kernal_out) * kernal_values[i])
                weight_change[j].append(-1 * change * kernal_values[i] * learning_rate)

            self.weights[j] = self.vector_add(self.weights[j], weight_change[j]);
            average_change += math.sqrt(self.vector_magnitude_squared(weight_change[j]))

        average_change /= self.outputs

        return average_change

    def vector_scal(self, x, f):
        """
        f * Vector x
        """
        out = []
        for i in range(len(x)):
            out.append(x[i] * f)

        return out

    def vector_add(self, x, y):
        """
        Vector x + Vector y
        """
        if (len(x) != len(y)):
            print("Vectors not same len")
            return

        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])

        return z

    def vector_subtract(self, x, y):
        """
        Vector x - Vector y
        """
        z = []
        if (len(x) != len(y)):
            print("Vectors not same len")
            print("x:")
            for i in range(len(x)):
                print(x[i])
            print("y:")
            for i in range(len(y)):
                print(y[i])
            return

        for i in range(len(x)):
            z.append(x[i] - y[i])

        return z

    def vector_magnitude_squared(self, x):
        """
        ||Vector x||^2
        """
        mag = 0
        for i in range(len(x)):
            mag += x[i] * x[i]

        return mag

#To test the RBF
tData = pre_processing.pre_processing("data/car.data")
tData2 = pre_processing.pre_processing("data/segmentation.data")
trainData = dataset.dataset(tData.getData())
trainData2 = dataset.dataset(tData2.getData())
rb = rb_neural_net(trainData.getTrainingSet(0), trainData.getTestSet(0), 1, 2)
rb2 = rb_neural_net(trainData2.getTrainingSet(0), trainData2.getTestSet(0), 4, 2)
rb.train(0.1)
rb2.train(0.1)
mse = rb.test()
acc = rb2.test()
print("Mean squared error: " + str(mse))
print("Accuracy: " + str(acc))
