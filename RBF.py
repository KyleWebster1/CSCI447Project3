# Radial Basis Function Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

from KNN import k_nearest_neighbor
import random
import math
import pre_processing
import dataset
import VectorUtilities


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
        self.weights = []

        for j in range(outputs):
            self.weights.append([])
            for i in range(gaussians):
                self.weights[j].append(random.uniform(-.01, .01))

        # find gaussians unsupervised
        knn_instance = k_nearest_neighbor()
        unsupervised_set = []
        for i in range(len(training_set)):
            sample = self.training_set[i].copy()
            del sample[-1]
            unsupervised_set.append(sample)

        self.gaussians, self.clusters = knn_instance.kMeans(unsupervised_set, gaussians)
        print("g: " + str(len(self.gaussians)))
        print("o: " + str(outputs))

        return

    def run_sample(self, sample, target):

        kernal_values = []

        # calculate kernal values
        for i in range(len(self.gaussians)):
            kernal_values.append(self.gaussian(sample, i))

        # calculate output values
        output_values = []
        for j in range(self.outputs):
            output_values.append(float(0))
            for i in range(len(kernal_values)):
                output_values[j] = output_values[j] + (kernal_values[i] * self.weights[j][i])

        return kernal_values, output_values

    def train(self, learning_rate):
        """
        :learning_rate: The learning rate for training this net
        """
        print("training")
        # repeat until convergence
        change = [1,1,1,1]
        running_change = 4
        while running_change > 0.01:
            sample = random.choice(self.training_set).copy()  # choose sample at random
            target = sample[-1]
            del sample[-1]

            kernal_values, output_values = self.run_sample(sample, target)

            # apply gradient descent and track change in weights
            if self.outputs == 1:
                change.append(self.gradient_descent_regression(kernal_values, output_values[0], target, learning_rate))
            else:
                change.append(self.gradient_descent_classification(kernal_values, output_values, target, learning_rate))

            del change[0]
            running_change = change[-1] + change[-2] + change[-3] + change[-4]
        return

    def test(self):
        """
        :return: average mean squared error of test set
        """
        print("testing")
        if (self.outputs == 1):
            mean_sqr_err = 0

            for x in range(len(self.test_set)):
                sample = self.test_set[x].copy()
                target = sample[-1]
                del sample[-1]

                k, o = self.run_sample(sample, target)
                err = target - o[0]
                mean_sqr_err += err * err

            mean_sqr_err /= len(self.test_set)

            return mean_sqr_err
        else:
            acc = 0

            for x in range(len(self.test_set)):
                sample = self.test_set[x].copy()
                target = sample[-1]
                del sample[-1]

                k, o = self.run_sample(sample, target)
                prediction = self.predict_class(o)

                if int(target) == prediction:
                    acc += 1

            acc /= len(self.test_set)

            return acc

    def gaussian(self, input, i):
        """
        :param i:
        :param j:
        :return:
        """
        diffVect = VectorUtilities.vector_subtract(input, self.gaussians[i])
        diffMagSqr = VectorUtilities.vector_magnitude_squared(diffVect)
        sig = self.sigma(i)
        if (sig != 0):
            result = math.exp(-diffMagSqr / (2 * math.pow(, 2)))
        else:
            result = 0

        return result

    def sigma(self, i):

        mean_dist = 0
        for n in range(len(self.clusters[i])):
            mean_dist += math.sqrt(VectorUtilities.vector_magnitude_squared(VectorUtilities.vector_subtract(self.clusters[i][n], self.gaussians[i])))
        mean_dist /= len(self.clusters[i])

        return mean_dist

    def gradient_descent_regression(self, kernal_values, predicted_value, target_value, learning_rate):

        error = (target_value - predicted_value)
        weight_change = []

        for i in range(len(kernal_values)):
            weight_change.append(error * kernal_values[i] * learning_rate)
        self.weights[0] = VectorUtilities.vector_add(self.weights[0], weight_change)

        return math.sqrt(VectorUtilities.vector_magnitude_squared(weight_change))

    def gradient_descent_classification(self, kernal_values, output_values, target_class, learning_rate):

        d = []
        for j in range(len(output_values)):
            if j == int(target_class):
                d.append(1)
            else:
                d.append(0)

        total_change = 0
        change_vectors = []
        for j in range(len(output_values)):
            change = learning_rate * (d[j]-output_values[j])
            change_vectors.append(VectorUtilities.vector_scale(kernal_values, change))
            total_change += change

        for j in range(len(output_values)):
            self.weights[j] = VectorUtilities.vector_add(self.weights[j], change_vectors[j])

        return total_change / len(output_values)

    def predict_class(self, output_values):
        highest_class_value = None
        for i in range(len(output_values)):
            if highest_class_value == None or output_values[i] > highest_class_value:
                highest_class = i
                highest_class_value = output_values[i]
        return highest_class
