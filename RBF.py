# Radial Basis Function Network Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import KNN
import FFN

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
        :param training_set:
        :param test_set:
        :param outputs:
        :param gaussians:
        """
        self.training_set = training_set
        self.test_set = test_set
        self.outputs = outputs
        self.gaussians = gaussians

        #TODO
        
    def train(self, k, momentum):
        """
        :param k:
        :param momentum:
        :return:
        """
        #TODO
        pass

    def test(self, k):
        """
        :param k:
        :return:
        """
        result = None
        #TODO
        return result

    def activation(self, i, j):
        """
        :param i:
        :param j:
        :return:
        """
        result = None
        #TODO
        return result
        
    
