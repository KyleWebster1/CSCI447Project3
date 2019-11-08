# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import pre_processing
from dataset import dataset
from RBF import rb_neural_net
import FFN

files = ["data/car.data",
             "data/forestfires.csv",

             "data/segmentation.data",
             "data/abalone.data",

             "data/machine.data",

             "data/winequality-red.csv",
             "data/winequality-white.csv"]

#for file in files:
#print('=========================================\n' + file + '\n=========================================')
tData = pre_processing.pre_processing("data/abalone.data")
trainData = dataset(tData.getData())
for i in range(10):
    rb = rb_neural_net(trainData.getTrainingSet(i), trainData.getTestSet(i), 25, 8)
    rb.train(0.01)
    mse = rb.test()
    print("Mean squared error: " + str(mse))
"""
for file in files:
    print('=========================================\n' + file + '\n=========================================')
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    knn = KNN.k_nearest_neighbor()
    trainSet = trainData.getTestSet()
    testSet = trainData.getTestSet()
    rb = RBF.rb_neural_net(trainSet, testSet, 4, len(knn.condenseSets(trainSet, testSet, 5)))
    rb.train(0.01)
    acc = rb.test()
    print("Mean squared error: ", acc)
"""
