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
    rb = rb_neural_net(trainData.getTrainingSet(i), trainData.getTestSet(i), 4, 4)
    rb.train(0.01)
    mse = rb.test()
    print("Mean squared error: " + str(mse))
