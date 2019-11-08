# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import pre_processing
from dataset import dataset
from RBF import rb_neural_net
import FFN

classification = ["data/car.data","data/segmentation.data","data/abalone.data"]
regression = ["data/forestfires.csv","data/machine.data","data/winequality-red.csv","data/winequality-white.csv"]

# print('=========================================\n' + "CLASSIFICATION" + '\n=========================================')
#
# for n in range(len(classification)):
#     print('=========================================\n' + classification[n] + '\n=========================================')
#
#     tData = pre_processing.pre_processing(classification[n])
#     trainData = dataset(tData.getData())
#     acc = 0
#     if n == 0:
#         o = 6
#         g = 8
#     elif n == 1:
#         o = 10
#         g = 12
#     else:
#         o = 25
#         g = 12
#     for i in range(10):
#         rb = rb_neural_net(trainData.getTrainingSet(i), trainData.getTestSet(i), o, g)
#         rb.train(0.01)
#         acc += rb.test()
#
#     #acc *= 100
#     print("Accuracy: " + str(acc))


for n in range(len(classification)):
    print('=========================================\n' + classification[n] + '\n=========================================')

    tData = pre_processing.pre_processing(classification[n])
    trainData = dataset(tData.getData())
    mse = 0
    for i in range(10):
        rb = rb_neural_net(trainData.getTrainingSet(i), trainData.getTestSet(i), 1, 8)
        rb.train(0.01)
        mse += rb.test()

    mse /= 10
    print("Mean Squared Error: " + str(mse))

for n in range(len(regression)):
    print('=========================================\n' + regression[n] + '\n=========================================')

    tData = pre_processing.pre_processing(regression[n])
    trainData = dataset(tData.getData())
    mse = 0
    for i in range(10):
        rb = rb_neural_net(trainData.getTrainingSet(i), trainData.getTestSet(i), 1, 8)
        rb.train(0.01)
        mse += rb.test()

    mse /= 10
    print("Mean Squared Error: " + str(mse))

