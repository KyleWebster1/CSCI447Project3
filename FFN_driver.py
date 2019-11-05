import FFN
import pre_processing
import dataset
import random

# randomize data so that when we select training and test sets, we get a variety of each class
def randomizeData(data):
    randomSet = []
    copy = list(data)
    while len(randomSet) < len(data):
        index = random.randrange(len(copy))
        randomSet.append(copy.pop(index))
    return randomSet

files = ["data/segmentation.data",
         "data/forestfires.csv",
         "data/car.data",
         "data/abalone.data",
         "data/machine.data",
         "data/winequality-red.csv",
         "data/winequality-white.csv"]

classification = ["data/segmentation.data",
                  "data/car.data",
                  "data/abalone.data"]

regression = ["data/forestfires.csv",
              "data/machine.data",
              "data/winequality-red.csv",
              "data/winequality-white.csv"]

for f in files:
    print("Pre-Processing file {}".format(f))
    p = pre_processing.pre_processing(f)

    if f in classification:
        print("Processing Categorical Classification using Similarity Matrix")
        inData = p.processClassification(p.getData(), f)
    else:
        inData = p.getData()
    randomizedData = randomizeData(inData)
    data = dataset.dataset(randomizedData)

    # get all training sets
    training_sets = data.getTrainingSet()
    test_sets = data.getTestSet()

    for t1,t2 in zip(training_sets,test_sets):
        net = FFN.ff_neural_net(t1,t2,20,1,100)
        
    
