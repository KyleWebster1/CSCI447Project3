# class for storing the data sets
class dataset:
    total_set = []  # total data set
    training_set = []  # list of training sets
    test_set = []  # list of test sets (respective to training sets)
    classes = []

    def __init__(self, data):
        self.total_set = data
        self.kFoldCross(10)
        self.getClasses()

    def getTotalSet(self):
        return self.total_set

    def getTrainingSet(self,k):
        return self.training_set[k]

    def getTestSet(self, k):
        return self.test_set[k]

    def getClasses(self):
        for t in self.total_set:
            classValue = t[-1]

            if classValue not in self.classes:
                self.classes.append(classValue) 

    def getNumClasses(self):
        return len(self.classes)

    # k is number of training/test set pairs
    def kFoldCross(self, k):
        training_set = []
        test_set = []
        splitRatio = .9
        for i in range(k):
            testSize = int(len(self.total_set) - len(self.total_set) * splitRatio)
            index = i * testSize

            trainSet = list(self.total_set)
            testSet = []

            for j in range(testSize):
                testSet.append(trainSet.pop(index))

            training_set.append(trainSet)
            test_set.append(testSet)
        self.training_set = training_set
        self.test_set = test_set
