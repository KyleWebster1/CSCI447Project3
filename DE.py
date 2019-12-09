import FFN
import pre_processing
import dataset
import numpy as np
import random

class differentialEvolution:
    """
    Class performs the differential evolution learning for an input neural network
    trainingSet is the input training data
    mutation is the mutation rate
    crossOver is the crossover rate
    net is the network to be trained
    maxIter is the maximum number of iterations the algorithm will run
    populationSize is the size of the population for a given generation
    """
    def __init__(self, trainingSet, mutation, crossOver, net, maxIter, populationSize):
        self.trainingSet = trainingSet
        self.classes = []
        self.training = random.choices(self.trainingSet, k = (int(len(trainingSet)*.1)))
        for i in range(len(self.training)):
            self.classes.append(self.training[i][-1])
        self.mutation = mutation
        self.crossOver = crossOver
        self.layerSize = len(trainingSet[0]) - 1
        self.layers = net.layers
        self.net = net
        self.maxIter = maxIter
        self.populationSize = populationSize
        self.bounds = []
        for i in range(self.layerSize):
            min = 100000
            max = 0
            for vec in self.trainingSet:
                if vec[i] < min:
                    min = vec[i]
                if vec[i] > max:
                    max = vec[i]
            self.bounds.append((min, max))

    def fixBounds(self, vector):
        for i in range(len(vector)):
            if vector[i] < self.bounds[i%(len(self.trainingSet[0])-2)][0]:
                vector[i] = self.bounds[i%(len(self.trainingSet[0])-2)][0]
            elif vector[i] > self.bounds[i%(len(self.trainingSet[0])-2)][1]:
                vector[i] = self.bounds[i%(len(self.trainingSet[0])-2)][0]
        return vector

    @property
    def train(self):
        bestInd = []

        population = []
        dimensions = 0
        #Adjusts length of vectors based on number of layers
        for layer in self.net.layers:
            dimensions += layer.weightMatrix.size
        #initialize population
        for i in range(self.populationSize):
            r = np.random.randn(dimensions)
            population.append(r)
        print("Population", population)
        for i in range(self.maxIter):
            generationScores = []
            for j in range(self.populationSize):
                #Selection
                candidates = range(0, self.populationSize)
                randomIndex = random.sample(candidates, 4)

                x1 = population[randomIndex[0]]
                x2 = population[randomIndex[1]]
                x3 = population[randomIndex[2]]
                target = population[randomIndex[3]]
                #Mutation
                trial = []
                for k in range(len(x1)):
                    trial.append(x1[k]+self.mutation*(x2[k]-x3[k]))
                trial = self.fixBounds(trial)
                print("Mutation: ", trial)
                offspring = []
                #Crossover
                for k in range(len(trial)):
                    crossover = random.random()
                    if crossover <= self.crossOver:
                        offspring.append(target[k])
                    else:
                        offspring.append(trial[k])
                print("Crossovered Offspring:", offspring)
                #Selection
                strial = self.fitness(offspring, self.training)
                starget = self.fitness(target, self.training)
                if strial < starget:
                    population[j] = trial
                    generationScores.append(strial)
                else:
                    generationScores.append(starget)
                bestIndex = generationScores.index(min(generationScores))
                bestInd = population[bestIndex]
                print("Best Individual of generation:", bestInd)
        return bestInd

    def makeWeightMatrix(self, w):
        matrix = []
        n = 0
        # for each layer
        for layer in range(len(self.net.layers)):
            matrix.append([])

            # for each node in the layer
            for node in range(len(self.net.layers[layer].weightMatrix)):
                # for each weight in the layer
                matrix[layer].append([])
                for weight in range(len(self.net.layers[layer].weightMatrix[node])):
                    # assign weight as value under pointer
                    matrix[layer][node].append(w[n])
                    # increment pointer
                    n += 1
        return matrix

    def fitness(self, w, test_set):

        # set weights to the weight matrix of the state
        self.net.setWeights(self.makeWeightMatrix(w))

        # test the current net (with updated matricies) on the test set
        f = self.net.test(test_set, self.classes)

        return f
classification = ["data/car.data", "data/segmentation.data", "data/abalone.data"]
regression = ["data/forestfires.csv", "data/machine.data", "data/winequality-red.csv", "data/winequality-white.csv"]


for file in classification:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    print(file)
    print("Num classes: " + str(trainData.getNumClasses()))
    k = 0
    f = 0
    for i in range(len(trainData.training_set)):
        net = FFN.FeedForwardNeuralNetwork(len(trainData.getTrainingSet(i)[0]) - 1, trainData.getNumClasses(), 2)
        newWeights = differentialEvolution(trainData.getTrainingSet(i), .8, .76, net, 500, 20)
        best = newWeights.train
        f += newWeights.fitness(best, trainData.test_set[i])
        k += 1

    f /= k
    print("TOTAL ACC: " + str(f))

for file in regression:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    print(file)
    k = 0
    f = 0
    for i in range(len(trainData.training_set)):
        net = FFN.FeedForwardNeuralNetwork(len(trainData.getTrainingSet(i)[0]) - 1, trainData.getNumClasses(), 2)
        newWeights = differentialEvolution(trainData.getTrainingSet(i), .8, .76, net, 500, 20)
        best = newWeights.train
        f += newWeights.fitness(best, trainData.test_set[i])
        k += 1

    f /= k
    print("TOTAL MSE: " + str(f))
