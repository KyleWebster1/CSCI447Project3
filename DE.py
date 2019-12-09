import FFN
import pre_processing
import dataset
import numpy as np
import random

class differentialEvolutionL:
    def __init__(self, trainingSet, mutation, crossOver, net, maxIter, populationSize):
        self.trainingSet = trainingSet
        self.training = random.choices(self.trainingSet, k = (int(len(trainingSet)*.1)))
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
            if vector[i] < self.bounds[i%len(self.trainingSet[0])][0]:
                vector[i] = self.bounds[i%len(self.trainingSet[0])][0]
            elif vector[i] > self.bounds[i%len(self.trainingSet[0])][1]:
                vector[i] = self.bounds[i%len(self.trainingSet[0])][0]
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
                offspring = []
                #Crossover
                for k in range(len(trial)):
                    crossover = random.random()
                    if crossover <= self.crossOver:
                        offspring.append(target[k])
                    else:
                        offspring.append(trial[k])
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
        f = self.net.test(test_set, self.trainingSet.classes)

        return f

tData = pre_processing.pre_processing("data/car.data")
trainData = dataset.dataset(tData.getData())
net = FFN.FeedForwardNeuralNetwork(len(trainData.getTrainingSet(0)[0]) - 1, trainData.getNumClasses(), 2)
newWeights = differentialEvolutionL(trainData.getTrainingSet(0), .2, .76, net, 25, 20)
print(newWeights.train)
net.setWeights(newWeights.makeWeightMatrix(newWeights.train))
for i in trainData.getTestSet(0):
    print(net.makePrediction(i[:-1]))
    print(i[-1])
