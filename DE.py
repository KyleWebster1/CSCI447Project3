import FFN
import pre_processing
import dataset
import numpy as np
import random

class differentialEvolutionL:
    def __init__(self, trainingSet, testSet, mutation, crossOver, net, maxIter, populationSize):
        self.trainingSet = trainingSet
        self.testSet = testSet
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
            if vector[i] < self.bounds[i][0]:
                vector[i] = self.bounds[i][0]
            elif vector[i] > self.bounds[i][1]:
                vector[i] = self.bounds[i][0]
        return vector

    @property
    def train(self):
        bestInd = []

        population = []
        for i in range(self.populationSize):
            individual = []
            for j in range(self.layerSize):
                individual.append(random.uniform(self.bounds[j][0], self.bounds[j][1]))
            population.append(individual)

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
                    trial.append(x1[k]-self.mutation*x2[k]-x3[k])
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
                strial = self.fitness(offspring)
                starget = self.fitness(target)
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
        for l in range(len(self.layers) -1):
            matrix.append([])
            for i in range(self.layerSize):
                matrix[l].append([])
                for j in range(self.layerSize):
                    matrix[l][i].append(w[n])
                    n += 1
                n = 0
        matrix.append([])
        while(n < len(w)):
            matrix[-1].append(w[n])
            n += 1

        return matrix

    def fitness(self, w):

        self.net.setWeights(self.makeWeightMatrix(w))

        mse = 0
        for x in self.trainingSet:
            y = x.copy()
            del y[-1]
            e = x[-1] - self.net.makePrediction(y)
            mse += e * e

        mse /= len(self.trainingSet)

        return mse

tData = pre_processing.pre_processing("data/car.data")
trainData = dataset.dataset(tData.getData())
net = FFN.FeedForwardNeuralNetwork(len(trainData.getTrainingSet(0)) - 1, trainData.getNumClasses(), 2)
newWeights = differentialEvolutionL(trainData.getTrainingSet(0), trainData.getTestSet(0), .2, .76, net, 25, 20)
print(newWeights.train)
net.setWeights(newWeights.makeWeightMatrix(newWeights.train))
for i in trainData.getTestSet(0):
    print(net.makePrediction(i[:-1]))
    print(i[-1])
