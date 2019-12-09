#PSO Implementation

import numpy
import FFN
import pre_processing
import dataset
import types

class particle:

    def __init__(self, state):
        self.state = state
        self.velocity = [0] * len(state)
        self.p_best = state
        self.fp_best = None;

class particle_swarm:

    #construct particle swarm using a training set and test set, as well as the three tuning parameters and the amount of outputs for the dataset
    def __init__(self, dataset, inertiaP, localP, globalP, isRegression):

        #store dataset
        self.dataset = dataset

        #Create neural net to be trained
        if (isRegression):
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, dataset.getNumClasses())
        else:
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, 1)


        #initialize population and find the dimensions of the vector for creating the population
        self.population = []
        self.dimensions = 0
        for layer in self.net.layers:
            self.dimensions += layer.weightMatrix.size

        #initialize tuning variables
        self.inertiaP = inertiaP
        self.localP = localP
        self.globalP = globalP

    def initPop(self):
        
        #Common population size is 3 * D
        self.population = []
        for i in range(3 * self.dimensions):
            self.population.append(particle(numpy.random.randn(self.dimensions)))
            
    #train the network on the given train/test sets
    def train(self, training_set):
        iteration = 0
        self.initPop()
        #choose random g_best vector
        self.g_best = self.population[numpy.random.randint(0,len(self.population)-1)].state
        #Repeat for 100 iterations
        while (iteration < 25):
            #find the fitness of g_best vector
            fg = self.fitness(self.g_best, training_set)

            print("iteration: " + str(iteration) + ", gBest: " + str(fg))

            #for each individual p
            for p in self.population:

                #find fitness of p
                fx = self.fitness(p.state, training_set)

                if (self.net.outputNumber == 1):
                    
                    #update personal best if smaller
                    if (p.fp_best == None or fx <= p.fp_best):
                        p.p_best = p.state
                        p.fp_best = fx

                        #update global best if smaller
                        if (fx < fg):
                            self.g_best = p.state
                            fg = fx
                else:
                    #update personal best if bigger
                    if (p.fp_best == None or fx >= p.fp_best):
                        p.p_best = p.state
                        p.fp_best = fx

                        #update global best if bigger
                        if (fx > fg):
                            self.g_best = p.state
                            fg = fx

            #find and update velocity and state using given functions
            for p in self.population:
                p.velocity = self.find_velocity(p)
                p.state = numpy.add(p.state, p.velocity)
            iteration += 1



    #return the velocity of the particle based on tuning parameters inertiaP, localP, and globalP
    def find_velocity(self, particle):
        x1 = numpy.multiply(self.inertiaP, particle.velocity)
        x2 = numpy.multiply(numpy.random.uniform(0,self.localP, len(particle.state)), numpy.subtract(particle.p_best, particle.state))
        x3 = numpy.multiply(numpy.random.uniform(0,self.globalP, len(particle.state)), numpy.subtract(self.g_best, particle.state))
        return numpy.add(numpy.add(x1, x2), x3)


    #return the fitness of a particle's state vector by generating a weight matrix
    def fitness(self, w, test_set):

        #set weights to the weight matrix of the state
        self.net.setWeights(self.makeWeightMatrix(w))

        #test the current net (with updated matricies) on the test set
        f = self.net.test(test_set, self.dataset.classes)

        return f

    #create weight matrix out of a particle state vector
    def makeWeightMatrix(self, w):
        matrix = []
        n = 0
        #for each layer
        for layer in range(len(self.net.layers)):
            matrix.append([])

            #for each node in the layer
            for node in range(len(self.net.layers[layer].weightMatrix)):
                #for each weight in the layer
                matrix[layer].append([])
                for weight in range(len(self.net.layers[layer].weightMatrix[node])):
                    #assign weight as value under pointer
                    matrix[layer][node].append(w[n])
                    #increment pointer
                    n += 1
        return matrix

    #run program across all k-folds
    def run(self):

        k = 0
        f = 0
        for i in range(len(self.dataset.training_set)):
            self.train(self.dataset.training_set[i])
            f += self.fitness(self.g_best, self.dataset.test_set[i])
            k += 1

        f /= k
        if self.net.outputNumber == 1:
            print ("TOTAL MSE: " + str(f))
        else:
            print ("TOTAL ACC: " + str(f))
        

classification = ["data/car.data","data/segmentation.data","data/abalone.data"]
regression = ["data/forestfires.csv","data/machine.data","data/winequality-red.csv","data/winequality-white.csv"]

for file in classification:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    print (file)
    print ("Num classes: " + str(trainData.getNumClasses()))
    swarm = particle_swarm(trainData, .8, .7, 1.3, False)
    swarm.run()

for file in regression:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    swarm = particle_swarm(trainData, .8, .7, 1.3, True)
    swarm.run()
