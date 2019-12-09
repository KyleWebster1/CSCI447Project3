
import numpy
import FFN
import pre_processing
import dataset

class particle:

    def __init__(self, state):
        self.state = state
        self.velocity = [0] * len(state)
        self.p_best = state
        self.fp_best = 10000;

class particle_swarm:

    def __init__(self, training_set, test_set, inertiaW, localW, globalW):

        self.training_set = training_set
        self.training_set_features = training_set.copy()

        self.test_set = test_set

        net = FFN.FeedForwardNeuralNetwork(len(training_set[0]) - 1, 1, 2)
        self.net = net

        self.layerSize = len(training_set[0]) - 1
        self.layers = net.layers

        self.dimensions = (self.layerSize * self.layerSize * (len(self.layers)-1)) + self.layerSize
        print(self.dimensions)

        self.population = []
        for i in range(3 * self.dimensions):
            self.population.append(particle(numpy.random.randn(self.dimensions)))

        self.g_best = self.population[numpy.random.randint(0,len(self.population)-1)].state
        self.inertiaW = inertiaW
        self.localW = localW
        self.globalW = globalW

    def train(self):
        iteration = 0
        while (iteration < 1000):
            fg = self.fitness(self.g_best)
            print("iteration: " + str(iteration) + ", gBest: " + str(fg))
            for p in self.population:
                fx = self.fitness(p.state)
                if (fx <= p.fp_best):
                    p.p_best = p.state
                    p.fp_best = fx
                    if (fx < fg):
                        self.g_best = p.state
                        fg = fx
                        print("!!!GB " + str(fx))
            for p in self.population:
                p.velocity = self.find_velocity(p)
                p.state = numpy.add(p.state, p.velocity)
            iteration += 1

    def find_velocity(self, particle):
        x1 = numpy.multiply(self.inertiaW, particle.velocity)
        x2 = numpy.multiply(numpy.random.uniform(0,self.localW, len(particle.state)), numpy.subtract(particle.p_best, particle.state))
        x3 = numpy.multiply(numpy.random.uniform(0,self.globalW, len(particle.state)), numpy.subtract(self.g_best, particle.state))
        return numpy.add(numpy.add(x1, x2), x3)


    def fitness(self, w):

        self.net.setWeights(self.makeWeightMatrix(w))

        mse = self.net.test(self.training_set)

        return mse

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
        matrix.append([])
        while(n < len(w)):
            matrix[-1].append(w[n])
            n += 1

        return matrix
tData = pre_processing.pre_processing("data/forestfires.csv")
trainData = dataset.dataset(tData.getData())
swarm = particle_swarm(trainData.getTrainingSet(0), trainData.getTestSet(0), .8, .4, .5)
swarm.train()
