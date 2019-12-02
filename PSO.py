
import numpy
import FFN
import pre_processing
import dataset

class particle:

    def __init__(self, state):
        self.state = state
        self.velocity = [0] * len(state)
        self.p_best = state

class particle_swarm:

    def __init__(self, training_set, test_set, inertiaW, localW, globalW):

        self.training_set = training_set
        self.test_set = test_set

        net = FFN.FeedForwardNeuralNetwork(len(training_set[0]) - 1, len(training_set[0]) - 1, [1])
        self.net = net
        
        self.weightSize = net.sizes[1]
        self.weightCount =  net.sizes[0]
        self.layerCount = net.numberLayers
        self.dimensions = self.weightSize * self.weightCount * self.layerCount
        print(self.dimensions)
        self.population = []
        for i in range(3 * self.dimensions):
            self.population.append(particle(numpy.random.randn(self.dimensions)))

        self.g_best = [0] * self.dimensions
        self.inertiaW = inertiaW
        self.localW = localW
        self.globalW = globalW

    def train(self):
        iteration = 0
        while (iteration < 1000):
            fg = self.try_weight(self.g_best)
            print("iteration: " + str(iteration) + ", gBest: " + str(fg))
            for p in self.population:
                fx = self.try_weight(p.state)
                fp = self.try_weight(p.p_best)
                if (fx < fp):
                    p.p_best = p.state
                    if (fx < fg):
                        self.g_best = newState
                        fg = self.try_weight(g_best)
            for p in self.population:
                p.velocity = self.find_velocity(p)
                p.state = numpy.add(p, p.velocity)

    def find_velocity(self, particle):
        return (inertiaW * particle.velocity) + (localW * particle.p_best) + (globalW * g_best)


    def try_weight(self, w):

        self.net.weight = self.makeWeightMatrix(w)

        mse = 0
        for x in self.training_set:
            print(*x)
            mse += self.net.regressionPred(x)

        mse /= len(self.training_set)

        return mse

    def makeWeightMatrix(self, w):
        matrix = []
        for l in range(self.layerCount):
            matrix.append([])
            n = 0
            for i in range(self.weightCount):
                for j in range(self.weightSize):
                    matrix[l].append(w[n])
                    n += 1
        return matrix
tData = pre_processing.pre_processing("data/forestfires.csv")
trainData = dataset.dataset(tData.getData())
swarm = particle_swarm(trainData.getTrainingSet(0), trainData.getTestSet(0), .1, .4, .5)
swarm.train()
