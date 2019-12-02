
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
        self.weightSize = len(net.weight[1][0])
        self.weightCount =  len(net.weight[1])
        self.layerCount = net.numberLayers
        self.dimensions = weightSize * weightCount * layerCount

        self.population = []
        for i in range(3 * d):
            population.append(particle(np.random.randn(dimensions)))

        self.g_best = [0] * dimensions
        self.inertiaW = inertiaW
        self.localW = localW
        self.globalW = globalW

    def train(self):
        iteration = 0
        while (iteration < 1000):
            fg = try_weight(g_best)
            for p in population:
                fx = try_weight(p.state)
                fp = try_weight(p.p_best)
                if (fx < fp):
                    p.p_best = p.state
                    if (fx < fg):
                        g_best = newState
                        fg = try_weight(g_best)
            for p in population:
                p.velocity = find_velocity(p)
                p.state = np.add(p, p.velocity)

    def find_velocity(self, particle):
        return (inertiaW * particle.velocity) + (localW * particle.p_best) + (globalW * g_best)


    def try_weight(self, w):

        net.weight = makeWeightMatrix(self, w)

        mse = 0
        for x in training_set:
            mse += net.regressionPred(x)

        mse /= len(training_set)

        return mse

    def makeWeightMatrix(self, w):
        matrix = [[]]
        for l in range(layerCount):
            matrix.append([])
            n = 0
            for i in range(weightCount):
                for j in range(weightSize):
                    matrix[l].append(weight_array[n])
                    n += 1
        return matrix
tData = pre_processing.pre_processing("data/forestfires.csv")
trainData = dataset.dataset(tData.getData())
swarm = particle_swarm(trainData.getTrainingSet(0), trainData.getTestSet(0), .1, .4, .5)
