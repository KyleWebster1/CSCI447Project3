import math
import random
import FFN
import pre_processing
import dataset
import numpy as np

class GA:
    def __init__(self, dataset, isRegression):

        #store dataset
        self.dataset = dataset

        #Create neural net to be trained
        if (isRegression):
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, 1)
        else:
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, dataset.getNumClasses())

        #initialize population
        self.population = []
        self.popSize = 10
        self.initPop()

        
        self.t = 0
    def initPop(self):
        self.population = []
        #Record info about weights so recombining is easy
        self.num_layers = 0
        self.weight_len = []
        pop = []
        data = []
        for layer in self.net.layers:
            self.num_layers += 1
            newWeightLen = []
            for weight in layer.weightMatrix:
                newWeightLen.append(len(weight))
                for w in weight:
                    data.append(w)
            self.weight_len.append(newWeightLen)

        
        pop.append(data)
        
        
        dataLen = len(data)
        #generate more random weights
        for i in range(self.popSize):
            newData = []
            for j in range(dataLen):
                newData.append(random.gauss(0, .01))
            pop.append(newData)
        self.population.append(pop)
        
    def evalFit(self,chrm,test_set):
        meanOfMeans = 0

        for i in range(len(chrm)):
            meanOfMeans += self.evalSingleFit(chrm[i],test_set)
        return meanOfMeans/(len(chrm))

    def evalSingleFit(self,chrm,test_set):
        
        self.net.setWeights(self.makeWeightMatrix(chrm))

        f = self.net.test(test_set, self.dataset.classes)
        
        return f

    #create weight matrix out of a vector
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
            

    def sort(self,chrm):
        s = chrm
        
        for i in range(len(s)):
            for j in range(len(s)-i):
                if j+1 < len(s):
                    for k in range(len(self.dataset.test_set)):
                        if self.evalSingleFit(s[j],self.dataset.test_set[k]) > self.evalSingleFit(s[j+1],self.dataset.test_set[k]):
                            #swap
                            temp = s[j]
                            s[j] = s[j+1]
                            s[j+1] = temp
        return s

    #select finds parents that will be used to make the next gen
    def select(self,chrm):
        #sort by fitness
        sortedChrm = self.sort(chrm)
        #select the top half
        selectNum = math.floor(len(chrm)/2)
        select = []
        for i in range(selectNum):
            select.append(sortedChrm[i])

        return select

    def unique(self, p): 
        unique_p = []
        
        for x in p: 
            # check if exists in unique_p or not 
            if x not in unique_p: 
                unique_p.append(x)
                
        return unique_p

    def singleCrossover(self,chrm):
        newChrm = []
        #choose place to crossover
        for i in range(len(chrm)):
            for k in range(i+1,len(chrm)):
                
                pt = random.randint(1, len(chrm[i]))

                newChrm.append(chrm[i][:pt]+chrm[k][pt:])
        newChrm = self.unique(newChrm)
        return newChrm

    #recombine uses the parents to make new children
    def recombine(self, chrm):
        #newChrm = uniformCrossover(chrm)

        newChrm = self.singleCrossover(chrm)
                    
        return newChrm

    def mutate(self, chrm):
        newChrm = chrm

        for i in range(len(newChrm)):
            temp = newChrm[i]
            for j in range(len(temp)):
                creep = random.gauss(0,0.1)
                temp[j] = temp[j] + creep
            
            newChrm[i] = temp
        return newChrm

    def replace(self,children,parents):
        '''
        newpop = parents

        for c in children:
            newpop.append(c)

        newpop = self.sort(newpop)

        return newpop[:self.popSize]
        '''
        return children
    def run(self,isRegression):
        fit = []
        for i in range(len(self.dataset.test_set)):
            fit.append(self.evalFit(self.population[self.t], self.dataset.test_set[i]))

            for j in range(2):
                self.t += 1
                c = self.select(self.population[self.t-1])
                cp = self.recombine(c)
                cpp = self.mutate(cp)
                newpop = self.replace(cpp,self.population[self.t-1])
                self.population.append(newpop)
                fit.append(self.evalFit(self.population[self.t], self.dataset.test_set[i]))
                print(fit[self.t])
                if(fit[self.t] < 0.01):
                    j = 20
        if isRegression:
            print ("TOTAL MSE: " + str(fit[self.t]))
        else:
            print ("TOTAL ACC: " + str(fit[self.t]))
        
        

classification = ["data/test.csv"]
regression = ["data/machine.data"]
for file in regression:
    print (file)
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    ga = GA(trainData, True)
    ga.run(True)
for file in classification:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    print (file)
    print ("Num classes: " + str(trainData.getNumClasses()))
    ga = GA(trainData, False)
    ga.run(False)
    







