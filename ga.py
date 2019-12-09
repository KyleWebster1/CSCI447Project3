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
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, dataset.getNumClasses())
        else:
            self.net = FFN.FeedForwardNeuralNetwork(len(dataset.training_set[0][0]) - 1, 2, 1)

        t = 0
    
        #init population
        pop = []
        data = []

        for i in range(50):
            data.append([random.randint(-3, 3),random.randint(-3, 3),random.randint(-3, 3),random.randint(-3, 3)])
        
        pop.append(data)

        #init fitness
        fit = []
        fit.append(self.evalFit(pop[t]))

        while(self.evalFit(pop[t]) > .05):
            t += 1
            c = self.select(pop[t-1])
            cp = self.recombine(c)
            cpp = self.mutate(cp)
            newpop = self.replace(cpp,pop[t-1])
            pop.append(newpop)
            fit.append(self.evalFit(pop[t]))


        print(pop[t])
        print(fit[t])
        print("# of iterations: ",t)
        print("# of unique indiv from pop: ",len(self.unique(pop[t])))
        pop[t] = self.sort(pop[t])
        print(pop[t][:1])
        print("//////////")        


        
    def evalFit(self,chrm):
        ideal = [4,4,4,4]

        meanOfMeans = 0

        for i in range(len(chrm)):
            meanOfMeans += self.evalSingleFit(chrm[i])
        return meanOfMeans/(len(chrm))

    def evalSingleFit(self,chrm):
        ideal = [4,4,4,4]
        mean = 0
        for i in range(len(chrm)):
            mean += abs(chrm[i]-ideal[i])
        return mean/len(chrm)

    def sort(self,chrm):
        s = chrm
        
        for i in range(len(s)):
            for j in range(len(s)-i):
                if j+1 < len(s):
                    if self.evalSingleFit(s[j]) > self.evalSingleFit(s[j+1]):
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

    def uniformCrossover(self, chrm):
        newChrm = []
        #crossover each bit randomly
        for i in range(len(chrm)):
            for k in range(len(chrm)):
                p1 = chrm[i]
                p2 = chrm[k]
                for x in range(len(p1)):
                    if bool(random.getrandbits(1)) and p1[x] != p2[x]:
                        temp = p1[x]
                        p1[x] = p2[x]
                        p2[x] = temp

                        newChrm.append(p1)
                        newChrm.append(p2)
                

                chrm[i] = p1
                chrm[k] = p2
        newChrm = unique(newChrm)

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
                creep = random.gauss(0,0.01)
                temp[j] = round(temp[j] + creep,3)
            
            newChrm[i] = temp
        return newChrm

    def replace(self,children,parents):
        newpop = parents

        for c in children:
            newpop.append(c)

        newpop = sort(newpop)

        return newpop[:50]

def main():
    layerData = []
    layerWeightData = []
    weightData = []
    
    for layer in ffn.weight:
        layerData.append(len(ffn.weight[layer]))
        for lw in range(len(ffn.weight[layer])):
            layerWeightData.append(len(ffn.weight[layer][lw]))
            for w in ffn.weight[layer][lw]:
                weightData.append(w)

    for i in range (2000):
        ffn.update(.1, x, y)

    reg = ffn.regressionPred(x)
    print(reg)
    print(y)
    print(ffn.MSE(y, reg))

classification = ["data/car.data","data/segmentation.data","data/abalone.data"]
regression = ["data/forestfires.csv","data/machine.data","data/winequality-red.csv","data/winequality-white.csv"]

for file in classification:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    print (file)
    print ("Num classes: " + str(trainData.getNumClasses()))
    ga = GA(trainData, False)
    ga.run()

for file in regression:
    tData = pre_processing.pre_processing(file)
    trainData = dataset.dataset(tData.getData())
    ga = GA(trainData, True)
    ga.run()
