import math
import random
import FFN
import pre_processing
import dataset
import numpy as np

#Current Fitness Eval is just for testing. Will be updated to be
#related to the error of the network
def evalFit(chrm):
    ideal = [4,4,4,4]

    meanOfMeans = 0

    for i in range(len(chrm)):
        meanOfMeans += evalSingleFit(chrm[i])
    return meanOfMeans/(len(chrm))

def evalSingleFit(chrm):
    ideal = [4,4,4,4]
    mean = 0
    for i in range(len(chrm)):
        mean += abs(chrm[i]-ideal[i])
    return mean/len(chrm)

def sort(chrm):
    s = chrm
    
    for i in range(len(s)):
        for j in range(len(s)-i):
            if j+1 < len(s):
                if evalSingleFit(s[j]) > evalSingleFit(s[j+1]):
                    #swap
                    temp = s[j]
                    s[j] = s[j+1]
                    s[j+1] = temp
    return s

#select finds parents that will be used to make the next gen
def select(chrm):
    #sort by fitness
    sortedChrm = sort(chrm)
    #select the top half
    selectNum = math.floor(len(chrm)/2)
    select = []
    for i in range(selectNum):
        select.append(sortedChrm[i])
                
    return select

def unique(p): 
    unique_p = []
    
    for x in p: 
        # check if exists in unique_p or not 
        if x not in unique_p: 
            unique_p.append(x)
            
    return unique_p

def uniformCrossover(chrm):
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

def singleCrossover(chrm):
    newChrm = []
    #crossover each bit randomly
    for i in range(len(chrm)):
        for k in range(i+1,len(chrm)):
            
            pt = random.randint(1, len(chrm[i]))

            newChrm.append(chrm[i][:pt]+chrm[k][pt:])
    newChrm = unique(newChrm)
    return newChrm

#recombine uses the parents to make new children
def recombine(chrm):
    #newChrm = uniformCrossover(chrm)

    newChrm = singleCrossover(chrm)
    
                
    return newChrm

def mutate(chrm):
    newChrm = chrm

    for i in range(len(newChrm)):
        temp = newChrm[i]
        for j in range(len(temp)):
            creep = random.gauss(0,0.01)
            temp[j] = round(temp[j] + creep,3)
        
        newChrm[i] = temp
    return newChrm

def replace(children,parents):
    newpop = parents

    for c in children:
        newpop.append(c)

    newpop = sort(newpop)

    return newpop[:50]

def main():
    tData = pre_processing.pre_processing("data/machine.data")
    trainData = dataset.dataset(tData.getData())
    original=np.array(trainData.getTrainingSet(0))
    test = np.array(trainData.getTestSet(0))
    x = original[:,:-1]
    y = original[:,-1]
    xsize = x.shape #6 is input nodes Last column is correct_answer
    ysize = np.unique(y).shape
    
    ffn = FFN.FeedForwardNeuralNetwork(xsize[1], ysize[0], [xsize[1]]*2)
    
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
    
    t = 0
    
    #init population
    pop = []
    data = []

    for i in range(50):
        data.append([random.randint(-3, 3),random.randint(-3, 3),random.randint(-3, 3),random.randint(-3, 3)])
    
    pop.append(data)

    #init fitness
    fit = []
    fit.append(evalFit(pop[t]))

    while(evalFit(pop[t]) > .05):
        t += 1
        c = select(pop[t-1])
        cp = recombine(c)
        cpp = mutate(cp)
        newpop = replace(cpp,pop[t-1])
        pop.append(newpop)
        fit.append(evalFit(pop[t]))


    print(pop[t])
    print(fit[t])
    print("# of iterations: ",t)
    print("# of unique indiv from pop: ",len(unique(pop[t])))
    pop[t] = sort(pop[t])
    print(pop[t][:1])
    print("//////////")        

main()
