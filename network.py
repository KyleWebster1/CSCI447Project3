import math
import random

class Neuron:
    def __init__(self,layerNum,idNum, inputs):
        self.layer = layerNum
        self.id = idNum
        self.prev = []
        #w[0] will be the weight from this node to the next layers first node
        self.w = {}

        #this makes all the connections for hidden layers
        for i in inputs:
            self.prev.append(i)
            i.addNext(self)
            i.addWeight(self.getId(),random.random())
        self.next = []

    def input(self, i):
        return self.sigmoid(i)
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def addNext(self, n):
        self.next.append(n)

    def addWeight(self, i, w):
        self.w[i] = w

    def getWeight(self, i):
        return self.w[i]

    def getNext(self):
        return self.next

    def getPrev(self):
        return self.prev

    def getLayer(self):
        return self.layer

    def getId(self):
        return self.id
    
    def __str__(self):
        mStr = "Me: Layer: "+ str(self.getLayer()) + " ID: " + str(self.getId())
        pstr = "\nPrev Neurons: "
        pList = self.getPrev()
        for p in range(len(pList)):
            if(p == 0):
                pstr += "Layer: "+str(pList[p].getLayer())
            pstr += " ID: " + str(pList[p].getId())+" "
        
        nstr = "\nNext Neurons: "
        nList = self.getNext()
        for n in range(len(nList)):
            if(n == 0):
                nstr += "Layer: "+str(nList[n].getLayer())
            nstr +=  " ID: " + str(nList[n].getId())+" W: {:0.5f} ".format(self.getWeight(nList[n].getId()))
        return "{} {} {}".format(mStr,pstr, nstr)

def getOutput(network, vector):
    for n in range(len(network)):
        nodeOutput = []
        for j in range(len(network[n])):
            if n == 0:
                nodeOutput.append(network[n][j].input(vector[n][j]))
            else:
                sumInput = 0
                cNode = network[n][j]
                cPrev = cNode.getPrev()
                for p in cPrev:
                    sumInput += vector[n][p.getId()]*p.getWeight(cNode.getId())
                nodeOutput.append(network[n][j].input(sumInput))
        vector.append(nodeOutput)
    return vector[-1]

num_inputs = 4
num_hidden_layers = 2
num_hidden_nodes = 4
num_outputs = 3

network = []
#input node
inputLayer = []
for i in range(num_inputs):
    n = Neuron(0,i,[])
    inputLayer.append(n)
network.append(inputLayer)

#hidden node
for i in range(num_hidden_layers):
    if i == 0:
        firstHiddenLayer = []
        for j in range(num_inputs):
            n = Neuron(i+1,j,[network[0][j]])
            firstHiddenLayer.append(n)
        network.append(firstHiddenLayer)
    else:
        nextLayer = []
        for j in range(num_hidden_nodes):
            n2 = Neuron(i+1,j,network[i])
            nextLayer.append(n2)
        network.append(nextLayer)

#output node
outputLayer = []
nSize = len(network)
for i in range(num_outputs):
    oN = Neuron(nSize,i,network[nSize-1])
    outputLayer.append(oN)
network.append(outputLayer)

for l in network:
    for n in l:
        print(n)
    print("////////////////////////")

#input vector
vector = [[0,1,0,1]]
output = getOutput(network,vector)
print(output)

vector = [[1,1,1,1]]
output = getOutput(network,vector)
print(output)
            




    
