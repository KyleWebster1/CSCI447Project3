# K-Nearest Neighbor Implementation Project
# Alexander Alvarez
# Matt Wintersteen
# Kyle Webster
# Greg Martin

import math
import FFN
import random
import time
import VectorUtilities
from operator import add
import sys

def minkowskiDistance(v1, v2, p):
    """
    generalized minkowski distance calculation
    :param v1: input vector 1
    :param v2: input vector 2
    :param p: value to raise the vectors by, distinguises between the norms
    :return: returns the distance between the vectors
    """
    if type(p) == str:
        maxDistance = 0
        for x in range(len(v1)):
            maxDistance = max(maxDistance, abs(v1[x] - v2[x]))
        return maxDistance
    else:
        distance = 0
        # assume: v1 and v2 are equal length
        for x in range(len(v1) - 1):
            distance = pow((abs(v1[x] - v2[x])), p)
        return pow(distance, 1.0 / p)


# randomize data so that when we select training and test sets, we get a variety of each class
def randomizeData(data):
    """
    Randomizes data for training
    :param data:
    :return:
    """
    randomSet = []
    copy = list(data)
    while len(randomSet) < len(data):
        index = random.randrange(len(copy))
        randomSet.append(copy.pop(index))
    return randomSet


# class containing methods implementing the K-NN algorithms
class k_nearest_neighbor:
    def __init__(self):
        pass

    # Percormes KNN to get distances and neighbors
    @staticmethod
    def knn(trainingSets, t, k):

        distances = []

        # calculate distances for each training set
        for x in range(len(trainingSets)):
            dist = minkowskiDistance(t, trainingSets[x], 2)
            distances.append((trainingSets[x], dist))

        # find k nearest neighbors
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # calculate class from neighbors using voting
    @staticmethod
    def getClass(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][len(neighbors[x]) - 1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        # sortedVotes[0] is the class with the most votes
        # sortedVotes[0][0] is the class sortedVotes[0][1] is the number of votes
        return sortedVotes[0][0]

    # calculate mean from neighbors
    @staticmethod
    def getMean(neighbors):
        total = 0.0
        for x in range(len(neighbors)):
            response = neighbors[x][len(neighbors[x]) - 1]
            total += response

        avg = float(total) / len(neighbors)

        return avg

    # edit training sets using test sets
    def condenseSets(self, trainingSets, testSets, k):

        condensedSets = []

        for n in range(len(trainingSets)):

            trainingSet = trainingSets[n]
            condensedSetBefore = []
            condensedSetAfter = []
            while (True):
                for i in range(len(trainingSet)):
                    x = trainingSet[i]
                    condensedSetBefore = condensedSetAfter
                    if (condensedSetBefore == []):
                        condensedSetAfter.append(x)
                        condensedSetBefore = condensedSetAfter
                    else:
                        if (len(condensedSetAfter) < k):
                            neighbors = self.knn(condensedSetAfter, x, 1)
                        else:
                            neighbors = self.knn(condensedSetAfter, x, k)
                        if (neighbors[0][len(neighbors[0]) - 1] != x[len(x) - 1]):
                            condensedSetAfter.append(x)
                            condensedSetBefore = condensedSetAfter
                if (condensedSetBefore == condensedSetAfter):
                    break

            condensedSets.append(condensedSetAfter)
        return condensedSets

    # Reducing dataset to centroids centered around the mean
    def kMeans(self, data, k):
        meansAfter = []
        u = []
        centroids = []
        change = 1
        for i in range(k):
            u.append(random.choice(data))
            centroids.append([])

        while change > .01:
            for i in range(k):
                centroids[i] = []

            for x in data:
                minDistance = None
                min = None
                for m in range(len(u)):
                    dist = math.sqrt(VectorUtilities.vector_magnitude_squared(VectorUtilities.vector_subtract(x,u[m])))
                    if minDistance == None:
                        minDistance = dist
                        min = m
                    elif dist < minDistance:
                        minDistance = dist
                        min = m
                centroids[min].append(x)

            change = 0
            for i in range(len(u)):
                average_vect = u[i]
                for x in range(len(centroids[i])):
                    average_vect = VectorUtilities.vector_add(average_vect,centroids[i][x])
                for j in range(len(average_vect)):
                    average_vect[j] /= len(centroids[i]) + 1
                change += minkowskiDistance(u[i], average_vect, 2)
                u[i] = average_vect


        return u, centroids

    # Function to determine the Medoids for K-Nearest Clustering
    def kMedoids(self, data, k):
        medoidsAfter = []
        for d in data:
            u = []
            change = 1
            for i in range(k):
                u.append(random.choice(d))
            while change > .1:
                centroids = {}
                for x in d:
                    minDistance = None
                    min = None
                    for m in u:
                        dist = minkowskiDistance(x, m, 2)
                        if minDistance == None:
                            minDistance = dist
                            min = m
                        elif dist < minDistance:
                            minDistance = dist
                            min = m
                    a = u.index(min)
                    try:
                        centroids[a].append(x)
                    except:
                        centroids.setdefault(a, [])
                        centroids[a].append(x)
                for i in u:
                    a = u.index(i)
                    try:
                        temp = centroids[a]
                    except:
                        del u[u.index(i)]
                    total = temp[0]
                    count = 1
                    for j in temp[1:]:
                        total = list(map(add, total, j))
                        count += 1
                    # print(total)
                    mean = [x / float(count) for x in total]
                    oldU = u
                    closestPoint = None
                    closestValue = 0
                    for i in temp:
                        if closestPoint == None:
                            closestPoint = i
                            closestValue = minkowskiDistance(i, mean, 2)
                        else:
                            distance = minkowskiDistance(i, mean, 2)
                            if distance < closestValue:
                                closestPoint = i
                                closestValue = distance
                    # Error detection
                    try:
                        u[u.index(i)] = closestPoint
                    except:
                        mean = mean
                comb = 0
                countC = 0
                for i in range(len(u)):
                    comb += minkowskiDistance(u[i], oldU[i], 2)
                    countC += 1
                change = comb / float(countC)
            medoidsAfter.append(u)
        return medoidsAfter
