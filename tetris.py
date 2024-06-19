import numpy as np
import random
import copy
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import timeit

class Piece():
    def __init__(self, skirt, heights):
        self.skirt = np.array(skirt)
        self.heights = np.array(heights)
        self.width = self.heights.size

class Board():
    def __init__(self):
        self.width = 10
        self.height = 20
        self.state = np.zeros((self.height, self.width), dtype=int)
        self.tops = np.zeros(self.width, dtype=int)
        self.maxHeight = max(self.tops)
        self.linesCleared = 0
        self.totalLinesCleared = 0
        self.fitness = 0
        self.weights = np.zeros(7)
        self.features = np.zeros(7)
        self.score = 0
        self.setPieces()

    def printState(self):
        for i in self.state[::-1]:
            for j in i:
                print(j, end=' ')
            print()
        print()

    def drop(self, piece, xOffset):
        # returns true if succesful drop
        temp = []
        for i in range(piece.width):
            temp.append(self.tops[xOffset + i] - piece.skirt[i])

        tempMax = max(temp)
        for i in range(len(temp)):
            temp[i] = tempMax - temp[i]

        for i in range(len(temp)):
            temp[i] += self.tops[xOffset + i]

        for j in range(piece.width):
            for i in range(piece.heights[j]):
                if i + temp[j] >= self.height:
                    return False
                self.state[i + temp[j]][j + xOffset] = 1
            self.tops[xOffset + j] = temp[j] + piece.heights[j]

        # update maxHeight
        self.maxHeight = max(self.tops)

        # check for cleared lines
        self.clearLines()

        return True

    def clearLines(self):
        # checks for full lines, returns number of lines cleared
        count = 0
        i = 0

        while i < self.maxHeight:
            # check if lines are full
            if 0 not in self.state[i]:
                # move lines down
                self.state[i:-1] = self.state[i + 1:]
                self.state[-1] = np.zeros(self.width)
                # keep track of lines cleared
                count += 1
                # update other states
                self.maxHeight -= 1
                self.tops -= 1
            else:
                i += 1
        self.totalLinesCleared += count
        self.linesCleared = count

        scores = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}

        self.fitness += scores[count]

    def setPieces(self):
        iPiece1 = Piece([0], [4])
        iPiece2 = Piece([0, 0, 0, 0], [1, 1, 1, 1])
        jPiece1 = Piece([0, 0, 0], [2, 1, 1])
        jPiece2 = Piece([0, 2], [3, 1])
        jPiece3 = Piece([1, 1, 0], [1, 1, 2])
        jPiece4 = Piece([0, 0], [1, 3])
        lPiece1 = Piece([0, 0, 0], [1, 1, 2])
        lPiece2 = Piece([0, 0], [3, 1])
        lPiece3 = Piece([0, 1, 1], [2, 1, 1])
        lPiece4 = Piece([2, 0], [1, 3])
        oPiece1 = Piece([0, 0], [2, 2])
        sPiece1 = Piece([0, 0, 1], [1, 2, 1])
        sPiece2 = Piece([1, 0], [2, 2])
        tPiece1 = Piece([0, 0, 0], [1, 2, 1])
        tPiece2 = Piece([0, 1], [3, 1])
        tPiece3 = Piece([1, 0, 1], [1, 2, 1])
        tPiece4 = Piece([1, 0], [1, 3])
        zPiece1 = Piece([1, 0, 0], [1, 2, 1])
        zPiece2 = Piece([0, 1], [2, 2])

        iPieces = [iPiece1, iPiece2]
        jPieces = [jPiece1, jPiece2, jPiece3, jPiece4]
        lPieces = [lPiece1, lPiece2, lPiece3, lPiece4]
        oPieces = [oPiece1]
        sPieces = [sPiece1, sPiece2]
        tPieces = [tPiece1, tPiece2, tPiece3, tPiece4]
        zPieces = [zPiece1, zPiece2]

        # self.pieces -> piece -> orientation of piece
        self.pieces = [iPieces, jPieces, lPieces, oPieces, sPieces, tPieces, zPieces]

    def setWeights(self, weights):
        # assign weights passed in
        self.weights = weights

    def calcFeatures(self):
        # calculates the features:

        # number of holes and depth of wholes
        holes = 0
        holeDepths = 0
        # iterate through each col
        for i in range(self.width):
            curr = self.state[:self.tops[i], i]
            # find the zeros
            zeros = np.where(curr == 0)[0]
            # add the number of zeros
            holes += zeros.size
            # add the depth of the zeros, calculated by the self.tops[i] - index
            holeDepths += np.sum(self.tops[i] - zeros)

        # number of blocks(non zero) in column 0
        col0 = np.count_nonzero(self.state[:self.tops[0], 0])

        # bumpiness of the tops, sum and abs
        diff = np.diff(self.tops)
        bumps = np.sum(np.abs(diff))

        # max height
        maxHeight = self.maxHeight

        # lines cleared from this drop
        linesCleared = self.linesCleared

        # if tetris
        tetris = 1 if linesCleared == 4 else 0

        self.features =  [holes, holeDepths, col0, bumps, maxHeight, linesCleared, tetris]

    def calcScore(self):
        # calculates the score based off the weights and features
        self.score = np.dot(self.weights, self.features)

    def nextState(self):
        # finds the best next state and updates itself to it
        # returns False if game is over
        bestScore = -np.inf
        # select a random piece
        temp = random.randint(0, len(self.pieces) - 1)
        nextPiece = self.pieces[temp]
        # print(temp)

        alive = False

        for orientation in nextPiece:
            for offset in range(self.width - orientation.width + 1):
                # for each possible position, place and append to states
                nextBoard = copy.deepcopy(self)
                if not nextBoard.drop(orientation, offset):
                    continue
                alive = True
                nextBoard.calcFeatures()
                nextBoard.calcScore()
                if nextBoard.score > bestScore:
                    bestScore = nextBoard.score
                    bestState = nextBoard

        if alive:
            self.__dict__ = copy.deepcopy(bestState.__dict__)
        return alive

class GeneticAlgorithm():
    def __init__(self, children=16, mutationRate=0.2):
        self.mutationRate = mutationRate
        self.childrenWeights = []
        for i in range(children):
            self.childrenWeights.append(np.random.uniform(-1, 1, 7))
        self.generations = 0
        self.scores = []
        self.bestScore = 0

    def mutate(self):
        # generate new weights by mutation

        # mutate each into n/2 children
        newChildrenWeights = []
        for currWeights in self.parents:
            newChildrenWeights.append(currWeights)
            for i in range(int(len(self.childrenWeights) / 2) - 1):
                newWeights = currWeights + np.random.normal(0, self.mutationRate, 7)
                newWeights = np.clip(newWeights, -1, 1)
                newChildrenWeights.append(newWeights)

        self.childrenWeights = newChildrenWeights

    def play(self):
        self.generations += 1
        fitnessWeights = []
        with Pool(cpu_count()) as p:
            fitnessWeights.append(p.map(self.playHelper, self.childrenWeights))

        # take top 2 fitness and store their weight
        sortedWeights = sorted(fitnessWeights[0], key=lambda x: x[0], reverse=True)
        sortedWeights = np.array(sortedWeights)
        self.parents = sortedWeights[:2, 1:]
        self.scores.append(sortedWeights[0, 0])

        if sortedWeights[0, 0] > self.bestScore:
            self.bestScore = sortedWeights[0, 0]
            self.bestWeights = sortedWeights[0, 1:]

    def playHelper(self, weights):
        # returns [fitness,]
        board = Board()
        board.setWeights(weights)

        while board.nextState():
            pass

        return np.concatenate((([board.fitness]), (board.weights)))

if __name__ == '__main__':
    ga = GeneticAlgorithm(children=128)

    epochs = 500

    start = timeit.default_timer()

    for i in range(epochs):
        if i % 10 == 0:
            print("epoch: ", i)
        ga.play()
        ga.mutate()

    print("time: ", timeit.default_timer() - start)
    print("best score: ", ga.bestScore)
    print("weights: ", ga.bestWeights)
    print("most recent weights: ", ga.parents[0])


    # plt.plot(ga.scores)
    # plt.show()
