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
        self.nextPiece = self.pieces[random.randint(0, len(self.pieces) - 1)]
        self.nextNextPiece = self.pieces[random.randint(0, len(self.pieces) - 1)]

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

        # update next piece
        self.nextPiece = self.nextNextPiece
        self.nextNextPiece = self.pieces[random.randint(0, len(self.pieces) - 1)]

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

    def nextStateLookAhead(self):
        # finds the best next state by looking at next 2 states and updates itself to next state
        # returns False if no more possible moves

        bestScore, bestState = self.nextStateHelper(1)

        # if we we are still alive
        if bestState is not None:
            self.__dict__ = copy.deepcopy(bestState.__dict__)
            return True
        else:
            return False


    def nextStateHelper(self, depth):
        # recursive helper function for next state
        # returns the next best score and best state (-inf if no possible moves)
        bestScore = -np.inf
        bestState = None
        # bestChild = None

        for orientation in self.nextPiece:
            for offset in range(self.width - orientation.width + 1):
                # for each orientation and position, drop piece
                nextBoard = copy.deepcopy(self)
                # if the drop is not succesful, continue to next
                if not nextBoard.drop(orientation, offset):
                    continue

                # if leaf node
                if depth == 2:
                    nextBoard.calcFeatures()
                    nextBoard.calcScore()
                    if nextBoard.score > bestScore:
                        bestScore = nextBoard.score
                        bestState = nextBoard

                # elif parent, recursievley call on child
                elif depth == 1:
                    childScore, childState = nextBoard.nextStateHelper(2)
                    # if the child is best, than keep track of the nextBoard
                    if childScore > bestScore:
                        bestState = copy.copy(nextBoard)
                        bestScore = childScore
                        # bestChild = childState

        # if depth == 1:
        #     bestState.printState()
        #     bestChild.printState()

        return bestScore, bestState

class GeneticAlgorithm():
    def __init__(self, children=16, mutationRate=0.2):
        self.mutationRate = mutationRate
        self.childrenWeights = np.random.uniform(-1, 1, (children, 7))
        self.children = int(children)
        self.generations = 0
        self.scores = []
        self.bestScore = 0

    def mutate(self):
        # generate new weights by mutation

        # mutate each into n/4 children
        newChildrenWeights = []
        for currWeights in self.parents:
            newChildrenWeights.append(currWeights)
            for i in range(int(len(self.childrenWeights) / 4) - 1):
                newWeights = currWeights + np.random.normal(0, self.mutationRate, 7)
                newWeights = np.clip(newWeights, -1, 1)
                newChildrenWeights.append(newWeights)

        # remainig will be random new weights
        randomWeights = np.random.uniform(-1, 1, (self.children - len(newChildrenWeights), 7))
        newChildrenWeights = np.concatenate((newChildrenWeights, randomWeights))

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
        board.weights = weights

        while board.nextStateLookAhead():
        # while board.nextState():
            pass

        return np.concatenate((([board.fitness]), (board.weights)))

if __name__ == '__main__':
    ga = GeneticAlgorithm(children=64)

    epochs = 10

    start = timeit.default_timer()

    for i in range(epochs):
        # if i % 10 == 0:
        print("epoch: ", i)
        ga.play()
        ga.mutate()

    print("time: ", timeit.default_timer() - start)
    print("best score: ", ga.bestScore)
    print("weights: ", ga.bestWeights)
    print("most recent weights: ", ga.parents[0])


    plt.plot(ga.scores)
    plt.show()