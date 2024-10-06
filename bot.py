import mss
import pyautogui
import cv2
import time
from pynput.keyboard import Key, Controller
import numpy as np
import random
import copy
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import timeit

class Piece():
    def __init__(self, skirt, heights, rotation):
        self.skirt = np.array(skirt)
        self.heights = np.array(heights)
        self.width = self.heights.size
        self.rotation = rotation

class Board():
    def __init__(self):
        self.width = 10
        self.height = 20

        self.state = np.zeros((self.height, self.width), dtype=int)
        self.tops = np.zeros(self.width, dtype=int)
        self.maxHeight = 0
        self.linesCleared = 0
        self.totalLinesCleared = 0
        self.igScore = 0
        self.hold = -1
        self.justHeld = False
        self.prev = -1
        self.nextPiece = -1
        self.nextNextPiece = -1
        self.tetris = 0

        self.weights = None

        self.createPieces()

    def printState(self):
        for i in self.state[::-1]:
            for j in i:
                print(j, end=' ')
            print()
        print()

    def updateState(self, newState):
        self.state = newState
        self.tops = np.zeros(self.width, dtype=int)

        for i in range(self.height):
            for j in range(self.width):
                if self.state[i][j] == 1:
                    self.tops[j] = i + 1

        self.maxHeight = max(self.tops)

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
        # checks for full lines
        count = 0
        i = 0

        while i < self.maxHeight:
            # check if lines are full
            if 0 not in self.state[i] and 2 not in self.state[i]:
                # move lines down
                self.state[i:-1] = self.state[i + 1:]
                self.state[-1] = np.zeros(self.width)
                # keep track of lines cleared
                count += 1
            else:
                i += 1

        # update tops and maxheight
        self.tops = [0] * self.width
        for i in range(self.height):
            for j in range(self.width):
                if self.state[i][j] == 1:
                    self.tops[j] = i + 1

        self.maxHeight = max(self.tops)

        self.totalLinesCleared += count
        self.linesCleared = count

        if count == 4:
            self.tetris = 1
        else:
            self.tetris = 0

        # scores = {0: 0, 1: 100, 2: 300, 3: 500, 4: 1200}
        # scores = [0, 100, 200, 300, 500, 2400]
        scores = [0, 1, 3, 5, 20]

        self.igScore += scores[count]

    def createPieces(self):
        iPiece1 = Piece([0, 0, 0, 0], [1, 1, 1, 1], 0)
        iPiece1.color = 135
        iPiece2 = Piece([0], [4], 1)
        iPiece2.color = 135
        jPiece1 = Piece([0, 0, 0], [2, 1, 1], 0)
        jPiece1.color = 72
        jPiece2 = Piece([0, 2], [3, 1], 1)
        jPiece2.color = 72
        jPiece3 = Piece([1, 1, 0], [1, 1, 2], 2)
        jPiece3.color = 72
        jPiece4 = Piece([0, 0], [1, 3], 3)
        jPiece4.color = 72
        lPiece1 = Piece([0, 0, 0], [1, 1, 2], 0)
        lPiece1.color = 126
        lPiece2 = Piece([0, 0], [3, 1], 1)
        lPiece2.color = 126
        lPiece3 = Piece([0, 1, 1], [2, 1, 1], 2)
        lPiece3.color = 126
        lPiece4 = Piece([2, 0], [1, 3], 3)
        lPiece4.color = 126
        oPiece1 = Piece([0, 0], [2, 2], 0)
        oPiece1.color = 166
        sPiece1 = Piece([0, 0, 1], [1, 2, 1], 0)
        sPiece1.color = 142
        sPiece2 = Piece([1, 0], [2, 2], 1)
        sPiece2.color = 142
        tPiece1 = Piece([0, 0, 0], [1, 2, 1], 0)
        tPiece1.color = 95
        tPiece2 = Piece([0, 1], [3, 1], 1)
        tPiece2.color = 95
        tPiece3 = Piece([1, 0, 1], [1, 2, 1], 2)
        tPiece3.color = 95
        tPiece4 = Piece([1, 0], [1, 3], 3)
        tPiece4.color = 95
        zPiece1 = Piece([1, 0, 0], [1, 2, 1], 0)
        zPiece1.color = 93
        zPiece2 = Piece([0, 1], [2, 2], 1)
        zPiece2.color = 93

        iPieces = [iPiece1, iPiece2]
        jPieces = [jPiece1, jPiece2, jPiece3, jPiece4]
        lPieces = [lPiece1, lPiece2, lPiece3, lPiece4]
        oPieces = [oPiece1]
        sPieces = [sPiece1, sPiece2]
        tPieces = [tPiece1, tPiece2, tPiece3, tPiece4]
        zPieces = [zPiece1, zPiece2]

        # self.pieces -> piece -> orientation of piece
        self.pieces = [iPieces, jPieces, lPieces, oPieces, sPieces, tPieces, zPieces]

    def ReLU(self, Z):
        return np.clip(Z, 0, None)

    def calcScoreNN(self):
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

            ones = np.where(curr == 1)[0]

            # print(curr)
            for j in range(len(curr)):
                if curr[j] == 0:
                    holeDepths += len([x for x in ones if x > j])

        # number of blocks(non zero) in column 0
        col0 = np.count_nonzero(self.state[:self.tops[0], 0])

        # bumpiness of the tops, sum and abs
        diff = np.diff(self.tops)
        bumps = np.sum(np.abs(diff))

        features = [-holes, -col0, -bumps, -self.maxHeight, -holeDepths, self.linesCleared, self.tetris]

        W1, b1, W2, b2 = self.weights

        Z1 = np.dot(W1, features) + b1
        Y1 = self.ReLU(Z1)
        Z2 = np.dot(W2, Y1) + b2
        Y2 = self.ReLU(Z2)

        return Y2

    def nextStateLookAhead(self):
        # finds the best next state by looking at next 2 states
        # returns xOffset and rotation number
        # returns None if no more possible moves

        first, second = self.nextStateHelper(1)

        if first == -1:
            return False
        elif first == -2:
            self.justHeld = True
            if self.hold == -1:
                self.hold = self.nextPiece
                self.nextPiece = self.nextNextPiece
            else:
                temp = self.hold
                self.hold = self.nextPiece
                self.nextPiece = temp
            return -1, -1
        else:
            self.prev = self.nextPiece
            self.drop(first, second)
            self.nextPiece = self.nextNextPiece
            self.justHeld = False
            return first.rotation, second

    def nextStateHelper(self, depth):
        nextPiece = self.pieces[self.nextPiece]
        nextNextPiece = self.pieces[self.nextNextPiece]
        if self.hold != -1:
            # print(self.hold, self.nextPiece)
            hold = self.pieces[self.hold]
        else:
            hold = -1

        if depth == 1:
            bestScore = -np.inf
            successful = False
            for orientation in nextPiece:
                for offset in range(self.width - orientation.width + 1):
                    nextBoard = copy.deepcopy(self)
                    if nextBoard.drop(orientation, offset):
                        nextBoard.justHeld = False
                        successful = True
                        nextScore, dump = nextBoard.nextStateHelper(2)
                        if nextScore > bestScore:
                            bestScore = nextScore
                            bestMove = [orientation, offset]

            if self.hold != -1 and self.justHeld == False:
                for orientation in hold:
                    for offset in range(self.width - orientation.width + 1):
                        nextBoard = copy.deepcopy(self)
                        if nextBoard.drop(orientation, offset):
                            nextBoard.hold = self.nextPiece
                            nextBoard.justHeld
                            nextScore, dump = nextBoard.nextStateHelper(2)
                            successful = True
                            if nextScore > bestScore:
                                bestScore = nextScore
                                bestMove = [-2, -2]
            elif self.justHeld == False:
                nextBoard = copy.deepcopy(self)
                nextBoard.hold = self.nextPiece
                nextBoard.justHeld = True
                nextScore, dump = nextBoard.nextStateHelper(2)
                successful = True
                if nextScore > bestScore:
                    bestScore = nextScore
                    bestMove = [-2, -2]
            if not successful:
                return [-1, -1]
            return bestMove
        elif depth == 2:
            bestScore = -np.inf
            for orientation in nextNextPiece:
                for offset in range(self.width - orientation.width + 1):
                    nextNextBoard = copy.deepcopy(self)
                    nextNextBoard.drop(orientation, offset)
                    nextNextBoard.linesCleared += self.linesCleared
                    nextNextBoard.tetris += self.tetris
                    bestScore = max(nextNextBoard.calcScoreNN(), bestScore)

            if (self.hold != -1 and self.justHeld == False):
                for orientation in hold:
                    for offset in range(self.width - orientation.width + 1):
                        nextNextBoard = copy.deepcopy(self)
                        nextNextBoard.drop(orientation, offset)
                        nextNextBoard.linesCleared += self.linesCleared
                        nextNextBoard.tetris += self.tetris
                        bestScore = max(nextNextBoard.calcScoreNN(), bestScore)

            return [bestScore, 0]

class Bot():
    def __init__(self):
        self.board = Board()
        self.keyboard = Controller()
        self.delay = 0.5

    def loadWeights(self):
        with open("output/bestWeights.txt", 'r') as file:
            numbers = []
            for line in file:
                strNums = line.split()
                nums = [float(num) for num in strNums]
                numbers.append(nums)

        w1 = []
        b1 = []
        w2 = []
        b2 = []
        for i in range(5):
            curr = []
            for j in range(7):
                curr.append(numbers[i][j])
            w1.append(curr)

        b1 = numbers[5]
        w2 = numbers[6]
        b2 = numbers[7]

        self.board.weights = w1, b1, w2, b2

    def setWeights(self, weights):
        self.board.weights = weights

    def getState(self):
        w, h = pyautogui.size()
        monitor = {"top": 0, "left": 0, "width": w, "height": h}
        with mss.mss() as sct:
            img = sct.grab(monitor)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            # plt.imshow(small, cmap='gray')

            next = max(small[190:215, 233])
            nextNext = max(small[210, 375:420])

            bottomLeft = [588, 141]
            topRight = [183, 345]

            state = np.zeros((20, 10), dtype=int)

            xLoc = np.linspace(bottomLeft[1], topRight[1], 11, dtype=int)[:10]
            xLoc += int((xLoc[1] - xLoc[0]) / 2)

            yLoc = np.linspace(bottomLeft[0], topRight[0], 21, dtype=int)[:18]
            yLoc += int((yLoc[1] - yLoc[0]) / 2)

            for y in range(18):
                for x in range(10):
                    if small[yLoc[y], xLoc[x]] in [135, 72, 126, 166, 142, 95, 93, 153]:
                        # print(1, end='\t')
                        state[y, x] = 1
                    elif small[yLoc[y], xLoc[x]] == 106:
                        state[y, x] = 2
                    else:
                        state[y, x] = 0

        return next, nextNext, state

    def placeBlock(self, next, rotation, offset):

        pieceN = self.colorToIndex(next)

        # hold
        if offset == -1:
            self.keyboard.press('c')
            self.keyboard.release('c')
            time.sleep(np.random.uniform(0, self.delay))
            return

        outputOffset = offset - 3

        if rotation == 1 or pieceN == 3:
            outputOffset -= 1
            if pieceN == 0:
                outputOffset -= 1

        # print(pieceN, outputOffset, offset)

        for i in range(rotation):
            self.keyboard.press(Key.up)
            self.keyboard.release(Key.up)
            time.sleep(np.random.uniform(0, self.delay))

        if outputOffset < 0:
            for i in range(abs(outputOffset)):
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
                time.sleep(np.random.uniform(0, self.delay))
        else:
            for i in range(outputOffset):
                self.keyboard.press(Key.right)
                self.keyboard.release(Key.right)
                time.sleep(np.random.uniform(0, self.delay))

        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

    def colorToIndex(self, color):
        colorToPiece = {
            135: 0,
            72: 1,
            126: 2,
            166: 3,
            142: 4,
            95: 5,
            93: 6,
        }
        return colorToPiece[color]

    def play(self):

        # start game
        self.keyboard.press(Key.cmd)
        self.keyboard.press(Key.tab)
        self.keyboard.release(Key.tab)
        self.keyboard.release(Key.cmd)

        time.sleep(1)
        self.keyboard.press(Key.f4)
        self.keyboard.release(Key.f4)

        # wait for first piece to load
        while self.getState()[0] < 15:
            pass

        next, nextNext, state = self.getState()
        self.board.nextPiece = self.colorToIndex(next)
        self.board.nextNextPiece = self.colorToIndex(nextNext)
        output = self.board.nextStateLookAhead()

        # gameplay loop
        while True:
            # get next pieces
            # wait for piece to load
            while self.getState()[0] < 15:
                pass

            getColors = timeit.default_timer()

            next, nextNext, state = self.getState()
            # if dead
            if next == 106:
                print("game over")
                return


            if (state!=self.board.state).any():
                self.board.updateState(state)
                print("received pieces")
                self.board.printState()

            self.board.nextPiece = self.colorToIndex(next)
            self.board.nextNextPiece = self.colorToIndex(nextNext)

            print("getting colors:\t", timeit.default_timer() - getColors)

            getMoves = timeit.default_timer()
            # get next best move
            output = self.board.nextStateLookAhead()

            print("getting moves:\t", timeit.default_timer() - getMoves)

            if output is False:
                return

            playingMoves = timeit.default_timer()

            bestRotation, bestOffset = output

            self.placeBlock(next, bestRotation, bestOffset)

            print("playing moves:\t", timeit.default_timer() - playingMoves)
            print()

            self.board.printState()

if __name__ == "__main__":
    bot = Bot()
    bot.delay = 0.1
    bot.loadWeights()
    bot.play()