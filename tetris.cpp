#include "tetris.h"
#include <algorithm>
#include <array>
#include <vector>
#include <time.h>
#include <ctime>
#include <random>

Tetris::Tetris()
{
    setPieces();
    seed = std::time(0);
    std::mt19937 random(seed);
    rg = random;
    nextPiece = random() % 7;
    nextNextPiece = random() % 7;

    w1 = new double[5][7]();
    b1 = new double[5]();
    w2 = new double[5]();
    b2 = new double[1]();

    clone = false;
}

Tetris::~Tetris()
{
    if (!clone)
    {
        delete[] w1;
        delete[] b1;
        delete[] w2;
        delete[] b2;
    }
}

Tetris::Tetris(const Tetris &rhs)
{
    width = rhs.width;
    height = rhs.height;
    state = rhs.state;
    tops = rhs.tops;
    maxHeight = rhs.maxHeight;
    hold = rhs.hold;
    justHeld = rhs.justHeld;

    linesCleared = rhs.linesCleared;
    totalLinesCleared = rhs.totalLinesCleared;
    igScore = rhs.igScore;
    tetris = rhs.tetris;
    weights = rhs.weights;
    allPieces = rhs.allPieces;
    nextPiece = rhs.nextPiece;
    nextNextPiece = rhs.nextNextPiece;
    rg = rhs.rg;
    seed = rhs.seed;
    clone = true;

    w1 = rhs.w1;
    b1 = rhs.b1;
    w2 = rhs.w2;
    b2 = rhs.b2;
}

Tetris &Tetris::operator=(const Tetris &rhs)
{
    width = rhs.width;
    height = rhs.height;
    state = rhs.state;
    tops = rhs.tops;
    maxHeight = rhs.maxHeight;
    hold = rhs.hold;
    justHeld = rhs.justHeld;

    linesCleared = rhs.linesCleared;
    totalLinesCleared = rhs.totalLinesCleared;
    igScore = rhs.igScore;
    tetris = rhs.tetris;
    weights = rhs.weights;
    allPieces = rhs.allPieces;
    nextPiece = rhs.nextPiece;
    nextNextPiece = rhs.nextNextPiece;
    rg = rhs.rg;
    seed = rhs.seed;
    clone = true;

    w1 = rhs.w1;
    b1 = rhs.b1;
    w2 = rhs.w2;
    b2 = rhs.b2;
    return *this;
}

void Tetris::printState()
{
    // create bitset
    std::vector<std::bitset<32>> bits(10);
    std::bitset<32> curr;
    for (int i = 0; i < width; ++i)
    {
        curr = state[i];
        bits[i] = curr;
    }

    for (int i = height; i >= 0; --i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << bits[j][i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool Tetris::drop(int pieceNumber, int orientation, int xOffset)
{
    Piece *piece = &allPieces[pieceNumber][orientation];

    // returns true if successful drop
    int temp[4];
    int yOffset = 0;

    for (int i = 0; i < piece->width; ++i)
    {
        temp[i] = tops[xOffset + i] - piece->skirt[i];
        yOffset = std::max(yOffset, temp[i]);
    }

    int block = 0;
    for (int i = 0; i < piece->width; ++i)
    {
        // bitshift left, bottom is lsb
        block = piece->blocks[i];
        block = block << (yOffset + piece->skirt[i]);

        // use bitwise OR to place block
        state[xOffset + i] = state[xOffset + i] | block;

        // return false if too large
        if (state[xOffset + i] > 1048576)
        {
            return false;
        }

        // update tops
        tops[xOffset + i] = 32 - std::countl_zero(state[xOffset + i]);
    }

    // update max height
    maxHeight = *std::max_element(tops.begin(), tops.end());

    // check for cleared lines
    clearLines();

    return true;
}

void Tetris::clearLines()
{
    int count = 0;
    unsigned bitMask = 1;
    unsigned curr = 0;
    bool full = true;
    unsigned save;

    for (int row = 0; row < maxHeight; ++row)
    {
        // std::cout << "row: " << row << " ";
        full = true;
        for (int col = 0; col < width; ++col)
        {
            // if the row at the colu is a 0, break to next row
            if (((state[col] & (bitMask << row)) == 0))
            {
                full = false;
                break;
            }
        }

        if (full)
        {
            // moves rows down, each column at a time
            for (int col = 0; col < width; ++col)
            {
                // save rows below
                save = state[col] & ((bitMask << row) - 1);

                // remove the below roes
                state[col] = (state[col] >> (row + 1)) << (row);

                // restore save
                state[col] = state[col] | save;
            }

            // decrement row to revisit row;
            row--;

            // update maxHeight, count, and tops
            maxHeight--;
            count++;
            for (int i = 0; i < width; ++i)
            {
                tops[i] = 32 - std::countl_zero(state[i]);
            }
        }
    }

    totalLinesCleared += count;
    linesCleared = count;

    if (count == 4)
    {
        tetris = 1;
    }
    else
    {
        tetris = 0;
    }

    if (count > 0)
    {
        combo++;
    }
    else
    {
        combo = 0;
    }

    int scores[5] = {0, 1, 3, 5, 16};

    igScore += (scores[count]);
}

void Tetris::setWeights(std::vector<double> &newWeights)
{
    weights = newWeights;
}

void Tetris::setWeightsNN(std::vector<std::vector<double>> &w1_, std::vector<double> &b1_, std::vector<double> &w2_, std::vector<double> &b2_)
{
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            w1[i][j] = w1_[i][j];
        }
        b1[i] = b1_[i];
        w2[i] = w2_[i];
    }
    b2[0] = b2_[0];
}

void Tetris::printWeights()
{
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            std::cout << w1[i][j] << "\t";
        }
        std::cout << "\n";
    }

    for (int i = 0; i < 5; ++i)
    {
        std::cout << b1[i] << "\t";
    }
    std::cout << "\n";

    for (int i = 0; i < 5; ++i)
    {
        std::cout << w2[i] << "\t";
    }

    std::cout << "\n"
              << b2[0] << std::endl;
}

int Tetris::getIgScore()
{
    return igScore;
}

int Tetris::getTetris()
{
    return tetris;
}

double Tetris::calculateLinScore()
{
    // # of holes
    int holes = 0;
    for (int col = 0; col < width; ++col)
    {
        holes += tops[col] - std::popcount(state[col]);
    }

    // depth of holes
    int depth = 0;
    int curr;
    int ones;
    for (int col = 0; col < width; ++col)
    {
        curr = state[col];
        ones = std::popcount(state[col]);
        for (int row = 0; row < maxHeight; ++row)
        {
            if ((curr & 1) == 0)
            {
                depth += ones;
            }
            else
            {
                ones--;
            }
            curr >>= 1;
        }
    }

    // # of blocks in col0
    int col0 = std::popcount(state[0]);

    // bumpiness
    int bumpiness = 0;
    int diff = 0;

    for (int col = 0; col < width - 1; ++col)
    {
        diff = tops[col] - tops[col + 1];
        bumpiness += abs(diff);
    }

    // maxHeight, linesCleared

    std::array<int, 7> features = {-holes, -col0, -bumpiness, -maxHeight, -depth, linesCleared, tetris};

    double dot = 0;
    for (int i = 0; i < features.size(); ++i)
    {
        dot += (1.0 * features[i]) * weights[i];
    }
    return dot;
}

double Tetris::calculateNNScore()
{
    // # of holes
    int holes = 0;
    for (int col = 0; col < width; ++col)
    {
        holes += tops[col] - std::popcount(state[col]);
    }

    // depth of holes
    int depth = 0;
    int curr;
    int ones;
    for (int col = 0; col < width; ++col)
    {
        curr = state[col];
        ones = std::popcount(state[col]);3
        for (int row = 0; row < maxHeight; ++row)
        {
            if ((curr & 1) == 0)
            {
                depth += ones;
            }
            else
            {
                ones--;
            }
            curr >>= 1;
        }
    }

    // # of blocks in col0
    int col0 = std::popcount(state[0]);

    // bumpiness
    int bumpiness = 0;
    int diff = 0;

    for (int col = 0; col < width - 1; ++col)
    {
        diff = tops[col] - tops[col + 1];
        bumpiness += abs(diff);
    }

    // maxHeight, linesCleared

    std::array<int, 7> features = {-holes, -col0, -bumpiness, -maxHeight, -depth, linesCleared, tetris};

    // matmul1
    double intermediate1[5] = {0};
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            intermediate1[i] += features[j] * w1[i][j];
        }
        intermediate1[i] += b1[i];
    }

    // RELU
    for (int i = 0; i < 5; ++i)
    {
        if (intermediate1[i] < 0)
            intermediate1[i] = 0;
    }

    // matmul2
    double intermediate2 = 0;
    for (int i = 0; i < 5; ++i)
    {
        intermediate2 += intermediate1[i] * w2[i];
    }

    intermediate2 += b2[0];

    // RELU
    if (intermediate2 < 0)
        intermediate2 = 0;

    return intermediate2;
}

void Tetris::updateSeed(int offset)
{
    seed += offset;
    rg.seed(seed);
}

bool Tetris::nextStateLookAhead()
{
    std::pair<int, int> bestMove = nextStateHelper(1);
    if (bestMove.first == -1 or totalLinesCleared >= 1000)
    {
        // DEAD
        return false;
    }
    else if (bestMove.first == -2)
    {
        // hold piece
        if (hold == -1)
        {
            hold = nextPiece;
            nextPiece = nextNextPiece;
            nextNextPiece = rg() % 7;
        }
        else
        {
            int temp = hold;
            hold = nextPiece;
            nextPiece = temp;
        }
        justHeld = true;
        return true;
    }
    // printState();
    drop(nextPiece, bestMove.first, bestMove.second);
    // printState();
    nextPiece = nextNextPiece;
    nextNextPiece = rg() % 7;
    justHeld = false;

    // prune
    if (igScore < -20)
    {
        std::cout << totalLinesCleared << "\t" << tetris << " pruned" << std::endl;
        return false;
    }

    tempLinesCleared += linesCleared;
    maxHeight = 0;

    if (tempLinesCleared > 5)
    {
        int rand = rg() % 10;
        int randLines = rg() % 4;
        for (int i = 0; i < width; ++i)
        {
            state[i] <<= randLines;
            if (i != rand)
            {
                state[i] |= ((1 << randLines) - 1);
            }
            tops[i] = 32 - std::countl_zero(state[i]);
            if (tops[i] > maxHeight)
                maxHeight = tops[i];
        }
        tempLinesCleared = 0;
    }
    return true;
}

std::pair<double, double> Tetris::nextStateHelper(int depth)
{
    if (depth == 1)
    {
        double nextScore;
        double bestScore = -__DBL_MAX__;
        std::pair<int, int> bestMove;
        Tetris nextBoard = *this;
        bool successful = false;
        for (int orientation = 0; orientation < allPieces[nextPiece].size(); ++orientation)
        {
            for (int xOffset = 0; xOffset < width - allPieces[nextPiece][orientation].width + 1; ++xOffset)
            {
                if (nextBoard.drop(nextPiece, orientation, xOffset))
                {
                    nextBoard.justHeld = false;
                    nextScore = nextBoard.nextStateHelper(2).first;
                    if (nextScore > bestScore)
                    {
                        bestScore = nextScore;
                        bestMove = {orientation, xOffset};
                    }
                    nextBoard = *this;
                    successful = true;
                }
            }
        }
        if (hold != -1 && justHeld == false)
        {
            for (int orientation = 0; orientation < allPieces[hold].size(); ++orientation)
            {
                for (int xOffset = 0; xOffset < width - allPieces[hold][orientation].width + 1; ++xOffset)
                {
                    if (nextBoard.drop(hold, orientation, xOffset))
                    {
                        // puts next piece to hold
                        nextBoard.hold = nextBoard.nextPiece;
                        nextBoard.justHeld = true;
                        nextScore = nextBoard.nextStateHelper(2).first;
                        if (nextScore > bestScore)
                        {
                            bestScore = nextScore;
                            bestMove = {-2, -2};
                        }
                        nextBoard = *this;
                        successful = true;
                    }
                }
            }
        }
        else if (justHeld == false)
        {
            // hold piece
            nextBoard.hold = nextBoard.nextPiece;
            nextBoard.justHeld = true;
            nextScore = nextBoard.nextStateHelper(2).first;
            if (nextScore > bestScore)
            {
                bestScore = nextScore;
                bestMove = {-2, -2};
            }
            nextBoard = *this;
            successful = true;
        }
        if (!successful)
        {
            return {-1, -1};
        }
        return bestMove;
    }
    else
    {
        double bestScore = -__DBL_MAX__;
        double nextNextScore;
        Tetris nextNextBoard = *this;
        for (int orientation = 0; orientation < allPieces[nextNextPiece].size(); ++orientation)
        {
            for (int xOffset = 0; xOffset < width - allPieces[nextNextPiece][orientation].width + 1; ++xOffset)
            {
                nextNextBoard.drop(nextNextPiece, orientation, xOffset);
                nextNextBoard.tetris += tetris;
                nextNextBoard.linesCleared += linesCleared;
                nextNextScore = nextNextBoard.calculateNNScore();
                if (nextNextScore > bestScore)
                {
                    bestScore = nextNextScore;
                }
                nextNextBoard = *this;
            }
        }
        // if there is a block in hold, drop held block
        if (hold != -1 && justHeld == false)
        {
            for (int orientation = 0; orientation < allPieces[hold].size(); ++orientation)
            {
                for (int xOffset = 0; xOffset < width - allPieces[hold][orientation].width + 1; ++xOffset)
                {
                    nextNextBoard.drop(hold, orientation, xOffset);
                    nextNextBoard.tetris += tetris;
                    nextNextBoard.linesCleared += linesCleared;
                    nextNextScore = nextNextBoard.calculateNNScore();
                    if (nextNextScore > bestScore)
                    {
                        bestScore = nextNextScore;
                    }
                    nextNextBoard = *this;
                }
            }
        }
        return {bestScore, 0};
    }
}

void Tetris::setPieces()
{
    Piece iPiece0;
    iPiece0.width = 4;
    iPiece0.skirt = {0, 0, 0, 0};
    iPiece0.blocks = {1, 1, 1, 1};

    Piece iPiece1;
    iPiece1.width = 1;
    iPiece1.skirt = {0, 0, 0, 0};
    iPiece1.blocks = {0b1111, 0, 0, 0};

    Piece jPiece0;
    jPiece0.width = 3;
    jPiece0.skirt = {0, 0, 0, 0};
    jPiece0.blocks = {0b11, 0b1, 0b1, 0};

    Piece jPiece1;
    jPiece1.width = 2;
    jPiece1.skirt = {0, 2, 0, 0};
    jPiece1.blocks = {0b111, 0b1, 0, 0};

    Piece jPiece2;
    jPiece2.width = 3;
    jPiece2.skirt = {1, 1, 0, 0};
    jPiece2.blocks = {0b1, 0b1, 0b11, 0};

    Piece jPiece3;
    jPiece3.width = 2;
    jPiece3.skirt = {0, 0, 0, 0};
    jPiece3.blocks = {1, 0b111, 0, 0};

    Piece lPiece0;
    lPiece0.width = 3;
    lPiece0.skirt = {0, 0, 0, 0};
    lPiece0.blocks = {1, 1, 0b11, 0};

    Piece lPiece1;
    lPiece1.width = 2;
    lPiece1.skirt = {0, 0, 0, 0};
    lPiece1.blocks = {0b111, 1, 0, 0};

    Piece lPiece2;
    lPiece2.width = 3;
    lPiece2.skirt = {0, 1, 1, 0};
    lPiece2.blocks = {0b11, 1, 1, 0};

    Piece lPiece3;
    lPiece3.width = 2;
    lPiece3.skirt = {2, 0, 0, 0};
    lPiece3.blocks = {1, 0b111, 0, 0};

    Piece oPiece0;
    oPiece0.width = 2;
    oPiece0.skirt = {0, 0, 0, 0};
    oPiece0.blocks = {0b11, 0b11, 0, 0};

    Piece sPiece0;
    sPiece0.width = 3;
    sPiece0.skirt = {0, 0, 1, 0};
    sPiece0.blocks = {1, 0b11, 1, 0};

    Piece sPiece1;
    sPiece1.width = 2;
    sPiece1.skirt = {1, 0, 0, 0};
    sPiece1.blocks = {0b11, 0b11, 0, 0};

    Piece tPiece0;
    tPiece0.width = 3;
    tPiece0.skirt = {0, 0, 0, 0};
    tPiece0.blocks = {1, 0b11, 1, 0};

    Piece tPiece1;
    tPiece1.width = 2;
    tPiece1.skirt = {0, 1, 0, 0};
    tPiece1.blocks = {0b111, 1, 0, 0};

    Piece tPiece2;
    tPiece2.width = 3;
    tPiece2.skirt = {1, 0, 1, 0};
    tPiece2.blocks = {1, 0b11, 1, 0};

    Piece tPiece3;
    tPiece3.width = 2;
    tPiece3.skirt = {1, 0, 0, 0};
    tPiece3.blocks = {1, 0b111, 0, 0};

    Piece zPiece0;
    zPiece0.width = 3;
    zPiece0.skirt = {1, 0, 0, 0};
    zPiece0.blocks = {1, 0b11, 1, 0};

    Piece zPiece1;
    zPiece1.width = 2;
    zPiece1.skirt = {0, 1, 0, 0};
    zPiece1.blocks = {0b11, 0b11, 0, 0};

    std::vector<Piece> iPieces = {iPiece0, iPiece1};
    std::vector<Piece> jPieces = {jPiece0, jPiece1, jPiece2, jPiece3};
    std::vector<Piece> lPieces = {lPiece0, lPiece1, lPiece2, lPiece3};
    std::vector<Piece> oPieces = {oPiece0};
    std::vector<Piece> sPieces = {sPiece0, sPiece1};
    std::vector<Piece> tPieces = {tPiece0, tPiece1, tPiece2, tPiece3};
    std::vector<Piece> zPieces = {zPiece0, zPiece1};

    allPieces = {iPieces, jPieces, lPieces, oPieces, sPieces, tPieces, zPieces};
}