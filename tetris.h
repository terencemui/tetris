#ifndef TETRIS_H
#define TETRIS_H

#include <iostream>
#include <array>
#include <vector>
#include <random>

struct Piece
{
    unsigned width;
    std::array<unsigned,4> skirt;
    std::array<unsigned, 4> blocks;
};

class Tetris
{
    public:
        int width = 10;
        int height = 20;
        std::array<unsigned, 10> state {0};
        std::array<int, 10> tops {0};
        int maxHeight = 0;
        int hold = -1;
        bool justHeld = false;

        int linesCleared = 0;
        int totalLinesCleared = 0;
        int tempLinesCleared = 0;
        int igScore = 0;
        int tetris = 0;
        int combo = 0;
        std::vector<double> weights {0};
        std::vector<std::vector<Piece>> allPieces;
        int nextPiece;
        int nextNextPiece;
        std::array<int, 3> nextPieces {0};
        std::mt19937 rg;
        int seed;
        bool clone;

        double(*w1)[7];
        double *b1;
        double *w2;
        double *b2;

    public:
        Tetris();
        ~Tetris();
        Tetris(const Tetris& rhs);
        Tetris& operator=(const Tetris& rhs);
        void printState();
        bool drop(int, int, int);
        void clearLines();
        void setPieces();
        int getIgScore();
        int getTetris();

        void setWeights(std::vector<double>&);
        void setWeightsNN
        (std::vector<std::vector<double>> &, std::vector<double>&, std::vector<double>&, std::vector<double>&);
        void printWeights();
        double calculateLinScore();
        double calculateNNScore();
        bool nextStateLookAhead();
        std::pair<double, double> nextStateHelper(int);
        void updateSeed(int);
};


#endif