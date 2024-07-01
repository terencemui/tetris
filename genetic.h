#ifndef GENETIC_H
#define GENETIC_H

#include <random>

class Genetic
{
    private:
        int children = 256;
        double mutationRate = 0.05;
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<std::vector<std::vector<double>>>> weightsNN;
        std::vector<int> scores;
        std::mt19937 rg;
        std::uniform_real_distribution<> dis;
        std::uniform_real_distribution<> mutation;
    public:
        Genetic();
        void mutate(std::vector<std::vector<double>>&, std::vector<double>&, int);
        void mutateNN(std::vector<std::vector<std::vector<std::vector<double>>>>&, std::vector<std::vector<std::vector<double>>>&, int);
        void generateRandom(std::vector<std::vector<double>>&, int, int);
        void play(std::vector<std::vector<double>>&);
        void playNN(std::vector<std::vector<std::vector<std::vector<double>>>>&);
        void train(int);
        void trainNN(int);
        void test(std::vector<double>&);
        void testNN();

        bool resume();
        bool resumeNN();
        void dump(std::vector<std::vector<std::vector<double>>>&, std::vector<double>&);
};

#endif