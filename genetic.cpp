#include "genetic.h"
#include "tetris.h"
#include <random>
#include <iostream>
#include <time.h>
#include <map>
#include <fstream>
#include <thread>
#include <future>
#include <fstream>

Genetic::Genetic()
{
    std::mt19937 _rg(std::time(0));
    std::uniform_real_distribution<> _dis(0.0, 1.0);
    std::uniform_real_distribution<> _mutation(-mutationRate, mutationRate);
    rg = _rg;
    dis = _dis;
    mutation = _mutation;
}

void Genetic::mutate(std::vector<std::vector<double>> &output, std::vector<double> &input, int n)
{
    std::vector<double> curr;
    double temp;
    double max = -1.0;
    curr.reserve(7);
    output.push_back(input);
    for (int i = 0; i < n - 1; ++i)
    {
        curr = {};
        for (int j = 0; j < 7; ++j)
        {
            temp = input[j] + mutation(rg);
            temp = std::max(0.0, temp);
            max = std::max(max, temp);
            curr.push_back(temp);
        }
        for (int j = 0; j < 7; ++j)
        {
            curr[j] = curr[j] / max;
        }
        output.push_back(curr);
    }
}

void Genetic::mutateNN(std::vector<std::vector<std::vector<std::vector<double>>>> &output, std::vector<std::vector<std::vector<double>>> &input, int n)
{
    std::vector<std::vector<std::vector<double>>> curr;
    std::vector<std::vector<double>> currCurr;
    std::vector<double> currCurrCurr;
    double temp;
    double max;
    output.push_back(input);
    output.reserve(output.size() + n);
    for (int child = 0; child < n - 1; ++child)
    {
        max = -1.0;
        curr = {};
        for (int i = 0; i < 4; ++i)
        {
            currCurr = {};
            for (int j = 0; j < input[i].size(); ++j)
            {
                currCurrCurr = {};
                for (int k = 0; k < input[i][j].size(); ++k)
                {
                    temp = input[i][j][k] + mutation(rg);
                    temp = std::max(0.0, temp);
                    max = std::max(max, temp);
                    currCurrCurr.push_back(temp);
                }
                currCurr.push_back(currCurrCurr);
            }
            curr.push_back(currCurr);
        }

        // normalize
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < input[i].size(); ++j)
            {
                for (int k = 0; k < input[i][j].size(); ++k)
                {
                    curr[i][j][k] /= max;
                }
            }
        }
        output.push_back(curr);
    }
}

void Genetic::generateRandom(std::vector<std::vector<double>> &output, int row, int col)
{
    std::vector<double> curr;
    curr.reserve(col);
    for (int i = 0; i < row; ++i)
    {
        curr = {};
        for (int j = 0; j < col; ++j)
        {
            curr.push_back(dis(rg));
        }
        output.push_back(curr);
    }
}

void Genetic::play(std::vector<std::vector<double>> &output)
{
    output = {};
    std::map<int, std::vector<double>> map;
    const int max_threads = 2;

    auto compute_score = [&](int i) -> std::pair<int, std::vector<double>>
    {
        i = 0;
        int rounds = 3;
        int worst = INT_MAX;
        for (int j = 0; j < rounds; ++j)
        {
            Tetris board;
            board.setWeights(weights[i]);
            board.updateSeed(j);
            while (board.nextStateLookAhead())
            {
            }
            if (board.getIgScore() < worst)
                worst = board.getIgScore();
            std::cout << board.seed << "\t" << board.igScore << "\t" << &board << std::endl;
        }
        return {-worst, weights[i]};
    };

    std::vector<std::future<std::pair<int, std::vector<double>>>> futures;
    for (int i = 0; i < children; i += max_threads)
    {
        int batch_size = std::min(max_threads, children - i);

        for (int j = 0; j < batch_size; ++j)
        {
            futures.push_back(std::async(std::launch::async, compute_score, i + j));
        }

        for (int j = 0; j < batch_size; ++j)
        {
            auto result = futures.back().get();
            futures.pop_back();
            map.insert(result);
        }
    }

    std::vector<std::vector<double>> bestWeights = {};

    std::ofstream scores("output/scores.txt", std::ios_base::app);
    std::ofstream weightsFile("output/weights.txt", std::ios_base::app);
    std::ofstream bestWeightsFile("output/bestWeights.txt");

    if (!weightsFile.is_open())
        std::cout << "error" << std::endl;

    std::cout << "scores: " << std::endl;
    int count = 0;
    for (const auto &[key, value] : map)
    {
        std::cout << -key << " ";
        scores << -key << " ";
        bestWeights.push_back(value);
        count++;
        if (count == 2)
        {
            break;
        }
    }
    std::cout << std::endl;
    weightsFile << "{";
    for (int i = 0; i < bestWeights[0].size(); ++i)
    {
        weightsFile << bestWeights[0][i] << ", ";
        bestWeightsFile << bestWeights[0][i] << " ";
    }
    weightsFile << "}" << std::endl;
    bestWeightsFile << std::endl;
    scores << std::endl;
    scores.close();
    weightsFile.close();
    output = bestWeights;
}

void Genetic::playNN(std::vector<std::vector<std::vector<std::vector<double>>>> &output)
{
    // output = {};
    // mapping scores to weights
    std::map<int, std::vector<std::vector<std::vector<double>>>> map;
    const int max_threads = std::thread::hardware_concurrency();

    // helper function to thread
    auto compute_score = [&](int i) -> std::pair<int, std::vector<std::vector<std::vector<double>>>>
    {
        int rounds = 1;
        int worst = INT_MAX;
        int sum = 0;
        // Tetris board;
        for (int j = 0; j < rounds; ++j)
        {
            // std::cout << i << "\t" << j << std::endl;
            Tetris board;
            board.setWeightsNN(weightsNN[i][0], weightsNN[i][1][0], weightsNN[i][2][0], weightsNN[i][3][0]);
            board.updateSeed(j);
            while (board.nextStateLookAhead())
            {
            }
            // sum += board.igScore;
            if (board.getIgScore() < worst)
                worst = board.getIgScore();
            // prune if score a 0
            if (worst == 0)
                return {-worst, weightsNN[i]};
        }
        return {-worst, weightsNN[i]};
    };

    std::vector<std::future<std::pair<int, std::vector<std::vector<std::vector<double>>>>>> futures;
    for (int i = 0; i < children; i += max_threads)
    {
        int batch_size = std::min(max_threads, children - i);

        std::cout << i << "/" << children << std::endl;

        for (int j = 0; j < batch_size; ++j)
        {
            futures.push_back(std::async(std::launch::async, compute_score, i + j));
        }

        for (int j = 0; j < batch_size; ++j)
        {
            auto result = futures.back().get();
            futures.pop_back();
            map.insert(result);
        }
    }

    // return top 2
    std::cout << "scores:\n";
    int count = 0;
    std::vector<double> scores;
    output = {};
    for (const auto &[key, value] : map)
    {
        std::cout << -key << "\t" << value[0][0][0] << std::endl;
        output.push_back(value);
        scores.push_back(key);
        count++;
        if (count == 2)
        {
            break;
        }
    }

    dump(output[0], scores);
}

void Genetic::train(int epochs)
{
    std::vector<std::vector<double>> random;
    std::vector<std::vector<double>> bestWeights;

    if (resume())
    {
        std::cout << "loaded old weights" << std::endl;
    }
    else
    {
        generateRandom(random, children, 7);
        weights = random;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < epochs; ++i)
    {
        std::cout << "epoch: " << i << std::endl;
        play(bestWeights);
        weights = {};

        if (bestWeights.size() == 1)
        {
            bestWeights.push_back(bestWeights[0]);
        }

        for (int j = 0; j < 2; ++j)
        {
            mutate(weights, bestWeights[j], children * 3 / 8);
        }
        generateRandom(weights, children / 4, 7);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "time elapsed: " << duration.count() << " seconds" << std::endl;
        start = std::chrono::steady_clock::now();
    }
}

void Genetic::trainNN(int epochs)
{
    std::vector<std::vector<std::vector<double>>> currChild;
    std::vector<std::vector<double>> random;
    // try to load, if not create children
    if (resumeNN())
    {
        std::cout << "loaded old weights" << std::endl;
    }
    else
    {
        std::cout << "no file found" << std::endl;
        for (int i = 0; i < children; ++i)
        {
            // create starting children
            currChild = {};
            random = {};
            generateRandom(random, 5, 7);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 5);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 5);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 1);
            currChild.push_back(random);
            weightsNN.push_back(currChild);
        }
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> bestWeights;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < epochs; ++i)
    {
        std::cout << "epoch: " << i << std::endl;
        playNN(bestWeights);
        weightsNN = {};

        if (bestWeights.size() == 1)
        {
            bestWeights.push_back(bestWeights[0]);
        }
        for (int j = 0; j < 2; ++j)
        {
            mutateNN(weightsNN, bestWeights[j], children * 3 / 8);
        }

        int remaining = children - weightsNN.size();
        for (int j = 0; j < remaining; ++j)
        {
            currChild = {};
            random = {};
            generateRandom(random, 5, 7);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 5);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 5);
            currChild.push_back(random);
            random = {};
            generateRandom(random, 1, 1);
            currChild.push_back(random);
            weightsNN.push_back(currChild);
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "time elapsed: " << duration.count() << " seconds" << std::endl;
        start = std::chrono::steady_clock::now();
    }
}

void Genetic::test(std::vector<double> &input)
{
    Tetris board;
    board.setWeights(input);

    while (board.nextStateLookAhead())
    {
        board.printState();
    }
    std::cout << "score: " << board.getIgScore() << std::endl;
    std::cout << "tetris: " << board.getTetris() << std::endl;
}

void Genetic::testNN()
{
    if (!resumeNN())
    {
        std::cout << "error loading file" << std::endl;
        return;
    }
    std::cout << "loaded files" << std::endl;
    Tetris board;
    board.setWeightsNN(weightsNN[0][0], weightsNN[0][1][0], weightsNN[0][2][0], weightsNN[0][3][0]);

    while (board.nextStateLookAhead())
    {
        board.printState();
    }
    std::cout << "score: " << board.getIgScore() << std::endl;
    std::cout << "tetris: " << board.getTetris() << std::endl;
}

bool Genetic::resume()
{
    std::ifstream input("output/bestWeights.txt");

    std::vector<double> temp;
    double curr;

    while (input >> curr)
    {
        temp.push_back(curr);
    }

    if (temp.size() == 0)
    {
        std::cout << "no weights found. creating new weights" << std::endl;
        return false;
    }

    mutate(weights, temp, children);
    return true;
}

bool Genetic::resumeNN()
{
    std::ifstream input("output/bestWeights.txt");

    std::vector<double> temp;
    double curr;
    while (input >> curr)
    {
        temp.push_back(curr);
    }

    input.close();

    std::vector<std::vector<std::vector<double>>> best;
    std::vector<std::vector<double>> bestBest;
    std::vector<double> bestBestBest;

    if (temp.size() == 0)
    {
        return false;
    }

    int count = 0;

    for (int i = 0; i < 5; ++i)
    {
        bestBestBest = {};
        for (int j = 0; j < 7; ++j)
        {
            bestBestBest.push_back(temp[count]);
            count++;
        }
        bestBest.push_back(bestBestBest);
    }
    best.push_back(bestBest);

    for (int i = 0; i < 2; ++i)
    {
        bestBest = {};
        bestBestBest = {};
        for (int j = 0; j < 5; ++j)
        {
            bestBestBest.push_back(temp[count]);
            count++;
        }
        bestBest.push_back(bestBestBest);
        best.push_back(bestBest);
    }
    best.push_back({{temp[count]}});

    mutateNN(weightsNN, best, children);

    return true;
}

void Genetic::dump(std::vector<std::vector<std::vector<double>>> &weights_, std::vector<double> &scores_)
{
    std::ofstream weightsFile("output/weightsNN.txt", std::ios_base::app);
    std::ofstream bestWeightsFile("output/bestWeights.txt");
    for (int i = 0; i < weights_.size(); ++i)
    {
        for (int j = 0; j < weights_[i].size(); ++j)
        {
            for (int k = 0; k < weights_[i][j].size(); ++k)
            {
                weightsFile << weights_[i][j][k] << " ";
                bestWeightsFile << weights_[i][j][k] << " ";
            }
            weightsFile << std::endl;
            bestWeightsFile << std::endl;
        }
    }
    weightsFile.close();
    bestWeightsFile.close();

    std::ofstream scoresFile("output/scoresNN.txt", std::ios_base::app);
    for (int i = 0; i < scores_.size(); ++i)
    {
        scoresFile << -scores_[i] << " ";
    }
    scoresFile << std::endl;
    scoresFile.close();
}