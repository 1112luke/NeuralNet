#ifndef TESTCLASS_H
#define TESTCLASS_H

#include <vector>
#include "Layer.hpp"

using namespace std;

class Neuralnet
{

public:
    vector<int> shape;
    vector<Layer> layers;

    Neuralnet(vector<int>);
    void printme();
    void printOutput();
    void Train(vector<vector<float>>, vector<vector<float>>, float, int, int);
    void Train(vector<vector<float>>, vector<vector<float>>, float, int, int, vector<vector<float>>, vector<vector<float>>, int);
    void signal(vector<vector<float>>, vector<vector<float>>, int);
    void batch(vector<vector<float>>, vector<vector<float>>);
    void save(string);
    void load(string);
    void analyze();
    vector<float> feedforward(vector<float>);
};

#endif