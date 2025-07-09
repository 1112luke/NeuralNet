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
    vector<float> feedforward(vector<float>);
};

#endif