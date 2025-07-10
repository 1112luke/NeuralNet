#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Node.hpp"

using namespace std;

class Layer
{
public:
    vector<Node> nodes;

    Layer(int, int);
    void printme();
    void feedforward(Layer &);
    void feedforward(vector<float>);
    void backwards(Layer &, Layer &);
    void backwards(Layer &, vector<float>);
    vector<float> getNodeVals();
};

#endif