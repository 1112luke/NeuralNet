#ifndef NODE_H
#define NODE_H

class Layer;

#include <vector>

using namespace std;

class Node
{
public:
    float val;
    float bias;
    float z;
    vector<float> weights;

    // backpropogation
    vector<float> tot_weight_gradient;
    float tot_bias_gradient = 0;
    int num_examples_in_batch = 0;

    float error;

    Node(int);
    void feedforward(vector<float>);
    void initbatch();
    void backwards(Layer &, Layer &, int);
    void backwards(Layer &, float);
    void updateparams(float);
    void say();
};

#endif