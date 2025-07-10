#include "Node.hpp"
#include "Layer.hpp"
#include "Mathfunctions.hpp"
#include <iostream>
#include <vector>

using namespace std;

Node::Node(int prevn)
{
    // initialize weights
    for (int i = 0; i < prevn; i++)
    {
        weights.push_back(Mathfunctions::randnum(-1, 1));
        tot_weight_gradient.push_back(0);
    }

    // initialize bias
    bias = Mathfunctions::randnum(-1, 1);
}

void Node::feedforward(vector<float> inputs)
{
    z = Mathfunctions::dot(inputs, weights) + bias;
    val = Mathfunctions::sigmoid(z);
    // cout << "I'm A Node! val: " << val << endl;
}

void Node::initbatch()
{
    for (int i = 0; i < tot_weight_gradient.size(); i++)
    {
        tot_weight_gradient[i] = 0;
    }
    tot_bias_gradient = 0;
    num_examples_in_batch = 0;
}

void Node::backwards(Layer &nextlayer, Layer &prevlayer, int me)
{
    // middle layer
    float weightedsum = 0;
    // loop through all nodes in next layer, sumweights times their error
    for (int i = 0; i < nextlayer.nodes.size(); i++)
    {
        weightedsum += nextlayer.nodes[i].weights[me] * nextlayer.nodes[i].error;
    }

    error = weightedsum * Mathfunctions::sigmoid_prime(z);

    // accumulate total weight gradient
    for (int i = 0; i < tot_weight_gradient.size(); i++)
    {
        tot_weight_gradient[i] += error * prevlayer.nodes[i].val;
    }
    // accumulate bias gradient
    tot_bias_gradient += error;

    num_examples_in_batch++;
}

void Node::backwards(Layer &prevlayer, float y)
{
    // last layer
    error = Mathfunctions::cost_prime(y, val) * Mathfunctions::sigmoid_prime(z);

    // accumulate total weight gradient
    for (int i = 0; i < tot_weight_gradient.size(); i++)
    {
        tot_weight_gradient[i] += error * prevlayer.nodes[i].val;
    }
    // accumulate bias gradient
    tot_bias_gradient += error;

    num_examples_in_batch++;
}

void Node::updateparams(float rate)
{
    for (int i = 0; i < weights.size(); i++)
    {
        weights[i] += -rate * (tot_weight_gradient[i] / num_examples_in_batch);
    }

    bias += -rate * (tot_bias_gradient / num_examples_in_batch);
}

void Node::say()
{
    cout << "Node: " << endl;
    cout << "   Val: " << val << endl;
    cout << "   Bias: " << bias << endl;
    // cout << "   weights.size() = " << weights.size() << ", tot_weight_gradient.size() = " << tot_weight_gradient.size() << endl;
    cout << "   Weights: " << endl;
    for (int i = 0; i < weights.size(); i++)
    {
        cout << "       " << weights[i] << endl;
    }

    cout
        << endl;
}