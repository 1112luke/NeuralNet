#include "Neuralnet.hpp"
#include "Mathfunctions.hpp"
#include <iostream>
#include <vector>
using namespace std;

Neuralnet::Neuralnet(vector<int> inputshape)
{
    shape = inputshape;

    for (int i = 0; i < shape.size(); i++)
    {
        if (i == 0)
        {
            layers.push_back(Layer(shape[i], 0));
        }
        else
        {
            layers.push_back(Layer(shape[i], shape[i - 1]));
        }
    }
}

void Neuralnet::printme()
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i].printme();
    }
};

void Neuralnet::printOutput()
{
    for (int i = 0; i < layers[layers.size() - 1].nodes.size(); i++)
    {
        cout << layers[layers.size() - 1].nodes[i].val << endl;
    }
};

vector<float> Neuralnet::feedforward(vector<float> a)
{
    // return output if a is input

    // for each layer
    for (int i = 0; i < layers.size(); i++)
    {
        // feedforward based on previous layer
        if (i == 0)
        {
            layers[i].feedforward(a);
        }
        else
        {
            layers[i].feedforward(layers[i - 1]);
        }
    }
    return (layers[layers.size() - 1].getNodeVals());
}