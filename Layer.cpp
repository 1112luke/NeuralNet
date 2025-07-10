#include "Layer.hpp"
#include "Node.hpp"
#include "Mathfunctions.hpp"
#include <iostream>
#include <vector>

Layer::Layer(int n, int prevn)
{
    for (int i = 0; i < n; i++)
    {
        nodes.push_back(Node(prevn));
    }
}

void Layer::printme()
{
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i].say();
    }
}

void Layer::feedforward(Layer &prev)
{
    // given previous layer, feedforward
    vector<float> inputs = prev.getNodeVals();

    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i].feedforward(inputs);
    }
}

void Layer::feedforward(vector<float> a)
{
    // vector version, first layer, copy input to values
    for (int i = 0; i < a.size(); i++)
    {
        nodes[i].val = a[i];
    }
}

void Layer::backwards(Layer &nextlayer, Layer &prevlayer)
{
    // middle layer
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i].backwards(nextlayer, prevlayer, i);
    }
}

void Layer::backwards(Layer &prevlayer, vector<float> y)
{
    // last layer
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i].backwards(prevlayer, y[i]);
    }
}

vector<float> Layer::getNodeVals()
{

    vector<float> out;

    for (int i = 0; i < nodes.size(); i++)
    {
        out.push_back(nodes[i].val);
    }

    return out;
}