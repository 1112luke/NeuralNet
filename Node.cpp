#include "Node.hpp"
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
    }

    // initialize bias
    bias = Mathfunctions::randnum(-1, 1);
}

void Node::feedforward(vector<float> inputs)
{
    val = Mathfunctions::sigmoid(Mathfunctions::dot(inputs, weights) + bias);
    // cout << "I'm A Node! val: " << val << endl;
}

void Node::say()
{
    cout << "Hi\n";
}