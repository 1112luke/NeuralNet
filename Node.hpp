#ifndef NODE_H
#define NODE_H

#include <vector>

using namespace std;

class Node
{
public:
    float val;
    float bias;
    vector<float> weights;

    Node(int);
    void feedforward(vector<float>);
    void say();
};

#endif