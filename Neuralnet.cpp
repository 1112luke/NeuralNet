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

void Neuralnet::Train(vector<vector<float>> inputs, vector<vector<float>> expected_outs, float learning_rate, int batch_size, int epochs)
{
    // split training data into random batches

    // run for epochs
    for (int l = 0; l < epochs; l++)
    {

        // loop through all batches
        for (int k = 0; k < inputs.size() / batch_size; k++)
        {

            //  run a batch -> get, getting cost for each example. output average cost and nudge in direction of negative gradient
            batch(inputs, expected_outs); //**THIS IS WRONG, BUT DOING FOR NOW**

            // apply gradient based on learningrate
            for (int i = 1; i < layers.size(); i++)
            {
                for (int j = 0; j < layers[i].nodes.size(); j++)
                {
                    layers[i].nodes[j].updateparams(learning_rate);
                }
            }
        }
        // cout << endl << "Epoch " << l << " Complete!";
    }
}

void Neuralnet::batch(vector<vector<float>> examples, vector<vector<float>> expected_outs) // return index 0 = weight gradient, index 1 = bias gradient
{
    // initializebatch
    for (int i = 0; i < layers.size(); i++)
    {
        for (int j = 0; j < layers[i].nodes.size(); j++)
        {
            layers[i].nodes[j].initbatch();
        }
    }

    // backpropogate for each example in batch
    for (int i = 0; i < examples.size(); i++)
    {
        // run example through forwards -- updates vals
        feedforward(examples[i]);

        // perform backpropogation, each node keeping track of a cumulative weight and bias as well as times through to be averaged upon updateparams()
        for (int j = layers.size() - 1; j >= 1; j--)
        {
            // compute error for each layer
            vector<float> currer;
            if (j == layers.size() - 1)
            {
                layers[j].backwards(layers[j - 1], expected_outs[i]);
            }
            else
            {
                layers[j].backwards(layers[j + 1], layers[j - 1]);
            }
        }
    }
}

void Neuralnet::analyze()
{
    for (int i = 0; i < layers.size(); i++)
    {
        for (int j = 0; j < layers[i].nodes.size(); j++)
        {
            if (!layers[i].nodes[j].val)
            {
                cout << "NODE BAD: " << i << ", " << j << endl;
                cout << "Val: " << layers[i].nodes[j].val << endl;
                cout << "z: " << layers[i].nodes[j].z << endl;
            }
            for (int k = 0; k < layers[i].nodes[j].weights.size(); k++)
            {
                if (!layers[i].nodes[j].weights[k])
                {
                    cout << "WEIGHT BAD: " << i << ", " << j << ", " << k << endl;
                }
            }
        }
    }
    cout << "analyze done" << endl;
}

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