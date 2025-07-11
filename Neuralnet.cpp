#include "Neuralnet.hpp"
#include "Mathfunctions.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <numeric>
#include <random>
#include <fstream>
#include <algorithm>
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
    int totalbatches = (int)(inputs.size() / batch_size);

    // run for epochs
    for (int l = 0; l < epochs; l++)
    {
        // create random indices list to start
        vector<int> indices(inputs.size());
        iota(indices.begin(), indices.end(), 0);

        // Use a random number generator
        random_device rd;
        default_random_engine g(rd());

        shuffle(indices.begin(), indices.end(), g);

        // loop through all batches
        for (int k = 0; k < totalbatches; k++)
        {
            vector<vector<float>> currdata;
            vector<vector<float>> currouts;

            for (int i = 0; i < batch_size; i++)
            {
                int idx = indices[i + (k * batch_size)];
                currdata.push_back(inputs[idx]);
                currouts.push_back(expected_outs[idx]);
            }

            //  run a batch -> get, getting cost for each example. output average cost and nudge in direction of negative gradient
            batch(currdata, currouts);

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

void Neuralnet::Train(vector<vector<float>> inputs, vector<vector<float>> expected_outs, float learning_rate, int batch_size, int epochs, vector<vector<float>> testdata, vector<vector<float>> testoutput, int signal_interval)
{
    // split training data into random batches
    int totalbatches = (int)(inputs.size() / batch_size);

    // run for epochs
    for (int l = 0; l < epochs; l++)
    {
        if (l % signal_interval == 0)
        {
            signal(testdata, testoutput, l);
        }

        // create random indices list to start
        vector<int> indices(inputs.size());
        iota(indices.begin(), indices.end(), 0);

        // Use a random number generator
        random_device rd;
        default_random_engine g(rd());

        shuffle(indices.begin(), indices.end(), g);

        // loop through all batches
        for (int k = 0; k < totalbatches; k++)
        {
            vector<vector<float>> currdata;
            vector<vector<float>> currouts;

            for (int i = 0; i < batch_size; i++)
            {
                int idx = indices[i + (k * batch_size)];
                currdata.push_back(inputs[idx]);
                currouts.push_back(expected_outs[idx]);
            }

            //  run a batch -> get, getting cost for each example. output average cost and nudge in direction of negative gradient
            batch(currdata, currouts);

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

void Neuralnet::signal(vector<vector<float>> testdata, vector<vector<float>> testoutput, int epoch)
{

    int total_correct = 0;
    int total = testdata.size();
    for (int i = 0; i < total; i++)
    {
        feedforward(testdata[i]);

        // calulate correctness
        int max = 0;
        for (int j = 0; j < layers[layers.size() - 1].nodes.size(); j++)
        {
            if (layers[layers.size() - 1].nodes[j].val > layers[layers.size() - 1].nodes[max].val)
            {
                max = j;
            }
        }

        if (testoutput[i][max] == 1)
        {
            total_correct++;
        }
    }
    cout << "Epoch " << epoch << ":" << endl;
    cout << "Total Correct: " << total_correct << "/" << total << endl;
    cout << "Percentage: " << ((float)total_correct) / ((float)total) << endl
         << endl;
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

void Neuralnet::save(string filename)
{

    ofstream my_file;
    my_file.open(filename);

    // output size
    for (int i = 0; i < shape.size(); i++)
    {
        my_file << shape[i] << ",";
    }

    my_file << endl;

    for (int i = 0; i < layers.size(); i++)
    {
        for (int j = 0; j < layers[i].nodes.size(); j++)
        {

            my_file << layers[i].nodes[j].bias << ",";

            for (int k = 0; k < layers[i].nodes[j].weights.size(); k++)
            {
                my_file << layers[i].nodes[j].weights[k] << ",";
            }
        }
    }
}

void Neuralnet::load(string filename)
{

    ifstream file(filename);
    string line;
    vector<float> shapevalues;
    vector<float> netvalues;

    int linesRead = 0;
    while (linesRead < 2 && getline(file, line))
    {
        stringstream ss(line);
        string token;

        while (getline(ss, token, ','))
        {
            float val = stof(token);
            linesRead == 0 ? shapevalues.push_back(val) : netvalues.push_back(val);
        }

        linesRead++;
    }

    // apply values to me
    if (shape.size() != shapevalues.size())
    {
        cout << "Invalid Shape!" << endl;
        return;
    }
    for (int i = 0; i < shape.size(); i++)
    {
        if (shape[i] != shapevalues[i])
        {
            cout << "Invalid Shape!" << endl;
            return;
        }
    }

    int idx = 0;
    for (int i = 0; i < layers.size(); i++)
    {
        for (int j = 0; j < layers[i].nodes.size(); j++)
        {

            // Load bias
            layers[i].nodes[j].bias = netvalues[idx];
            idx++;

            // Load weights
            for (int k = 0; k < layers[i].nodes[j].weights.size(); k++)
            {
                layers[i].nodes[j].weights[k] = netvalues[idx];
                idx++;
            }
        }
    }
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