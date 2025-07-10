#include <iostream>
#include <vector>
#include "Neuralnet.hpp"
#include "Mathfunctions.hpp"

using namespace std;

int main()
{

    vector<vector<float>> trainingdata;
    vector<vector<float>> trainingoutput;

    // generate lots of trainingdata
    for (int i = 0; i < 10; i++)
    {
        vector<float> newvec = {Mathfunctions::randnum(0, 1), Mathfunctions::randnum(0, 1), Mathfunctions::randnum(0, 1)};
        trainingdata.push_back(newvec);

        int max = 0;
        for (int j = 0; j < newvec.size(); j++)
        {
            if (newvec[j] > newvec[max])
            {
                max = j;
            }
        }
        vector<float> outvec;
        for (int j = 0; j < newvec.size(); j++)
        {
            if (j == max)
            {
                outvec.push_back(1);
            }
            else
            {
                outvec.push_back(0);
            }
        }
        trainingoutput.push_back(outvec);
    }

    cout << "output50: " << trainingdata[5][0] << " " << trainingdata[5][1] << " " << trainingdata[5][2] << endl;
    cout << "response50: " << trainingoutput[5][0] << " " << trainingoutput[5][1] << " " << trainingoutput[5][2] << endl;

    Neuralnet mynet({3, 3, 2, 8, 3});

    cout << "Pretrain: input data" << endl;
    // feedforward neuralnet

    mynet.feedforward({1, 0, 0});
    mynet.printOutput();
    cout << endl
         << endl;

    cout << "Training..." << endl
         << endl;

    mynet.Train(trainingdata, trainingoutput, 0.1, 10, 100000);

    cout << endl;

    cout << "Posttrain, training data: " << endl;
    // feedforward neuralnet
    mynet.feedforward({0.9, 0.5, 0.08});
    mynet.printOutput();
    cout << endl
         << endl;

    return 0;
}