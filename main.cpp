#include <iostream>
#include <vector>
#include "Neuralnet.hpp"
#include "Mathfunctions.hpp"

using namespace std;

int main()
{

    vector<vector<float>> trainingdata;
    vector<vector<float>> trainingoutput;

    vector<vector<float>> testingdata;
    vector<vector<float>> testingoutput;

    // generate lots of trainingdata
    for (int i = 0; i < 1100; i++)
    {
        vector<float> newvec = {Mathfunctions::randnum(0, 1), Mathfunctions::randnum(0, 1), Mathfunctions::randnum(0, 1)};

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

        if (i < 1000)
        {
            trainingdata.push_back(newvec);
            trainingoutput.push_back(outvec);
        }
        else
        {
            testingdata.push_back(newvec);
            testingoutput.push_back(outvec);
        }
    }

    cout << "output50: " << trainingdata[5][0] << " " << trainingdata[5][1] << " " << trainingdata[5][2] << endl;
    cout << "response50: " << trainingoutput[5][0] << " " << trainingoutput[5][1] << " " << trainingoutput[5][2] << endl;

    Neuralnet mynet({3, 3, 2, 8, 3});

    // mynet.Train(trainingdata, trainingoutput, 0.1, 30, 3000, testingdata, testingoutput, 20);

    // mynet.save("bestnet.csv");
    mynet.load("bestnet.csv");

    mynet.feedforward({0.95, 0.817, 0.90});

    mynet.printOutput();

    return 0;
}