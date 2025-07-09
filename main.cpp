#include <iostream>
#include <vector>
#include "Neuralnet.hpp"
#include "Mathfunctions.hpp"

using namespace std;

int main()
{

    Neuralnet mynet({10, 500, 300, 1000, 20, 2});

    // mynet.printme();

    vector<float> out = mynet.feedforward({0.1, 0.3, 0.36, 0.1, 0.6, 0.21, 0, 0.1, 1, 0.33});

    mynet.printOutput();

    return 0;
}