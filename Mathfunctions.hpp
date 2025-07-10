#ifndef MATHFUNCTIONS_H
#define MATHFUNCTIONS_H

#include <vector>

using namespace std;

class Mathfunctions
{
public:
    Mathfunctions();
    static int add(int, int);
    static float randnum(float, float);
    static float dot(vector<float>, vector<float>);
    static vector<float> vectorsub(vector<float>, vector<float>);
    static vector<float> vectoradd(vector<float>, vector<float>);
    static float mag(vector<float>);
    static float cost(vector<float>, vector<float>);
    static float cost_prime(float, float);
    static float sigmoid(float);
    static float sigmoid_prime(float);
};

#endif