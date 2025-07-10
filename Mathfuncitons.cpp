#include "Mathfunctions.hpp"
#include <ctime>
#include <vector>

using namespace std;

Mathfunctions::Mathfunctions()
{
    srand(time(0));
}

int Mathfunctions::add(int x, int y)
{
    return x + y;
}

float Mathfunctions::dot(vector<float> a, vector<float> b)
{
    float out = 0;

    for (int i = 0; i < a.size(); i++)
    {
        out += a[i] * b[i];
    }

    return out;
}

vector<float> Mathfunctions::vectorsub(vector<float> a, vector<float> b)
{
    vector<float> out;
    for (int i = 0; i < a.size(); i++)
    {
        out.push_back(a[i] - b[i]);
    }

    return out;
}

vector<float> Mathfunctions::vectoradd(vector<float> a, vector<float> b)
{
    vector<float> out;
    for (int i = 0; i < a.size(); i++)
    {
        out.push_back(a[i] + b[i]);
    }

    return out;
}

float Mathfunctions::mag(vector<float> a)
{
    float out = 0;
    for (int i = 0; i < a.size(); i++)
    {
        out += pow(a[i], 2);
    }

    return (float)sqrt(out);
}

float Mathfunctions::cost(vector<float> y, vector<float> a)
{
    return 0.5 * (1 / (float)y.size()) * pow(Mathfunctions::mag(Mathfunctions::vectorsub(y, a)), 2);
}

float Mathfunctions::cost_prime(float y, float a)
{
    return a - y;
}

float Mathfunctions::sigmoid(float z) // z is weights dot inputs + bias
{
    return 1 / (1 + exp(-z));
}

float Mathfunctions::sigmoid_prime(float z) // z is weights dot inputs + bias
{
    return sigmoid(z) * (1 - sigmoid(z));
}

float Mathfunctions::randnum(float low, float high)
{

    return (rand() / (float)RAND_MAX) * (high - low) + low;
}