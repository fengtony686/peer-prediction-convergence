#include <iostream>
#include <math.h>
#include <tuple>
#include <list>
#include <dbg.h>

float computeExpectedPayoff(double signalDistribution[4], double myStrategy[2], double yourStrategy[2], double eps)
{
    double term1 = signalDistribution[0] * (myStrategy[0] * yourStrategy[0] + (1 - myStrategy[0]) * (1 - yourStrategy[0])) + signalDistribution[1] * (myStrategy[0] * (1 - yourStrategy[1]) + (1 - myStrategy[0]) * yourStrategy[1]) + signalDistribution[2] * (myStrategy[1] * (1 - yourStrategy[0]) + (1 - myStrategy[1]) * yourStrategy[0]) + signalDistribution[3] * (myStrategy[1] * yourStrategy[1] + (1 - myStrategy[1]) * (1 - yourStrategy[1]));

    double term2 = pow(signalDistribution[0] + signalDistribution[1], 2) * (pow(myStrategy[0], 2) + pow(1 - myStrategy[0], 2)) + pow(signalDistribution[2] + signalDistribution[3], 2) * (pow(myStrategy[1], 2) + pow(1 - myStrategy[1], 2)) + 2 * (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[2] + signalDistribution[3]) * (myStrategy[0] * (1 - myStrategy[1]) + (1 - myStrategy[0]) * myStrategy[1]);
    term2 *= eps;

    double term3 = (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[0] + signalDistribution[2]) * (myStrategy[0] * yourStrategy[0] + (1 - myStrategy[0]) * (1 - yourStrategy[0])) + (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[1] + signalDistribution[3]) * (myStrategy[0] * (1 - yourStrategy[1]) + (1 - myStrategy[0]) * yourStrategy[1]) + (signalDistribution[2] + signalDistribution[3]) * (signalDistribution[0] + signalDistribution[2]) * ((1 - myStrategy[1]) * yourStrategy[0] + myStrategy[1] * (1 - yourStrategy[0])) + (signalDistribution[2] + signalDistribution[3]) * (signalDistribution[1] + signalDistribution[3]) * ((1 - myStrategy[1]) * (1 - yourStrategy[1]) + myStrategy[1] * yourStrategy[1]);

    // return float(term1 - term2 - term3);
    return float(term1 - term2);
}

auto findNashEquilibrium(double signalDistribution[4], double eps = 0.1, int samples = 30)
{
    float payoffMatrix1[(samples + 1) * (samples + 1)][(samples + 1) * (samples + 1)];
    float payoffMatrix2[(samples + 1) * (samples + 1)][(samples + 1) * (samples + 1)];
#pragma omp parallel for
    for (int i = 0; i < (samples + 1) * (samples + 1); i++)
    {
        fprintf(stderr, "\rProcessing %5.2f%%", 100.0 * i / (samples + 1) / (samples + 1));
        for (int j = 0; j < (samples + 1) * (samples + 1); j++)
        {
            double myStrategy[2] = {double(int(i / (samples + 1))) / samples, double((i % (samples + 1))) / samples};
            double yourStrategy[2] = {double(int(j / (samples + 1))) / samples, double((j % (samples + 1))) / samples};
            payoffMatrix1[i][j] = computeExpectedPayoff(signalDistribution, myStrategy, yourStrategy, eps);
            double newSignalDistribution[4] = {signalDistribution[0], signalDistribution[2], signalDistribution[1], signalDistribution[3]};
            payoffMatrix2[i][j] = computeExpectedPayoff(newSignalDistribution, yourStrategy, myStrategy, eps);
        }
    }
    std::list<std::tuple<double, double, double, double>> nashEquilibriums;
    for (int j = 0; j < (samples + 1) * (samples + 1); j++)
    {
        for (int i = 0; i < (samples + 1) * (samples + 1); i++)
        {
            double maxPayoff1 = payoffMatrix1[0][j];
            for (int k = 0; k < (samples + 1) * (samples + 1); k++)
            {
                if (maxPayoff1 <= payoffMatrix1[k][j])
                {
                    maxPayoff1 = payoffMatrix1[k][j];
                }
            }
            double maxPayoff2 = payoffMatrix2[i][0];
            for (int k = 0; k < (samples + 1) * (samples + 1); k++)
            {
                if (maxPayoff2 <= payoffMatrix2[i][k])
                {
                    maxPayoff2 = payoffMatrix2[i][k];
                }
            }
            if (maxPayoff1 == payoffMatrix1[i][j] && maxPayoff2 == payoffMatrix2[i][j])
            {
                nashEquilibriums.emplace_back(std::make_tuple(double(int(i / (samples + 1))) / samples, double((i % (samples + 1))) / samples, double(int(j / (samples + 1))) / samples, double((j % (samples + 1))) / samples));
            }
        }
    }
    return nashEquilibriums;
}

int main()
{
    double signal[4] = {0.65, 0.1, 0.1, 0.15};
    auto result = findNashEquilibrium(signal, 1);
    dbg(result);
    return 0;
}
