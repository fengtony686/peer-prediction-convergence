# Two agents with binary signals
from random import sample
from signal import signal
from tqdm import trange
import numpy as np


def computeExpectedPayoff(signalDistribution, myStrategy, yourStrategy, eps):
    term1 = signalDistribution[0] * (myStrategy[0] * yourStrategy[0] + (1 - myStrategy[0]) * (1 - yourStrategy[0])) + signalDistribution[1] * (myStrategy[0] * (1 - yourStrategy[1]) + (1 - myStrategy[0]) * yourStrategy[1]) + \
        signalDistribution[2] * (myStrategy[1] * (1 - yourStrategy[0]) + (1 - myStrategy[1]) * yourStrategy[0]) + \
        signalDistribution[3] * (myStrategy[1] * yourStrategy[1] +
                                 (1 - myStrategy[1]) * (1 - yourStrategy[1]))

    term2 = np.power(signalDistribution[0] + signalDistribution[1], 2) * (np.power(myStrategy[0], 2) + np.power(1 - myStrategy[0], 2)) + np.power(signalDistribution[2] + signalDistribution[3], 2) * (np.power(myStrategy[1], 2) + np.power(
        1 - myStrategy[1], 2)) + 2 * (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[2] + signalDistribution[3]) * (myStrategy[0] * (1 - myStrategy[1]) + (1 - myStrategy[0]) * myStrategy[1])
    term2 *= eps

    term3 = (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[0] + signalDistribution[2]) * (myStrategy[0] * yourStrategy[0] + (1 - myStrategy[0]) * (1 - yourStrategy[0])) + (signalDistribution[0] + signalDistribution[1]) * (signalDistribution[1] + signalDistribution[3]) * (myStrategy[0] * (1 - yourStrategy[1]) + (1 - myStrategy[0]) * yourStrategy[1]) + (
        signalDistribution[2] + signalDistribution[3]) * (signalDistribution[0] + signalDistribution[2]) * ((1 - myStrategy[1]) * yourStrategy[0] + myStrategy[1] * (1 - yourStrategy[0])) + (signalDistribution[2] + signalDistribution[3]) * (signalDistribution[1] + signalDistribution[3]) * ((1 - myStrategy[1]) * (1 - yourStrategy[1]) + myStrategy[1] * yourStrategy[1])

    return term1 - term2 - term3


def findNashEquilibrium(signalDistribution, eps=0.1, samples=30):
    payoffMatrix1 = np.ones(
        shape=((samples+1)*(samples+1), (samples+1)*(samples+1)))
    payoffMatrix1 = payoffMatrix1 * (-999)
    payoffMatrix2 = np.ones(
        shape=((samples+1)*(samples+1), (samples+1)*(samples+1)))
    payoffMatrix2 = payoffMatrix2 * (-999)
    for i in trange((samples+1)*(samples+1)):
        for j in range((samples+1)*(samples+1)):
            payoffMatrix1[i][j] = computeExpectedPayoff(signalDistribution=signalDistribution, myStrategy=[
                                                        int(i/(samples+1))/samples, (i % (samples+1))/samples], yourStrategy=[int(j/(samples+1))/samples, (j % (samples+1))/samples], eps = eps)
            payoffMatrix2[i][j] = computeExpectedPayoff(signalDistribution=[signalDistribution[0], signalDistribution[2], signalDistribution[1], signalDistribution[3]], myStrategy=[
                                                        int(j/(samples+1))/samples, (j % (samples+1))/samples], yourStrategy=[int(i/(samples+1))/samples, (i % (samples+1))/samples], eps = eps)
    nashEquilibriums = []
    for j in trange((samples+1)*(samples+1)):
        for i in range((samples+1)*(samples+1)):
            if np.max(payoffMatrix1[:,j]) == payoffMatrix1[i][j] and np.max(payoffMatrix2[i]) == payoffMatrix2[i][j]:
                nashEquilibriums.append((int(i/(samples+1))/samples, (i%(samples+1))/samples, int(j/(samples+1))/samples, (j%(samples+1))/samples))

    return nashEquilibriums

print(findNashEquilibrium([0.8, 0.05, 0.05, 0.1]))
# signal = np.random.rand(4)
# print(computeExpectedPayoff(signal, [0.5, 0.5], np.random.rand(2), 0.1))
# print(computeExpectedPayoff(signal, [0.5, 0.5], np.random.rand(2), 0.1))