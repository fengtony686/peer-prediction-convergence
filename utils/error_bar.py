"""
----------------------- Error Bars of Converge Rate Drawing ---------------------------------
"""

import numpy as np
from tqdm import trange
from game.game import informationElicitationGame
import matplotlib.pyplot as plt


def drawErrorBar(numOfRounds, numOfRepeating, probabilityMatrix, algorithm, markov=False):
    sumConvergeTimes = np.zeros(numOfRounds)
    maxConvergeTimes = np.zeros(numOfRounds)
    minConvergeTimes = np.ones(numOfRounds) * numOfRepeating
    for _ in range(10):
        convergeTimes = np.zeros(numOfRounds)
        for i in trange(numOfRepeating):
            newGame = informationElicitationGame(probabilityMatrix, markov)
            newGame.setAgents(algorithm)
            for _ in range(numOfRounds):
                newGame.run()
            index = numOfRounds - 1
            if newGame.options[0][-1] != newGame.options[1][-1]:
                continue
            converge = True
            for j in range(numOfRounds):
                if newGame.options[0][index] == newGame.options[0][-1]:
                    convergeTimes[index] += converge
                else:
                    converge = False
                index -= 1
            converge = True
            for j in range(numOfRounds):
                if newGame.options[1][index] == newGame.options[1][-1]:
                    convergeTimes[index] += converge
                else:
                    converge = False
                index -= 1
        convergeTimes = convergeTimes / (2 * numOfRepeating)
        sumConvergeTimes += convergeTimes
        maxConvergeTimes = np.array(
            (maxConvergeTimes, convergeTimes)).max(axis=0)
        minConvergeTimes = np.array(
            (minConvergeTimes, convergeTimes)).min(axis=0)
    plt.rcParams['figure.figsize'] = (25.0, 6.0)
    plt.plot(np.arange(numOfRounds), sumConvergeTimes / 10,
             label=algorithm, color='red')
    plt.fill_between(np.arange(numOfRounds), maxConvergeTimes,
                     minConvergeTimes, color="#FF7256")
    plt.xlabel("Round")
    plt.ylabel("Converge Proportion")
    plt.legend(loc="best")
    