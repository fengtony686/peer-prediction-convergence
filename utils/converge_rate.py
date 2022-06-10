'''
----------------------- Converge Rate Drawing ---------------------------------
'''

import numpy as np
from game.game import informationElicitationGame
from tqdm import trange
import matplotlib.pyplot as plt 


def drawConvergeRate(numOfRounds, numOfRepeating, probabilityMatrix, algorithm, markov=False):
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
    plt.rcParams['figure.figsize'] = (25.0, 6.0)
    plt.plot(np.arange(numOfRounds), convergeTimes,
             label=algorithm)
    plt.xlabel("Round")
    plt.ylabel("Converge Proportion")
    plt.legend(loc="best")
    