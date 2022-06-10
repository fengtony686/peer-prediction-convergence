# A Peer Prediction Simulator by Shi Feng, 2022
from cmath import nan
from hashlib import new
from traceback import print_tb
from turtle import color
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


'''
--------------------- Class of Game ----------------------------
'''


class informationElicitationGame:
    def __init__(self, probabilityDict, markovian=False):
        self.markovian = markovian
        self.probabilityDict = probabilityDict
        if self.markovian == False:
            self.agentNum = len(list(self.probabilityDict.keys())[0])
            self.optionNum = int(
                np.power(len(probabilityDict), 1 / self.agentNum))
        else:
            self.agentNum = len(list(self.probabilityDict[0].keys())[0])
            self.optionNum = int(
                np.power(len(probabilityDict[0]), 1 / self.agentNum))
        self.strategyNum = np.power(self.optionNum, self.optionNum)
        self.reports = np.array([[] for _ in range(self.agentNum)])
        self.options = np.array([[] for _ in range(self.agentNum)])
        self.signals = np.array([[] for _ in range(self.agentNum)])
        self.agents = []

    def setAgents(self, strategy):
        for _ in range(self.agentNum):
            self.agents.append(
                agent(strategy=strategy, optionNum=self.optionNum))

    def optionToReport(self, option, signal):
        report = (option % np.power(self.optionNum, self.optionNum - signal)
                  ) / np.power(self.optionNum, self.optionNum - signal - 1)
        return int(report)

    def generateSignals(self):
        if self.markovian == False:
            seed = np.random.rand()
            leftBound = 0
            for (signal, probability) in self.probabilityDict.items():
                rightBound = leftBound + probability
                if seed < rightBound and seed >= leftBound:
                    return signal
                else:
                    leftBound = rightBound
            return nan
        else:
            lastSignal = 0
            if self.signals.shape[1] != 0:
                for i in range(self.agentNum):
                    lastSignal += np.power(self.optionNum,
                                           i) * self.signals[i][-1]
            seed = np.random.rand()
            leftBound = 0
            for (signal, probability) in self.probabilityDict[int(lastSignal)].items():
                rightBound = leftBound + probability
                if seed < rightBound and seed >= leftBound:
                    return signal
                else:
                    leftBound = rightBound
            return nan

    def addReports(self, currentReport, currentOptions, currentSignals):
        self.reports = np.hstack((self.reports, np.array([currentReport]).T))
        # print(self.reports, currentReport)
        self.options = np.hstack((self.options, np.array([currentOptions]).T))
        self.signals = np.hstack((self.signals, np.array([currentSignals]).T))
        for index, i in enumerate(currentReport):
            self.agents[index].reports.append(i)
            self.agents[index].options.append(currentOptions[index])

    # Processing one round of game.
    def run(self):
        signals = self.generateSignals()
        currentOptions = []
        for i in range(self.agentNum):
            currentOptions.append(self.agents[i].chooseOption())
        currentReports = []
        for i in range(self.agentNum):
            currentReports.append(self.optionToReport(
                currentOptions[i], signals[i]))
        for i in range(self.agentNum):
            allStrategyPayoff = []
            for strategy in range(self.strategyNum):
                counterfactualReports = currentReports.copy()
                counterfactualReports[i] = self.optionToReport(
                    strategy, signals[i])
                allStrategyPayoff.append(
                    self.agreement(counterfactualReports, i))
            self.agents[i].possiblePayoffs = np.hstack(
                (self.agents[i].possiblePayoffs, np.array([allStrategyPayoff]).T))
        self.addReports(currentReports, currentOptions, signals)

    '''
    ------------------------ Here are the Mechanism Functions -----------------------
    '''

    def agreement(self, currentReport, currentAgentIndex):
        payoff = -1
        if (self.reports.shape[1] == 0):
            return 0
        for i in currentReport:
            if i == currentReport[currentAgentIndex]:
                payoff += 1
        payoff /= len(currentReport) - 1
        agreementTerm = 0
        for i in range(self.agentNum):
            if i != currentAgentIndex and self.reports[i][-1] == currentReport[currentAgentIndex]:
                agreementTerm += 1
        payoff -= (agreementTerm / (self.agentNum - 1))
        return payoff


'''
----------------------- Class of Agents --------------------------
'''


class agent:
    def __init__(self, strategy, optionNum):
        self.optionNum = optionNum
        self.strategy = strategy
        self.strategyNum = np.power(self.optionNum, self.optionNum)
        self.options = []
        self.reports = []
        self.strategyProbList = []
        self.possiblePayoffs = np.array([[] for _ in range(self.strategyNum)])

    def chooseOption(self):
        if self.strategy == "FPL2":
            accumulateRewards = np.array([])
            for i in range(self.strategyNum):
                accumulateRewards = np.append(accumulateRewards, np.sum(
                    self.possiblePayoffs[i]) + np.random.rand() * 2)
            return np.random.choice(np.flatnonzero(accumulateRewards == accumulateRewards.max()))
        elif self.strategy == "FTL":
            accumulateRewards = np.array([])
            for i in range(self.strategyNum):
                accumulateRewards = np.append(accumulateRewards, np.sum(
                    self.possiblePayoffs[i]))
            return np.random.choice(np.flatnonzero(accumulateRewards == accumulateRewards.max()))
        elif self.strategy == "FPL4":
            accumulateRewards = np.array([])
            for i in range(self.strategyNum):
                accumulateRewards = np.append(accumulateRewards, np.sum(
                    self.possiblePayoffs[i]) + np.random.rand() * 4)
            return np.random.choice(np.flatnonzero(accumulateRewards == accumulateRewards.max()))
        elif self.strategy == "FPL8":
            accumulateRewards = np.array([])
            for i in range(self.strategyNum):
                accumulateRewards = np.append(accumulateRewards, np.sum(
                    self.possiblePayoffs[i]) + np.random.rand() * 8)
            return np.random.choice(np.flatnonzero(accumulateRewards == accumulateRewards.max()))
        elif self.strategy == "Hedge Algorithm 2":
            if len(self.reports) == 0:
                for i in range(self.strategyNum):
                    self.strategyProbList.append(1 / self.strategyNum)
            else:
                beta = 1
                averageRewards = np.array([])
                for i in range(self.strategyNum):
                    averageRewards = np.append(averageRewards,
                                               self.possiblePayoffs[i][-1])
                totalAverage = np.sum([self.strategyProbList[i] * (np.exp(
                    beta * averageRewards[i])) for i in range(self.strategyNum)])
                newStrategyProbList = []
                for i in range(self.strategyNum):
                    newStrategyProbList.append(
                        self.strategyProbList[i] * (np.exp(beta * averageRewards[i])) / totalAverage)
                self.strategyProbList = newStrategyProbList
            randomSeed = np.random.rand()
            tmpTotal = 0
            for index, i in enumerate(self.strategyProbList):
                if tmpTotal <= randomSeed and tmpTotal + i > randomSeed:
                    return index
                else:
                    tmpTotal += i
        elif self.strategy == "Hedge Algorithm 1":
            if len(self.reports) == 0:
                for i in range(self.strategyNum):
                    self.strategyProbList.append(1 / self.strategyNum)
            else:
                beta = .5
                averageRewards = np.array([])
                for i in range(self.strategyNum):
                    averageRewards = np.append(averageRewards,
                                               self.possiblePayoffs[i][-1])
                totalAverage = np.sum([self.strategyProbList[i] * np.power(
                    np.sqrt(3), averageRewards[i]) for i in range(self.strategyNum)])
                newStrategyProbList = []
                for i in range(self.strategyNum):
                    newStrategyProbList.append(
                        self.strategyProbList[i] * (np.power(
                            np.sqrt(3), averageRewards[i])) / totalAverage)
                self.strategyProbList = newStrategyProbList
            randomSeed = np.random.rand()
            tmpTotal = 0
            for index, i in enumerate(self.strategyProbList):
                if tmpTotal <= randomSeed and tmpTotal + i > randomSeed:
                    return index
                else:
                    tmpTotal += i
        elif self.strategy == "Epsilon Greedy":
            accumulateRewards = np.array([])
            for i in range(self.strategyNum):
                accumulateRewards = np.append(accumulateRewards, np.sum(
                    self.possiblePayoffs[i]))
            eps = 1 / np.power((len(self.reports) + 1), 2)
            if np.random.random() > eps:
                return np.random.choice(np.flatnonzero(accumulateRewards == accumulateRewards.max()))
            else:
                return int(np.random.random() * self.strategyNum)


'''
----------------------- Converge Rate Drawing ---------------------------------
'''


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


'''
----------------------- Error Bars of Converge Rate Drawing ---------------------------------
'''


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


'''
----------------------- Comparison of Converge Rate ----------------------------
'''


probDict = {(0, 0): 0.4, (0, 1): 0.2, (1, 0): 0.2, (1, 1): 0.4}
drawConvergeRate(800, 400, probDict, "Epsilon Greedy")
drawConvergeRate(800, 400, probDict, "FTL")
drawConvergeRate(800, 400, probDict, "FPL2")
drawConvergeRate(800, 400, probDict, "FPL4")
drawConvergeRate(800, 400, probDict, "FPL8")
drawConvergeRate(800, 400, probDict, "Hedge Algorithm 1")
drawConvergeRate(800, 400, probDict, "Hedge Algorithm 2")


'''
----------------------- Draw Error Bars ----------------------------
'''


# probDict = {(0, 0): 0.4, (0, 1): 0.2, (1, 0): 0.2, (1, 1): 0.4}
# drawErrorBar(800, 400, probDict, "Epsilon Greedy")
# drawErrorBar(800, 400, probDict, "FTL")
# drawErrorBar(800, 400, probDict, "FPL2")
# drawErrorBar(800, 400, probDict, "FPL4")
# drawErrorBar(800, 400, probDict, "FPL8")
# drawErrorBar(800, 400, probDict, "Hedge Algorithm 1")
# drawErrorBar(800, 400, probDict, "Hedge Algorithm 2")


plt.show()
