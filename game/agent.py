'''
----------------------- Class of Agents --------------------------
'''

import numpy as np


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