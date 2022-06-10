'''
--------------------- Class of Game ----------------------------
'''

import numpy as np
from game.agent import agent
from cmath import nan


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