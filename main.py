import matplotlib.pyplot as plt
from utils.converge_rate import drawConvergeRate
from utils.error_bar import drawErrorBar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--converge_rate", action="store_true", help="draw converge rates of learning algorithms")
parser.add_argument("--error_bar", action="store_true",
                    help="draw error bars for converge rates of learning algorithms")
args = parser.parse_args()
probDict = {(0, 0): 0.4, (0, 1): 0.2, (1, 0): 0.2, (1, 1): 0.4}  # Distribution of private signals

'''
----------------------- Comparison of Converge Rate ----------------------------
'''

if args.converge_rate:
    drawConvergeRate(800, 400, probDict, "Epsilon Greedy")
    drawConvergeRate(800, 400, probDict, "FTL")
    drawConvergeRate(800, 400, probDict, "FPL2")
    drawConvergeRate(800, 400, probDict, "FPL4")
    drawConvergeRate(800, 400, probDict, "FPL8")
    drawConvergeRate(800, 400, probDict, "Hedge Algorithm 1")
    drawConvergeRate(800, 400, probDict, "Hedge Algorithm 2")
    plt.savefig("./results/converge_rate.png")

'''
----------------------- Draw Error Bars ----------------------------
'''

if args.error_bar:
    drawErrorBar(800, 400, probDict, "Epsilon Greedy")
    plt.savefig("./results/eps_greedy.png")
    drawErrorBar(800, 400, probDict, "FTL")
    plt.savefig("./results/FTL.png")
    drawErrorBar(800, 400, probDict, "FPL2")
    plt.savefig("./results/FPL2.png")
    drawErrorBar(800, 400, probDict, "FPL4")
    plt.savefig("./results/FPL4.png")
    drawErrorBar(800, 400, probDict, "FPL8")
    plt.savefig("./results/FPL8.png")
    drawErrorBar(800, 400, probDict, "Hedge Algorithm 1")
    plt.savefig("./results/hedge1.png")
    drawErrorBar(800, 400, probDict, "Hedge Algorithm 2")
    plt.savefig("./results/hedge2.png")
