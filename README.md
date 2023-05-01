# Peer Prediction for Learning Agents
[![](https://tokei.rs/b1/github/fengtony686/peer-prediction-convergence)](https://github.com/fengtony686/peer-prediction-convergence)    
This repository contains code of numerical experiments for paper:  
**Peer Prediction for Learning Agents**  
NeurIPS 2022  
[Shi Feng](https://fengshi.link), [Fang-Yi Yu](http://www-personal.umich.edu/~fayu/), [Yiling Chen](https://yiling.seas.harvard.edu/)  
[[ArXiv Version](https://arxiv.org/abs/2208.04433)]

## Usage
If you want to draw convergence rates of learning algorithms in CA mechanism, you need to run
```
python main.py --converge_rate
```

If you want to draw error bars for convergence rates of learning algorithms in CA mechanism, you need to run
```
python main.py --error_bar
```

Here is one of our experiment results:
![](https://github.com/fengtony686/peer-prediction-convergence/blob/main/results/converge_rate.png)

## File Hierarchy

```
.
├── game/                   # components in a peer prediction game
│   ├── agent.py            # implementing agents
│   └── game.py             # implementing signal generator and CA mechanism
├── utils/                  # drawing simulation results
│   ├── converge_rate.py
│   └── error_bar.py
├── results/                # our running samples
├── main.py                 # main file
├── .gitignore              # exclude some annoying files from git
├── LICENSE                 # MIT license
└── README.md               # what you are reading now
```

## Contact
If you have any questions, feel free to contact us through email (shifeng-thu@outlook.com) or Github issues. Enjoy!
