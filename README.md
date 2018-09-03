# Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications
## Daniel S. Brown and Scott Niekum

Inverse reinforcement learning (IRL) infers a reward function from demonstrations, allowing for policy improvement and generalization. 
However, despite much recent interest in IRL, little work has been done to understand of the minimum set of demonstrations needed to teach a specific sequential decision-making task. We formalize the problem of finding optimal demonstrations for IRL as a machine teaching problem where the goal is to find the minimum number of demonstrations needed to specify the reward equivalence class of the demonstrator. We extend previous work on algorithmic teaching for sequential decision-making tasks by showing an equivalence to the set cover problem, and use this equivalence to develop an efficient algorithm for determining the set of maximally-informative demonstrations. We apply our proposed machine teaching algorithm to two novel applications: benchmarking active learning IRL algorithms and developing an IRL algorithm that, rather than assuming demonstrations are i.i.d., uses counterfactual reasoning over informative demonstrations to learn more efficiently.

### Follow the instructions below to reproduce results in our [arXiv paper](https://arxiv.org/abs/1805.07687).



## Citations

```
@inproceedings{brown2018probabilistic,
     author = {Brown, Daniel S. and Niekum, Scott},
     title = {Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications},
     year = 2018,
     url={https://arxiv.org/abs/1805.07687}
}
```

  #### Dependencies
  - Matplotlib (for generating figures): https://matplotlib.org/users/installing.html
  - lpsolve (instructions for integration forthcoming)
  
  #### Getting started
  - Make a build directory: `mkdir build`
  - Make a data directory to hold results: `mkdir data`
  
  #### Comparison of Machine Teaching Algorithms (Table 1, in our [paper](https://arxiv.org/abs/1805.07687))
 <!-- - Use `make gridworld_basic_exp` to build the experiment.
  - Execute `./gridworld_basic_exp` to run. Data will be output to `./data/gridworld`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldBasicExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateGridWorldBasicPlots.py` to generate figures used in paper.
  - You should get something similar to the following two plots
<!--
<div>
  <img src="figs/boundAccuracy.png" width="350">
  <img src="figs/boundError.png" width="350">
</div>
  
  
