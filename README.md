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
  - Create a directory for data `mkdir -p data/algorithm_comparison`
  - Use `make algo_comp` to build the experiment.
  - Execute `./algo_comp scot random 1` to run scot algorithm. Data will be output to `./data/algorithm_comparison`
  - Execute `./algo_comp cakmak random 100000`
  - Execute `./algo_comp cakmak random 1000000`
  - Execute `./algo_comp cakmak random 10000000`
  - Experiments will take some time to run since it runs 20 replicates for each number of demonstrations. SCOT should be pretty fast, the main bottleneck is running IRL after SCOT is finished. 
  - Experiment parameters can be set in `src/machineTeachingAlgorithmComparison.cpp`. 
  - Once experiment has finished run `python scripts/makeTableComparingMachineTeachingAlgorithms.py` to generate table
  - You should get something similar to the following table. Note due to changes to seed and other parameters the results are slightly different from Table in paper.
  
   | Algorithm            | Ave. number of (s,a) pairs  | Ave. \% incorrect actions | Ave. time (s)|
| ------------------- |:-----:   | :----:   | :----:    | 
|SCOT | 17.250 | 0.185 | 1.274 |
| UVM (100000) | 3.450 | 37.531 | 951.479 |
| UVM (1000000) | 4.200 | 33.457 | 2096.638 |
| UVM (10000000) | 4.895 | 37.427 | 12068.098 |


  
  -The main thing to take away is that the UVM algorithm doesn't work well because it relies on MC volume estimation, whereas SCOT is fast and finds highly informative demonstration sets that allow IRL to find a good policy.
  
  
