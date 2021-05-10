### Solution Walkthrough


#### The environment

The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

![](./images/tennis.png)

TThe observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

#### Algorithm

The algorithm is used in this implementation is [Multi-Agent Deep Deterministic Policy Gradient MADDPG](https://arxiv.org/abs/1706.02275). MADDPG is based on DDPG algorithm that I used in the prevision Continuous Control project which provided a good starting point for prototyping and developing the final solution.

img src="images/multi-agent-actor-critic.png" width="40%" align="top-left" alt="" title="Multi-Agent Actor-Critic" />

> Multi-agent decentralized actor with centralized critic ([Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)).

To explore the action space, the Ornstein-Uhlenbeck noise function is used. This is like the epsilon-greedy function we’ve used in previous projects, but with the added benefit of time correlation. In other words, the noise added at each timestep is correlated with previous noise inputs, so the actions tend to stay in the same direction for longer periods of time leading to more ‘continuous’ or smooth actions through space. 

A centralized replay buffer is used to enable both agents to learn from each other’s experiences. Samples are collected randomly from the buffer for each learning step. 

The Neural Networks implemented consist of the following;
•Actor network – 3 fully connected linear layers. Fc1 is 256 units, fc2 is 128 units, using RELU activation, with fc3 using the tanh activation (maps states to actions). 
•Critic Network – 3 fully connected linear layers. Fc1 is 256 units, fc2 is 128 units, using RELU activation, with fc3 outputting a single value with no activation (maps state/action pair to a Qvalue).

The implementation of this MADDPG algorithm leverages and adapts code from the previous Udacity lessons on policy-based methods and actor-critic methods.
