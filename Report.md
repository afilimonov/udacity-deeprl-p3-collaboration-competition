### Solution Walkthrough


#### The environment

The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

![](./images/tennis.png)

TThe observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.


#### Solution components

##### Algorithm

The algorithm is used in this implementation is [Multi-Agent Deep Deterministic Policy Gradient MADDPG](https://arxiv.org/abs/1706.02275). MADDPG is based on DDPG agent that I used in the prevision Continuous Control project which provided a good starting point for prototyping and developing the final solution.

<img src="images/multi-agent-actor-critic.png" width="40%" align="top-left" alt="" title="Multi-Agent Actor-Critic" />

> _MADDPG decentralized actor with centralized critic [Lowe and Wu et al](https://arxiv.org/abs/1706.02275)._

The solutiion is based on decentralized actor with centralized critic architecture. It uses a single critic that receives as input the actions and state observations from all agents. This extra information makes training easier and allows for centralized training with decentralized execution i.e. each agents takes actions based on their own observations of the environment. 

##### Network architectues

[_Actor_](model.py) 
* First fully connected layer with input size 24 and output size 256
* Second fully connected layer with input size 256 and output size 128
* Third fully connected layer with input size 128 and output size 2


[_Critic_](model.py)
* First fully connected layer with input size 24 and output size 256
* Second fully connected layer with input size (256 + 2) = 258 and output size 128
* Third fully connected layer with input size 128 and output size 1
* Batch Normalization layer between first and second layers

I started with hidden layers sizes 512 and 384 for both actor and [384, 256] and ended up with [256, 128] model that converged reasonably well with with acceptable training time. I also tried using Batch Normalization for the actor network but it resulted in a worse traning performance.

##### OU Noise Function

To enable environment exploration especially at earlier 


Tranning results without noise decay

```
Episode 590, Average Score: 0.40, Max: 1.20, Min: 0.00, Avg: 0.57, Time: 4.66
Episode 600, Average Score: 0.41, Max: 0.90, Min: 0.00, Avg: 0.40, Time: 0.46
Episode 610, Average Score: 0.44, Max: 1.80, Min: 0.00, Avg: 0.57, Time: 8.51
Episode 620, Average Score: 0.44, Max: 1.40, Min: 0.10, Avg: 0.48, Time: 5.38
Episode 630, Average Score: 0.44, Max: 1.50, Min: 0.00, Avg: 0.47, Time: 1.65
Episode 640, Average Score: 0.46, Max: 1.40, Min: 0.00, Avg: 0.54, Time: 6.67


Environment solved in 648 episodes!	Moving Average Score: 0.503

Saving model ... done.
```

Test resluts without noise decay 
```
Total score (averaged over agents) this episode: 0.5450000083073974
Total score (averaged over agents) this episode: 0.1450000023469329
Total score (averaged over agents) this episode: 1.995000029914081
Total score (averaged over agents) this episode: 0.7950000129640102
Total score (averaged over agents) this episode: 0.8950000135228038
```


