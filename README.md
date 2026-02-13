# Offline Reinforcement Learning

## Project Overview

Offline Reinforcement Learning (Offline RL) learns policies from a fixed dataset, without interacting with the environment during training. This is useful when online interaction is costly, unsafe, or impractical.

However, removing exploration introduces major challenges, such as extrapolation error in value-based methods and a strong dependence on dataset composition and coverage. In particular, high-quality datasets collected from near-optimal or converged policies often lack sufficient state-action diversity and sub-optimal transitions, which are crucial for stable learning in Temporal Difference (TD)-based RL.

Modern approaches such as DQN+BC and TD3+BC address these issues by combining RL with Behavior Cloning (BC) regularization, merging RL and supervised learning in a deep learning setting.

This project investigates DQN+BC in the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment and:
- Separates BC pretraining from DQN+BC training;
- Studies the impact of state normalization;
- Confirms that replay buffer (RB) datasets collected during online training outperform final policy (FP) datasets collected from near-optimal policies, due to higher diversity, exploration, and the presence of sub-optimal transitions.

---

## Outline
- [Contributions](#contributions)
- [Datasets](#datasets)
- [State Normalization Techniques](#state-normalization-techniques)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations](#limitations)
- [How to Run](#how-to-run)


---

## Contributions
This project makes three main contributions to the study of offline DQN+BC and offline RL in general, focusing on training strategies, state representation, and dataset composition.

### Decoupled BC and DQN+BC Training
Standard DQN+BC implementations train the BC agent and Q-function together. This has its benefits but also creates some problems:
* __Pros__: It is simple, fast and does not require separate pipelines for training two separate agents;
* __Cons__: BC and RL have different training needs - RL benefits from small, noisy mini-batches, while BC requires larger batches and relatively stable loss gradients. This can make early BC updates unstable when trained jointly;

In this project, BC is trained first and then used in DQN+BC. This allows BC and RL to use different training strategies, such as separate pruning, early stopping, and batch sizes. Furtheremore, starting DQN+BC with a fully trained BC model improves stability and performance from the very first mini-batch, avoiding the initial “burn-in” period of joint training. The trade-off is a slightly higher computational cost and added implementational omplexity due to separate training pipelines.


### State Normalization Study:
Offline RL is sensitive to the distribution of states in the dataset, and feature normalization can help improve stability and performance. This project evaluates five state normalization methods: raw (no normalization), max-abs, min-max, robust, and standard (z-score).

The goal is to test whether normalizing states improves performance and to identify which normalization techniques work best in offline DQN+BC.


### Confirm Dataset Quality Diversity Trade-off
Prior works have shown that RB datasets, which contain diverse state-action pairs, suboptimal transitions and reflect more "exploratory" behavior, often outperform FP datasets collected from fully trained agents with mostly optimal actions.

This project tests this observation in the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment and provides insights into why dataset diversity and the presence of suboptimal transitions improve offline DQN+BC performance.

---

## Datasets
To test the hypotheses, each agent is evaluated across 10 variations: 5 normalization methods × 2 datasets. The datasets are:

* __Replay Buffer (RB) Dataset__ – Captures the agent’s learning experience throughout training in a typical online setting;

* __Final Policy (FP) Dataset__ – Generated using the fully trained DQN agent’s final policy;

> [!NOTE]
> The datasets used in this project were provided by my supervisor as part of the task description.


---

## State Normalization Techniques
In adition to the Raw state-action pairs the following 4 normalization techniques were systematically tested and evaluated for each agent variation:

- Min-Max [0,1] – Scales each feature to the [0,1] range:

$$x\prime = \frac{x - x_{min}}{x_{max} - x_{min}}$$

- Max-Abs[-1;1] - Scales features by their maximum absolute value. Preserves zeros and negative values:

$$x\prime = \frac{x}{|x_{max}|}$$


- Standard (z-score) - Centers features to mean 0 and standard deviation 1:

$$ x\prime= \frac{x - x_{mean}}{\sigma + \epsilon} $$

> [!Note] 
> To avoid division by 0 a small positive constant $\epsilon=10^{-6}$ was added to the denominator.

- Robust - Scales features using median and interquartile range (IQR). Makes the models less sensitive to outliers:

$$ x\prime = \frac{x-x_{median}}{x_{Q3} - x_{Q1}} $$



---

## Experimental Setup
TODO

---

## Results
TODO

---

## Limitations
TODO

---

## How to Run
TODO
```bash
git clone https://github.com/andonov16/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python train.py
```
---

## Author
- [Miroslav Andonov](https://github.com/andonov16)
