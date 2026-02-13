# Offline Reinforcement Learning

## Project Overview

Offline Reinforcement Learning (Offline RL) learns policies from a fixed dataset, without interacting with the environment during training. This is useful when online interaction is costly, unsafe, or impractical.

However, removing exploration introduces major challenges, such as extrapolation error in value-based methods and a strong dependence on dataset composition and coverage. In particular, high-quality datasets collected from near-optimal or converged policies often lack sufficient state-action diversity and sub-optimal transitions, which are crucial for stable learning in TD-based RL.

Modern approaches such as DQN+BC and TD3+BC address these issues by combining RL with Behavior Cloning (BC) regularization, merging RL and supervised learning in a deep learning setting.

This project investigates DQN+BC in the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment environment and:
- Separates BC pretraining from DQN+BC training;
- Studies the impact of state normalization;
- Confirms that replay buffer (RB) datasets collected during online training outperform final policy (FP) datasets collected from near-optimal policies, due to higher diversity, exploration, and the presence of sub-optimal transitions.

---

## Outline
- [Contributions](#contributions)
- [Environment](#environment)
- [Datasets](#datasets)
- [Algorithms](#algorithms)
- [State Normalization Techniques](#state-normalization-techniques)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations](#limitations)
- [How to Run](#how-to-run)


---

## Contributions
TODO

---

## Environment
TODO

---

## Datasets
TODO

---

## Algorithms
TODO

---

## State Normalization Techniques
TODO
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
