# Snake Reinforcement Learning (DQN)

This project implements the classic Snake game as a custom **Reinforcement Learning environment** and trains an agent using a **Deep Q-Network (DQN)**.

The goal is to explore core reinforcement learning concepts such as:
- State representation
- Reward design
- Exploration vs exploitation
- Experience replay
- Neural network function approximation

The project is written entirely in **Python**, using **Pygame** for the environment and **PyTorch** for the learning model.

---

## Status 
The agent is now fully connected end-to-end with the environment and training loop. Agent can be rendered and observed.

---

### Current Limitations
Agent frequently traps itself, leading to premature termination

---

### Next Milestone
Tune epsilon decay, learning rate, and reward strucutre to improve performance. 

---

## Learning Approach

The agent uses a **Deep Q-Network (DQN)**:
- A fully connected neural network approximates the Q-function
- Experience replay is implemented using a deque buffer
- Epsilon-greedy policy balances exploration and exploitation
- Bellman equation used for Q-value updates
